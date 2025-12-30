import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json

# ------------------ WebSocket ------------------

clients = set()

async def handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.remove(websocket)

# ------------------ Gaze Setup ------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = 468
RIGHT_IRIS = 473

screen_w, screen_h = 1280, 720
import json

with open("data/calibration.json", "r") as f:
    calib = json.load(f)

a = calib["a"]
b = calib["b"]
c = calib["c"]
d = calib["d"]

screen_w = calib["screen_w"]
screen_h = calib["screen_h"]


smoothed_x = None
smoothed_y = None
alpha = 0.2

def get_gaze(landmarks, w, h):
    def to_pixel(i):
        return np.array([int(landmarks[i].x * w), int(landmarks[i].y * h)])

    lc1, lc2 = to_pixel(33), to_pixel(133)
    li = to_pixel(468)
    left_center = (lc1 + lc2) / 2
    left_width = np.linalg.norm(lc1 - lc2)
    left_gaze = (li - left_center) / left_width

    rc1, rc2 = to_pixel(362), to_pixel(263)
    ri = to_pixel(473)
    right_center = (rc1 + rc2) / 2
    right_width = np.linalg.norm(rc1 - rc2)
    right_gaze = (ri - right_center) / right_width

    gaze = (left_gaze + right_gaze) / 2
    gaze[0] *= -1
    return gaze

# ------------------ Main Loop ------------------

async def gaze_loop():
    global smoothed_x, smoothed_y

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (screen_w, screen_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            gaze = get_gaze(landmarks, screen_w, screen_h)

            raw_x = a * gaze[0] + b
            raw_y = c * gaze[1] + d

            if smoothed_x is None:
                smoothed_x, smoothed_y = raw_x, raw_y
            else:
                smoothed_x = alpha * raw_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * raw_y + (1 - alpha) * smoothed_y

            norm_x = smoothed_x / screen_w
            norm_y = smoothed_y / screen_h

            norm_x = float(np.clip(norm_x, 0.0, 1.0))
            norm_y = float(np.clip(norm_y, 0.0, 1.0))

            if clients:
                msg = json.dumps({"nx": norm_x, "ny": norm_y})
                await asyncio.gather(*[c.send(msg) for c in clients])

        await asyncio.sleep(0.03)  # ~30 FPS

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Gaze WebSocket running on ws://localhost:8765")
        await gaze_loop()

asyncio.run(main())
