import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = 468
RIGHT_IRIS = 473

CALIB_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

calibration_data = []

def get_gaze(landmarks, w, h):
    def to_pixel(i):
        return np.array([int(landmarks[i].x * w), int(landmarks[i].y * h)])

    # Left eye
    lc1, lc2 = to_pixel(33), to_pixel(133)
    li = to_pixel(468)
    left_center = (lc1 + lc2) / 2
    left_width = np.linalg.norm(lc1 - lc2)
    left_gaze = (li - left_center) / left_width

    # Right eye
    rc1, rc2 = to_pixel(362), to_pixel(263)
    ri = to_pixel(473)
    right_center = (rc1 + rc2) / 2
    right_width = np.linalg.norm(rc1 - rc2)
    right_gaze = (ri - right_center) / right_width

    gaze = (left_gaze + right_gaze) / 2
    gaze[0] *= -1  # fix mirror inversion

    return gaze

screen_h, screen_w = 720, 1280
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", screen_w, screen_h)




for px, py in CALIB_POINTS:
    samples = []
    settle_time = 0.5      # time to move eyes to the dot
    capture_time = 1.5    # time to collect gaze samples
    start = time.time()

    while time.time() - start < (settle_time + capture_time):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (screen_w, screen_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            gaze = get_gaze(landmarks, screen_w, screen_h)
            if time.time() - start > settle_time:
                samples.append(gaze)

        # Draw calibration dot
        dot_x = int(px * screen_w)
        dot_y = int(py * screen_h)
        cv2.circle(frame, (dot_x, dot_y), 10, (0, 255, 0), -1)

        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    avg_gaze = np.mean(samples, axis=0)
    calibration_data.append((avg_gaze, (dot_x, dot_y)))
    time.sleep(0.3)

# --- COMPUTE MAPPING COEFFICIENTS ---

gaze_vals = np.array([g for g, _ in calibration_data])
screen_vals = np.array([s for _, s in calibration_data])

# Separate x and y
gx = gaze_vals[:, 0]
gy = gaze_vals[:, 1]
sx = screen_vals[:, 0]
sy = screen_vals[:, 1]

# Solve linear regression: y = a*x + b
a, b = np.polyfit(gx, sx, 1)
c, d = np.polyfit(gy, sy, 1)

print("Mapping coefficients:")
print(f"screen_x = {a:.2f} * gaze_x + {b:.2f}")
print(f"screen_y = {c:.2f} * gaze_y + {d:.2f}")

import json
import os

calib = {
    "a": float(a),
    "b": float(b),
    "c": float(c),
    "d": float(d),
    "screen_w": screen_w,
    "screen_h": screen_h
}

with open("data/calibration.json", "w") as f:
    json.dump(calib, f, indent=2)

print("Calibration saved to data/calibration.json")

cap = cv2.VideoCapture(0)
smoothed_x = None
smoothed_y = None
alpha = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_w, screen_h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        gaze = get_gaze(landmarks, screen_w, screen_h)

        # Apply calibration mapping
        raw_x = a * gaze[0] + b
        raw_y = c * gaze[1] + d

        if smoothed_x is None:
            smoothed_x = raw_x
            smoothed_y = raw_y
        else:
            smoothed_x = alpha * raw_x + (1 - alpha) * smoothed_x
            smoothed_y = alpha * raw_y + (1 - alpha) * smoothed_y

        screen_x = int(smoothed_x)
        screen_y = int(smoothed_y)


        # Clamp to screen
        screen_x = np.clip(screen_x, 0, screen_w)
        screen_y = np.clip(screen_y, 0, screen_h)

        # Draw cursor
        cv2.circle(frame, (screen_x, screen_y), 8, (0, 0, 255), -1)

    cv2.imshow("Calibrated Gaze Cursor", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

