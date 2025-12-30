import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Landmark indices (DO NOT CHANGE)
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = 468
RIGHT_IRIS = 473

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        def to_pixel(idx):
            return np.array([
                int(landmarks[idx].x * w),
                int(landmarks[idx].y * h)
            ])

        # --- LEFT EYE ---
        left_corner1 = to_pixel(LEFT_EYE_CORNERS[0])
        left_corner2 = to_pixel(LEFT_EYE_CORNERS[1])
        left_iris = to_pixel(LEFT_IRIS)

        left_eye_center = (left_corner1 + left_corner2) // 2
        left_eye_width = np.linalg.norm(left_corner1 - left_corner2)

        left_gaze = (left_iris - left_eye_center) / left_eye_width

        # --- RIGHT EYE ---
        right_corner1 = to_pixel(RIGHT_EYE_CORNERS[0])
        right_corner2 = to_pixel(RIGHT_EYE_CORNERS[1])
        right_iris = to_pixel(RIGHT_IRIS)

        right_eye_center = (right_corner1 + right_corner2) // 2
        right_eye_width = np.linalg.norm(right_corner1 - right_corner2)

        right_gaze = (right_iris - right_eye_center) / right_eye_width

        # --- AVERAGED GAZE ---
        gaze = (left_gaze + right_gaze) / 2

        # --- DRAW VISUALS ---
        # Eye corners
        cv2.circle(frame, tuple(left_corner1), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_corner2), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_corner1), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_corner2), 2, (0, 255, 0), -1)

        # Iris centers
        cv2.circle(frame, tuple(left_iris), 3, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_iris), 3, (0, 0, 255), -1)

        # Eye centers
        cv2.circle(frame, tuple(left_eye_center), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 3, (255, 0, 0), -1)

        # Gaze vectors
        cv2.line(frame, tuple(left_eye_center), tuple(left_iris), (255, 255, 0), 2)
        cv2.line(frame, tuple(right_eye_center), tuple(right_iris), (255, 255, 0), 2)

        # Text output
        cv2.putText(
            frame,
            f"gaze_x: {gaze[0]:.2f}, gaze_y: {gaze[1]:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Gaze Debug", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
