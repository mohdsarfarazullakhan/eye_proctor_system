import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Open webcam
cap = cv2.VideoCapture(0)

# Helper functions
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks):
    # 6-point eye for EAR
    A = euclidean_distance(landmarks[1], landmarks[5])
    B = euclidean_distance(landmarks[2], landmarks[4])
    C = euclidean_distance(landmarks[0], landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

EAR_THRESHOLD = 0.18
CONSEC_FRAMES = 3
closed_frames = 0
blink_count = 0
events = []

print("👁️ Eye monitoring started... Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])

        # Extract eyes
        left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

        # Draw eye landmarks
        for p in left_eye + right_eye:
            cv2.circle(frame, p, 1, (0, 255, 0), -1)

        # EAR for blink detection
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= CONSEC_FRAMES:
                blink_count += 1
                timestamp = time.strftime("%H:%M:%S")
                events.append({"event": "blink", "timestamp": timestamp})
                print(f"[{timestamp}] Blink detected (Total: {blink_count})")
            closed_frames = 0

        # Iris and gaze direction
        left_iris_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_IRIS]
        right_iris_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_IRIS]

        left_iris_center = np.mean(left_iris_pts, axis=0).astype(int)
        right_iris_center = np.mean(right_iris_pts, axis=0).astype(int)

        cv2.circle(frame, tuple(left_iris_center), 2, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_iris_center), 2, (255, 0, 0), -1)

        # Gaze direction estimation (simple)
        eye_center_x = np.mean([p[0] for p in left_eye])
        gaze_dir = "CENTER"
        if left_iris_center[0] < eye_center_x - 10:
            gaze_dir = "LEFT"
        elif left_iris_center[0] > eye_center_x + 10:
            gaze_dir = "RIGHT"

        cv2.putText(frame, f"Gaze: {gaze_dir}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if gaze_dir != "CENTER":
            timestamp = time.strftime("%H:%M:%S")
            events.append({"event": f"looking_{gaze_dir.lower()}", "timestamp": timestamp})
            print(f"[{timestamp}] Gaze detected: {gaze_dir}")

        # Display EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Eye Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Save logged events
df = pd.DataFrame(events)
df.to_csv("eye_events_log.csv", index=False)

print("\n✅ Eye monitoring ended.")
print(f"🧾 Logged events saved to 'eye_events_log.csv'")
