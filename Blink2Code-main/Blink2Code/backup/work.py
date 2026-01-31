import cv2
import dlib
import numpy as np
import time
import os
import torch
from collections import deque
from cnn_model import BlinkCNN  # Import trained CNN model

# 🔹 Load trained CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlinkCNN(input_dim=8).to(device)
model.load_state_dict(torch.load("models/blink_cnn.pth", map_location=device))
model.eval()

# 🔹 Load dlib's face detector and facial landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    print(f"Error: Model file '{PREDICTOR_PATH}' not found! Download it from:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 🔹 Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# 🔹 Function to preprocess features for CNN
def preprocess_features(features):
    features = torch.tensor(features, dtype=torch.float32).to(device)
    return features.unsqueeze(0)  # Add batch dimension

# 🔹 Blink detection parameters (Adjusted for better detection)
BLINK_THRESHOLD = 0.25  # Increased threshold for better detection
BLINK_CONSEC_FRAMES = 1  # Reduced to detect faster blinks
# 🔹 Adjusted threshold for short blink duration
SHORT_BLINK_DURATION = 0.4  
blink_queue = deque(maxlen=BLINK_CONSEC_FRAMES)

# 🔹 Blink tracking variables
short_blink_count = 0
long_blink_count = 0
blink_start_time = None
is_blinking = False

# 🔹 Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam. Check your camera settings.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0  # Average EAR

        # Blink detection using EAR
        blink_queue.append(avg_EAR < BLINK_THRESHOLD)

        if sum(blink_queue) == BLINK_CONSEC_FRAMES and not is_blinking:
            is_blinking = True
            blink_start_time = time.time()

        if is_blinking and sum(blink_queue) == 0:
            blink_duration = time.time() - blink_start_time
            is_blinking = False

            # Skip CNN check temporarily to see if EAR works
            if blink_duration < SHORT_BLINK_DURATION:
                short_blink_count += 1
                print(f"👁 Short Blink Detected! Count: {short_blink_count}")
            else:
                long_blink_count += 1
                print(f"👁 Long Blink Detected! Count: {long_blink_count}")

            blink_queue.clear()  # Reset queue after detection

        # Draw eye landmarks
        for point in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        # Display EAR and blink counts
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Short Blinks: {short_blink_count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Long Blinks: {long_blink_count}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Blink Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
