import cv2
import dlib
import numpy as np
import time
import os
import torch
from collections import deque
from cnn_model import BlinkCNN  # Import trained CNN model
from morse_translator import blinks_to_morse, morse_to_text  # Import translation functions

# 🔹 Load trained CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlinkCNN(input_dim=8).to(device)
model.load_state_dict(torch.load("models/blink_cnn.pth", map_location=device))
model.eval()

# 🔹 Load dlib's face detector and facial landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    print(f"Error: Model file '{PREDICTOR_PATH}' not found!")
    print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 🔹 Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# 🔹 Blink detection parameters (Adjusted)
BLINK_THRESHOLD = 0.25  
BLINK_CONSEC_FRAMES = 1  
SHORT_BLINK_DURATION = 0.3  # Reduced from 0.4 sec
MORSE_GAP_DURATION = 1.5  # Increased from 1.0 sec

blink_queue = deque(maxlen=BLINK_CONSEC_FRAMES)
blink_sequence = []  
last_blink_time = time.time()
short_blink_count = 0
long_blink_count = 0
blink_start_time = None
is_blinking = False

# 🔹 File to save translations
OUTPUT_FILE = "translated_text.txt"

# 🔹 Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam. Check your camera settings.")
    exit()

# 🔹 Test the camera feed
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("Error: Camera is not capturing frames! Check your webcam.")
    cap.release()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Ensure the frame is properly loaded before processing
    if frame is None or frame.size == 0:
        print("Error: Frame is empty. Skipping...")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = face_detector(gray)

    if len(faces) == 0:
        print("No face detected, skipping frame.")
        continue

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        # Extract eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0  

        # Blink detection using EAR
        blink_queue.append(avg_EAR < BLINK_THRESHOLD)

        if sum(blink_queue) == BLINK_CONSEC_FRAMES and not is_blinking:
            is_blinking = True
            blink_start_time = time.time()

        if is_blinking and sum(blink_queue) == 0:
            blink_duration = time.time() - blink_start_time
            is_blinking = False

            # Classify blink as short or long
            if blink_duration < SHORT_BLINK_DURATION:
                short_blink_count += 1
                blink_sequence.append('S')
                print(f"👁 Short Blink Detected! Count: {short_blink_count}")
            else:
                long_blink_count += 1
                blink_sequence.append('L')
                print(f"👁 Long Blink Detected! Count: {long_blink_count}")

            last_blink_time = time.time()  

        # Detect Morse Code input based on time gap
        if len(blink_sequence) > 0 and (time.time() - last_blink_time) > MORSE_GAP_DURATION:
            morse_code = blinks_to_morse(blink_sequence)
            translated_text = morse_to_text(morse_code)

            print(f"🔡 Morse: {morse_code} → Text: {translated_text}")

            # Save to file
            try:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
                    file.write(f"Morse: {morse_code} → Text: {translated_text}\n")
            except Exception as e:
                print(f"Error writing to file: {e}")

            # Reset blink sequence after translation
            blink_sequence.clear()
            blink_queue.clear()  # Moved here to avoid clearing blinks too soon

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
        print("Exiting... Output saved to translated_text.txt")
        break

cap.release()
cv2.destroyAllWindows()
