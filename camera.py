#!/usr/bin/python3
# Author Tyler K.

import time
import cv2
import os
import logging
import signal

def signal_handler(sig, frame):
    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Get user input with validation
def get_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError("The interval must be a positive integer.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive integer.")

snapshot_interval = get_positive_integer("Enter the time interval between snapshots (in seconds): ")
test_shot_interval = get_positive_integer("Enter the interval for test shots (in seconds): ")

# Set up logging
log_file_path = os.path.expanduser('~/Pictures/security_camera.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Create a folder for pictures if it doesn't exist
picture_folder = os.path.expanduser('~/Pictures/security')
os.makedirs(picture_folder, exist_ok=True)

# Load the Haar Cascade for face detection
face_cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)

if not cap.isOpened():
    logging.error("Cannot open camera")
    print("Error: Cannot open camera.")
    exit()

last_snapshot_time = 0
previous_frame = None
last_test_shot_time = 0

def take_snapshot(frame):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(picture_folder, f'snapshot_{timestamp}.jpg')
    cv2.imwrite(image_path, frame)

    if os.path.exists(image_path):
        print(f"Snapshot saved to: {image_path}")
        logging.info(f"Snapshot taken and saved to {image_path}")
        return f"Snapshot saved to: {image_path}"
    else:
        print(f"Failed to save snapshot to: {image_path}")
        return "Failed to save snapshot"

try:
    print("Press 'q' to quit the script.")
    ret, initial_frame = cap.read()
    if ret:
        notification = take_snapshot(initial_frame)
    last_test_shot_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame.")
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            continue

        frame_diff = cv2.absdiff(previous_frame, gray)
        _, threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        motion_detected = cv2.countNonZero(threshold)
        current_time = time.time()

        if motion_detected > 500:  # Adjust based on environment
            if current_time - last_snapshot_time >= snapshot_interval:
                notification = take_snapshot(frame)
                last_snapshot_time = current_time

        if current_time - last_test_shot_time >= test_shot_interval:
            notification = take_snapshot(frame)
            last_test_shot_time = current_time

        # Reduce the number of face detections to improve performance
        if current_time - last_snapshot_time < snapshot_interval:  # Only detect faces if enough time has passed
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        previous_frame = gray

        if 'notification' in locals():
            cv2.putText(frame, notification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            time.sleep(1)
            del notification

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting the script...")
            break

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera feed closed.")
    print("Camera feed closed.")
