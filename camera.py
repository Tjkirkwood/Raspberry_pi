#!/usr/bin/python3
# Author Tyler K.

import time
import cv2
import os
import logging
import signal
import numpy as np

# Function to handle exit signals
def signal_handler(sig, frame):
    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

# Set up signal handling
signal.signal(signal.SIGINT, signal_handler)

# Ask the user for the time interval between snapshots
while True:
    try:
        snapshot_interval = int(input("Enter the time interval between snapshots (in seconds): "))
        if snapshot_interval <= 0:
            raise ValueError("The interval must be a positive integer.")
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter a positive integer.")

# Ask the user for the interval for test shots
while True:
    try:
        test_shot_interval = int(input("Enter the interval for test shots (in seconds): "))
        if test_shot_interval <= 0:
            raise ValueError("The interval must be a positive integer.")
        break
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter a positive integer.")

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Set automatic white balance and exposure
cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)  # Enable auto white balance
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure

time.sleep(2)  # Allow the camera to warm up

# Check if the camera opened correctly
if not cap.isOpened():
    logging.error("Cannot open camera")
    print("Error: Cannot open camera.")
    exit()

last_snapshot_time = 0  # Timestamp of the last snapshot
previous_frame = None  # To store the previous frame for motion detection
last_test_shot_time = 0  # Timestamp of the last test shot

# Function to take a snapshot
def take_snapshot(frame):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(picture_folder, f'snapshot_{timestamp}.jpg')
    cv2.imwrite(image_path, frame)

    if os.path.exists(image_path):  # Check if the image was saved
        print(f"Snapshot saved to: {image_path}")
    else:
        print(f"Failed to save snapshot to: {image_path}")

    logging.info(f"Snapshot taken and saved to {image_path}")
    return f"Snapshot saved to: {image_path}"  # Return the notification message

# Function to enhance image quality
def enhance_image(frame):
    # Denoise the image
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Histogram equalization on the Y channel of YUV color space
    yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Equalize only the Y channel
    enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Optionally, apply sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_frame, -1, kernel)

    return sharpened

# Function to adjust camera settings based on image quality
def adjust_settings(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 50:  # Too dark
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # Increase exposure
    elif brightness > 200:  # Too bright
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Decrease exposure

try:
    print("Press 'q' to quit the script.")

    # Take an initial test shot at the start
    ret, initial_frame = cap.read()
    if ret:
        notification = take_snapshot(initial_frame)
    last_test_shot_time = time.time()  # Update the last test shot time

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame.")
            print("Error: Failed to capture frame.")
            break

        # Enhance image automatically
        enhanced_frame = enhance_image(frame)

        # Adjust camera settings based on current frame quality
        adjust_settings(enhanced_frame)

        # Draw rectangles around detected faces for visualization
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(enhanced_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around faces

        # Get the current time for snapshot timing
        current_time = time.time()

        # Check for motion (not implemented in this version for simplicity)
        # Add your motion detection logic here if needed

        # Periodic test shots
        if current_time - last_test_shot_time >= test_shot_interval:
            notification = take_snapshot(enhanced_frame)
            last_test_shot_time = current_time  # Update the last test shot time

        # Optional: Display the notification on the camera feed
        if 'notification' in locals():
            cv2.putText(enhanced_frame, notification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            time.sleep(1)  # Display the notification for a short time
            del notification  # Clear the notification after displaying

        # Display the camera feed
        cv2.imshow("Camera Feed", enhanced_frame)

        # Quit the script if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting the script...")
            break

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"Error: {e}")

finally:
    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera feed closed.")
    print("Camera feed closed.")
