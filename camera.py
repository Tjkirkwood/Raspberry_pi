#!/usr/bin/python3
# Author Tyler K.

import time
import cv2
import os
import logging
import signal
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

# Function to handle exit signals
def signal_handler(sig, frame):
    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

# Set up signal handling
signal.signal(signal.SIGINT, signal_handler)

# Create a simple GUI to get user input
def ask_for_intervals():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask for the snapshot interval
    snapshot_interval = simpledialog.askinteger("Input", "Enter the time interval between snapshots (in seconds):",
                                                 minvalue=1)
    if snapshot_interval is None:
        messagebox.showerror("Error", "You must enter a valid snapshot interval.")
        exit()

    # Ask for the test shot interval
    test_shot_interval = simpledialog.askinteger("Input", "Enter the interval for test shots (in seconds):",
                                                  minvalue=1)
    if test_shot_interval is None:
        messagebox.showerror("Error", "You must enter a valid test shot interval.")
        exit()

    return snapshot_interval, test_shot_interval

# Get user-defined intervals
snapshot_interval, test_shot_interval = ask_for_intervals()

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
time.sleep(2)  # Allow the camera to warm up

# Check if the camera opened correctly
if not cap.isOpened():
    logging.error("Cannot open camera")
    print("Error: Cannot open camera.")
    exit()

last_snapshot_time = 0  # Timestamp of the last snapshot
previous_frame = None  # To store the previous frame for motion detection
last_test_shot_time = 0  # Timestamp of the last test shot

# Function to enhance image quality
def enhance_image(frame):
    # Sharpen the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(frame, -1, kernel)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

    # Enhance contrast
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return enhanced_frame

# Function to take a snapshot
def take_snapshot(frame):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = os.path.join(picture_folder, f'snapshot_{timestamp}.jpg')

    # Enhance the image before saving
    enhanced_frame = enhance_image(frame)
    cv2.imwrite(image_path, enhanced_frame)

    if os.path.exists(image_path):  # Check if the image was saved
        print(f"Snapshot saved to: {image_path}")
    else:
        print(f"Failed to save snapshot to: {image_path}")

    logging.info(f"Snapshot taken and saved to {image_path}")
    return f"Snapshot saved to: {image_path}"  # Return the notification message

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

        # Convert the frame to grayscale for face detection and motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply Gaussian blur to reduce noise

        # Initialize previous_frame on the first run
        if previous_frame is None:
            previous_frame = gray
            continue  # Skip the first frame

        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(previous_frame, gray)
        _, threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels in the thresholded image to detect motion
        motion_detected = cv2.countNonZero(threshold)

        # Get the current time for snapshot timing
        current_time = time.time()

        # Check for motion
        if motion_detected > 500:  # Adjust this threshold based on your environment
            # If enough time has passed since the last snapshot
            if current_time - last_snapshot_time >= snapshot_interval:
                notification = take_snapshot(frame)
                last_snapshot_time = current_time  # Update the last snapshot time

        # Periodic test shots
        if current_time - last_test_shot_time >= test_shot_interval:
            notification = take_snapshot(frame)
            last_test_shot_time = current_time  # Update the last test shot time

        # Draw rectangles around detected faces for visualization
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around faces

        # Update the previous frame for motion detection
        previous_frame = gray

        # Optional: Display the notification on the camera feed
        if 'notification' in locals():
            cv2.putText(frame, notification, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            time.sleep(1)  # Display the notification for a short time
            del notification  # Clear the notification after displaying

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)

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
