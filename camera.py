#!/usr/bin/python3
# Author Tyler K.

import time
import cv2
import os
import logging
import signal
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

# Function to get user input for motion sensitivity, test shot interval, and snapshot interval
def get_user_settings():
    settings = {}

    # Get motion sensitivity
    while True:
        try:
            sensitivity = simpledialog.askinteger("Motion Sensitivity", 
                                                   "Enter the motion detection sensitivity (500-800):",
                                                   minvalue=500, maxvalue=800)
            if sensitivity is None:
                settings['motion_sensitivity'] = 500  # Default value
            else:
                settings['motion_sensitivity'] = sensitivity
            break
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))

    # Get test shot interval
    while True:
        try:
            interval = simpledialog.askinteger("Test Shot Interval", 
                                                "Enter the interval for test shots (in seconds):",
                                                minvalue=1)
            if interval is None:
                settings['test_shot_interval'] = 10  # Default value
            else:
                settings['test_shot_interval'] = interval
            break
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))

    # Get snapshot interval
    while True:
        try:
            snapshot_interval = simpledialog.askinteger("Snapshot Interval", 
                                                        "Enter the time interval between snapshots (in seconds):",
                                                        minvalue=1)
            if snapshot_interval is None:
                settings['snapshot_interval'] = 10  # Default value
            else:
                settings['snapshot_interval'] = snapshot_interval
            break
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))

    return settings

# Function to set up the GUI
def setup_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    return get_user_settings()

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

# Start GUI to get settings
settings = setup_gui()
motion_sensitivity = settings['motion_sensitivity']
test_shot_interval = settings['test_shot_interval']
snapshot_interval = settings['snapshot_interval']  # Get snapshot interval from settings
last_test_shot_time = time.time()  # Timestamp of the last test shot

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

try:
    print("Press 'q' to quit the script.")

    # Take an initial test shot at the start
    ret, initial_frame = cap.read()
    if ret:
        take_snapshot(initial_frame)

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
        if motion_detected > motion_sensitivity:  # Use dynamic sensitivity
            # If enough time has passed since the last snapshot
            if current_time - last_snapshot_time >= snapshot_interval:
                take_snapshot(frame)
                last_snapshot_time = current_time  # Update the last snapshot time

        # Periodic test shots
        if current_time - last_test_shot_time >= test_shot_interval:
            take_snapshot(frame)
            last_test_shot_time = current_time  # Update the last test shot time

        # Draw rectangles around detected faces for visualization
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around faces

        # Update the previous frame for motion detection
        previous_frame = gray

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
