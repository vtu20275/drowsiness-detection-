import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import datetime
import threading
import requests

# Initialize the sound system and load the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

# Initialize dlib's face detector and face landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

# Declare global variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Create a log file for detection results and feedback
log_filename = "drowsiness_detection_log.csv"
with open(log_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Status", "Feedback"])

# Pushover API credentials
PUSHOVER_API_TOKEN = "your_api_token"
PUSHOVER_USER_KEY = "uqhnpgec6c4dbah5bgyyk6y8njx5p4"

def send_alert(message):
    """Send a real-time alert via Pushover."""
    try:
        response = requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message
        })
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")

def log_alert_time(status):
    """Log the time when an alert is triggered."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, status])

def compute(ptA, ptB):
    """Compute the Euclidean distance between two points."""
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    """Determine if the eye is closed based on landmark positions."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eyes open
    elif 0.21 < ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eyes closed

def update_frame(cap, label, root):
    """Update the video frame and analyze the drowsiness."""
    global sleep, drowsy, active, status, color

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        root.after(10, update_frame, cap, label, root)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = frame.copy()
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                pygame.mixer.Sound.play(alarm_sound)
                send_alert(status)
                log_alert_time(status)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                pygame.mixer.Sound.play(alarm_sound)
                send_alert(status)
                log_alert_time(status)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    label.config(image=image)
    label.image = image

    root.after(10, update_frame, cap, label, root)

def get_feedback():
    """Prompt the user for feedback."""
    feedback_window = tk.Toplevel(root)
    feedback_window.title("Feedback")

    tk.Label(feedback_window, text="Please provide your feedback:").pack(pady=10)

    questions = [
        "How accurate was the drowsiness detection?",
        "Did you face any issues during the detection session?",
        "Any suggestions to improve the system?"
    ]

    entries = []
    for question in questions:
        tk.Label(feedback_window, text=question).pack(pady=5)
        entry = tk.Entry(feedback_window, width=50)
        entry.pack(pady=5)
        entries.append(entry)

    def submit_feedback():
        feedback = [entry.get() for entry in entries]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, status] + feedback)
        feedback_window.destroy()
        print("Feedback submitted:", feedback)

    submit_button = ttk.Button(feedback_window, text="Submit", command=submit_feedback)
    submit_button.pack(pady=10)

def start_camera():
    """Start the camera and begin the drowsiness detection process."""
    global sleep, drowsy, active, status, color

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    root.title("Drowsiness Detection App")

    label = tk.Label(root)
    label.pack(pady=10)

    start_button.config(state=tk.DISABLED)  # Disable the start button while the camera is running

    # Run the frame update in a separate thread to keep the GUI responsive
    threading.Thread(target=update_frame, args=(cap, label, root)).start()

    def stop_camera():
        """Stop the camera and close the application."""
        cap.release()
        cv2.destroyAllWindows()
        get_feedback()

    stop_button = ttk.Button(root, text="Stop", command=stop_camera)
    stop_button.pack(pady=10)

    root.mainloop()

root = tk.Tk()
start_button = ttk.Button(root, text="Start", command=start_camera)
start_button.pack(pady=10)

root.mainloop()
