import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Function to compute the distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = compute(eye[1], eye[5])
    B = compute(eye[2], eye[4])
    C = compute(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate the lip distance (LD)
def lip_distance(shape):
    top_lip = shape[50:53] + shape[61:64]
    bottom_lip = shape[56:59] + shape[65:68]
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    return compute(top_mean, bottom_mean)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Calculate stress level based on eye aspect ratio (EAR) and lip distance (LD)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        ld = lip_distance(shape)
        stress = (ear + ld) / 2.0

        # Display stress level on the frame
        if stress < 0.2:
            status = "Not Stressed"
            color = (0, 255, 0)  # Green color for not stressed
        else:
            status = "Stressed"
            color = (0, 0, 255)  # Red color for stressed
        cv2.putText(frame, f"Status: {status}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw facial landmarks on the frame
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
