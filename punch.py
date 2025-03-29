import cv2
import mediapipe as mp
import numpy as np
from twilio.rest import Client
import time

# 🔹 Twilio Credentials (Replace with actual credentials)
ACCOUNT_SID = "your_account_sid"
AUTH_TOKEN = "your_auth_token"
TWILIO_PHONE = "your_twilio_phone"
RECIPIENT_PHONE = "recipient_phone_number"

# Initialize Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for punch detection
prev_right_wrist = None
prev_left_wrist = None
prev_right_elbow = None
prev_left_elbow = None

punch_threshold = 80  # Adjusted for better accuracy
punch_detected_time = 0

# Function to send Twilio Alert
def send_alert(message):
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )
        print("🚀 Alert sent successfully!")
    except Exception as e:
        print(f"Error sending alert: {e}")

# Surveillance loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Extract pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmark positions
        landmarks = results.pose_landmarks.landmark

        # Extract wrist and elbow positions
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]])

        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]])

        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]])

        left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]])

        # Check movement speed (difference between current & previous frame)
        if prev_right_wrist is not None and prev_left_wrist is not None:
            right_wrist_speed = np.linalg.norm(right_wrist - prev_right_wrist)
            left_wrist_speed = np.linalg.norm(left_wrist - prev_left_wrist)

            right_elbow_speed = np.linalg.norm(right_elbow - prev_right_elbow)
            left_elbow_speed = np.linalg.norm(left_elbow - prev_left_elbow)

            # Detect punch based on wrist & elbow movement
            if (right_wrist_speed > punch_threshold and right_elbow_speed > punch_threshold) or \
               (left_wrist_speed > punch_threshold and left_elbow_speed > punch_threshold):

                print("🥊 Punch detected!")

                # Prevent multiple alerts within 5 seconds
                if time.time() - punch_detected_time > 5:
                    send_alert("🚨 Warning! A punch was detected in the surveillance area.")
                    punch_detected_time = time.time()

        # Update previous positions
        prev_right_wrist = right_wrist
        prev_left_wrist = left_wrist
        prev_right_elbow = right_elbow
        prev_left_elbow = left_elbow

    # Display video
    cv2.imshow("Punch Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
