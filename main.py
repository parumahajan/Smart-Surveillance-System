# import face_recognition
# import os
# import sys
# import cv2
# import numpy as np
# import math
# from twilio.rest import Client
# import threading
# import time
# from ultralytics import YOLO

# # Twilio credentials
# account_sid = 'ACf2e5c0bed7aa9c32701dbabe6924836b'
# auth_token = '508dd720b14fc044c44856666b8ade5a'
# twilio_phone_number = '+17629944504'
# recipient_phone = '+919108244780'

# # Directory to save detected problematic objects
# PROBOBJ_DIR = 'probobj'
# if not os.path.exists(PROBOBJ_DIR):
#     os.makedirs(PROBOBJ_DIR)

# # Load YOLO model for object detection (guns and knives)
# yolo_model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano) or a custom model
# TARGET_CLASSES = ['gun', 'knife']  # Adjust based on your model's class names

# def send_alarm_message(message_body="ALARM! Unknown person detected."):
#     """Send an SMS alert via Twilio."""
#     client = Client(account_sid, auth_token)
#     message = client.messages.create(
#         body=message_body,
#         from_=twilio_phone_number,
#         to=recipient_phone
#     )
#     print("SMS Sent! Message SID:", message.sid)

# def make_call():
#     """Make an automated voice call via Twilio."""
#     client = Client(account_sid, auth_token)
#     call = client.calls.create(
#         url='http://demo.twilio.com/docs/voice.xml',
#         to=recipient_phone,
#         from_=twilio_phone_number
#     )
#     print("Call Initiated! Call SID:", call.sid)

# def send_whatsapp_message(message_body="ALARM! Unknown person detected."):
#     """Send a WhatsApp message alert via Twilio."""
#     client = Client(account_sid, auth_token)
#     message = client.messages.create(
#         body=message_body,
#         from_='whatsapp:+14155238886',
#         to=f'whatsapp:{recipient_phone}'
#     )
#     print("WhatsApp Alert Sent! Message SID:", message.sid)

# def face_confidence(face_distance, face_match_threshold=0.6):
#     """Calculate confidence score for face recognition."""
#     if face_distance > 60.0:
#         return "0%"
#     confidence = (1.0 - face_distance) * 100
#     return f"{round(confidence, 2)}%"

# class SurveillanceSystem:
#     def __init__(self):
#         self.face_locations = []
#         self.face_encodings = []
#         self.face_names = []
#         self.known_face_encodings = []
#         self.known_face_names = []
#         self.process_current_frame = True
#         self.seen_objects = {}  # New: Dictionary to track seen objects
#         self.load_known_faces()

#     def load_known_faces(self):
#         """Load and encode known faces from 'faces' directory."""
#         if not os.path.exists('faces'):
#             print("Error: 'faces' directory not found!")
#             return
        
#         for image_file in os.listdir('faces'):
#             try:
#                 img_path = os.path.join('faces', image_file)
#                 face_image = face_recognition.load_image_file(img_path)
#                 encodings = face_recognition.face_encodings(face_image)
#                 if encodings:
#                     self.known_face_encodings.append(encodings[0])
#                     self.known_face_names.append(os.path.splitext(image_file)[0])
#                 else:
#                     print(f"Warning: No face found in {image_file}. Skipping.")
#             except Exception as e:
#                 print(f"Error processing {image_file}: {e}")
#         print("Loaded known faces:", self.known_face_names)

#     def detect_objects(self, frame):
#         """Detect guns and knives using YOLO, track new objects, and save detected objects."""
#         results = yolo_model(frame)
#         detected_objects = []
#         current_frame_objects = set()  # Track objects in this frame

#         for result in results:
#             for box in result.boxes:
#                 class_id = int(box.cls[0])
#                 label = yolo_model.names[class_id]
#                 confidence = float(box.conf)
#                 if label in TARGET_CLASSES and confidence > 0.5:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     obj_key = f"{label}_{x1}_{y1}_{x2}_{y2}"  # Unique key for object based on position
#                     current_frame_objects.add(obj_key)
#                     is_new = obj_key not in self.seen_objects

#                     detected_objects.append((label, confidence, (x1, y1, x2, y2), is_new))

#                     # Save the detected object image
#                     obj_img = frame[y1:y2, x1:x2]
#                     timestamp = time.strftime("%Y%m%d_%H%M%S")
#                     cv2.imwrite(os.path.join(PROBOBJ_DIR, f"{label}_{timestamp}.jpg"), obj_img)
#                     print(f"Saved {label} image to {PROBOBJ_DIR}")

#                     # Draw bounding box and label (orange for new, green for seen)
#                     color = (255, 165, 0) if is_new else (0, 255, 0)  # Orange for new, green for seen
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                     # Mark object as seen
#                     if is_new:
#                         self.seen_objects[obj_key] = time.time()  # Record time of first detection

#         # Clean up old objects no longer in frame (optional, to manage memory)
#         self.seen_objects = {k: v for k, v in self.seen_objects.items() if k in current_frame_objects or (time.time() - v) < 10}  # Keep for 10 seconds

#         return detected_objects

#     def run_recognition(self):
#         """Run real-time face recognition and object detection using webcam."""
#         video_capture = cv2.VideoCapture(0)
#         if not video_capture.isOpened():
#             sys.exit('Error: Video source not found.')

#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 print("Error: Failed to capture video frame.")
#                 continue

#             # Object detection (guns and knives)
#             detected_objects = self.detect_objects(frame)
#             if detected_objects:
#                 threading.Thread(target=self.trigger_twilio_alerts, args=("ALARM! Dangerous object detected!", detected_objects)).start()

#             # Face recognition (processed every other frame)
#             if self.process_current_frame:
#                 small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#                 rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#                 self.face_locations = face_recognition.face_locations(rgb_small_frame)
#                 if self.face_locations:
#                     self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
#                 else:
#                     self.face_encodings = []

#                 self.face_names = []
#                 for face_encoding in self.face_encodings:
#                     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#                     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#                     name = "Unknown"
#                     confidence = "N/A"
#                     if len(face_distances) > 0:
#                         best_match_index = np.argmin(face_distances)
#                         if matches[best_match_index]:
#                             name = self.known_face_names[best_match_index]
#                             confidence = face_confidence(face_distances[best_match_index])
#                     self.face_names.append(f'{name} ({confidence})')

#             self.process_current_frame = not self.process_current_frame

#             # Draw face rectangles and labels
#             for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
#                 top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
#                 if name.startswith("Unknown"):
#                     threading.Thread(target=self.trigger_twilio_alerts, args=("ALARM! Unknown person detected!", None)).start()
#                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#                 cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#                 cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

#             cv2.imshow('Surveillance System', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break

#         video_capture.release()
#         cv2.destroyAllWindows()

#     def trigger_twilio_alerts(self, message_body, detected_objects=None):
#         """Trigger Twilio alerts for unknown faces or detected objects."""
#         print(f"🚨 ALERT: {message_body}")
#         send_alarm_message(message_body)
#         make_call()
#         send_whatsapp_message(message_body)
#         time.sleep(3)  # Prevent multiple triggers

# # Run the surveillance system
# if __name__ == "__main__":
#     ss = SurveillanceSystem()
#     ss.run_recognition()

import face_recognition
import os
import sys
import cv2
import numpy as np
import math
from twilio.rest import Client
import threading
import time
from ultralytics import YOLO

# Twilio credentials
account_sid = 'ACf2e5c0bed7aa9c32701dbabe6924836b'
auth_token = '508dd720b14fc044c44856666b8ade5a'
twilio_phone_number = '+17629944504'
recipient_phone = '+919108244780'

# Directory to save detected problematic objects
PROBOBJ_DIR = 'probobj'
if not os.path.exists(PROBOBJ_DIR):
    os.makedirs(PROBOBJ_DIR)

# Load YOLO model for object detection
yolo_model = YOLO('yolov8n.pt')  # Use yolov8n.pt (nano) or a custom model
TARGET_CLASSES = ['gun', 'knife']  # Guns and knives get special treatment

def send_alarm_message(message_body="ALARM! Unknown person detected."):
    """Send an SMS alert via Twilio."""
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=recipient_phone
    )
    print("SMS Sent! Message SID:", message.sid)

def make_call():
    """Make an automated voice call via Twilio."""
    client = Client(account_sid, auth_token)
    call = client.calls.create(
        url='http://demo.twilio.com/docs/voice.xml',
        to=recipient_phone,
        from_=twilio_phone_number
    )
    print("Call Initiated! Call SID:", call.sid)

def send_whatsapp_message(message_body="ALARM! Unknown person detected."):
    """Send a WhatsApp message alert via Twilio."""
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message_body,
        from_='whatsapp:+14155238886',
        to=f'whatsapp:{recipient_phone}'
    )
    print("WhatsApp Alert Sent! Message SID:", message.sid)

def face_confidence(face_distance, face_match_threshold=0.6):
    """Calculate confidence score for face recognition."""
    if face_distance > 60.0:
        return "0%"
    confidence = (1.0 - face_distance) * 100
    return f"{round(confidence, 2)}%"

class SurveillanceSystem:
    
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.load_known_faces()

    def load_known_faces(self):
        """Load and encode known faces from 'faces' directory."""
        if not os.path.exists('faces'):
            print("Error: 'faces' directory not found!")
            return
        
        for image_file in os.listdir('faces'):
            try:
                img_path = os.path.join('faces', image_file)
                face_image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(face_image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(image_file)[0])
                else:
                    print(f"Warning: No face found in {image_file}. Skipping.")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        print("Loaded known faces:", self.known_face_names)

    def detect_objects(self, frame):
        """Detect all objects using YOLO, save guns/knives, and draw colored boxes."""
        results = yolo_model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = yolo_model.names[class_id]
                confidence = float(box.conf)
                if confidence > 0.5:  # Apply threshold to all objects
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Determine color: green for guns/knives, orange for others
                    color = (0, 255, 0) if label in TARGET_CLASSES else (255, 165, 0)  # Green for guns/knives, orange for others
                    
                    # Only save guns and knives to probobj
                    if label in TARGET_CLASSES:
                        obj_img = frame[y1:y2, x1:x2]
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(PROBOBJ_DIR, f"{label}_{timestamp}.jpg"), obj_img)
                        print(f"Saved {label} image to {PROBOBJ_DIR}")
                        detected_objects.append((label, confidence, (x1, y1, x2, y2)))
                        
                    elif label == "fire":
                        threading.Thread(target=self.trigger_fire_alerts, args=("🔥 FIRE DETECTED! Calling 911...",)).start()

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return detected_objects  # Only guns/knives for alerts

    def run_recognition(self):
        """Run real-time face recognition and object detection using webcam."""
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Error: Video source not found.')

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture video frame.")
                continue

            # Object detection (all objects, special handling for guns/knives)
            detected_objects = self.detect_objects(frame)
            if detected_objects:  # Trigger alerts only for guns/knives
                threading.Thread(target=self.trigger_twilio_alerts, args=("ALARM! Dangerous object detected!", detected_objects)).start()

            # Face recognition (processed every other frame)
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                if self.face_locations:
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                else:
                    self.face_encodings = []

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "N/A"
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Draw face rectangles and labels (red)
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                if name.startswith("Unknown"):
                    threading.Thread(target=self.trigger_twilio_alerts, args=("ALARM! Unknown person detected!", None)).start()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Surveillance System', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def trigger_twilio_alerts(self, message_body, detected_objects=None):
        """Trigger Twilio alerts for unknown faces or detected objects."""
        print(f"🚨 ALERT: {message_body}")
        send_alarm_message(message_body)
        make_call()
        send_whatsapp_message(message_body)
        time.sleep(3)  # Prevent multiple triggers

# Run the surveillance system
if __name__ == "__main__":
    ss = SurveillanceSystem()
    ss.run_recognition()