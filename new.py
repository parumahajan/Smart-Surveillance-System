import cv2
import numpy as np
import pyttsx3
from twilio.rest import Client

# 🔹 Twilio Credentials (Replace with actual credentials)
ACCOUNT_SID = "ACf2e5c0bed7aa9c32701dbabe6924836b"
AUTH_TOKEN = "508dd720b14fc044c44856666b8ade5a"
TWILIO_PHONE = "+17629944504"
RECIPIENT_PHONE = "+919108244780"

# 🔹 Load YOLO model (Ensure these files exist in your directory)
yolo_weights = "yolov3.weights"
yolo_cfg = "yolov3.cfg"
yolo_classes = "coco.names"

try:
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    
    # 🔹 Fix: Handle different output formats of getUnconnectedOutLayers()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if len(unconnected_out_layers.shape) == 1:  # If it's a 1D array
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    else:  # If it's a 2D array (older versions)
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# 🔹 Load class names for YOLO
with open(yolo_classes, "r") as f:
    classes = f.read().strip().split("\n")

# 🔹 Load Video Stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 🔹 Function to send Twilio alerts
def send_alert(message):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        # Send SMS
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )

        # Send WhatsApp Alert
        client.messages.create(
            body=message,
            from_="whatsapp:" + TWILIO_PHONE,
            to="whatsapp:" + RECIPIENT_PHONE
        )

        # Make a phone call
        client.calls.create(
            twiml=f'<Response><Say>{message}</Say></Response>',
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )
        print("🚀 Alert sent successfully!")
    except Exception as e:
        print(f"Error sending alert: {e}")

# 🔹 Function to detect fire (Color-based)
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 120, 200])  # Adjust fire color range
    upper_fire = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    return cv2.countNonZero(mask) > 500  # Fire threshold

# 🔹 Surveillance Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    fire_detected = detect_fire(frame)

    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)

        if class_id >= len(scores):  # Avoid index errors
            continue

        confidence = scores[class_id]
        if confidence > 0.6:  # Confidence threshold
            object_name = classes[class_id]  # Use class names from coco.names
            print(f"⚠️ Detected: {object_name} - Confidence: {confidence:.2f}")

            # If a weapon is detected, send an alert
            if object_name.lower() in ["gun", "knife"]:
                send_alert(f"🚨 Warning! {object_name} detected in the surveillance area.")

    # 🔹 If fire is detected, send an alert
    if fire_detected:
        print("🔥 Fire detected!")
        send_alert("🚨 Emergency! Fire detected in the surveillance area. Please take immediate action.")

    cv2.imshow("Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np
import face_recognition
import pyttsx3
from twilio.rest import Client

# Twilio Credentials (Replace with your credentials)
ACCOUNT_SID = "your_account_sid"
AUTH_TOKEN = "your_auth_token"
TWILIO_PHONE = "+your_twilio_phone"
RECIPIENT_PHONE = "+recipient_phone_number"

# Load YOLO model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load known face encodings (if any)
known_face_encodings = []  # Add known encodings
known_face_names = []  # Add corresponding names

# Open video stream
cap = cv2.VideoCapture(0)

# Twilio Alert Function
def send_alert(message):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        
        # Send SMS
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )

        # Send WhatsApp Alert
        client.messages.create(
            body=message,
            from_="whatsapp:" + TWILIO_PHONE,
            to="whatsapp:" + RECIPIENT_PHONE
        )

        # Make a phone call
        call = client.calls.create(
            twiml=f'<Response><Say>{message}</Say></Response>',
            from_=TWILIO_PHONE,
            to=RECIPIENT_PHONE
        )

    except Exception as e:
        print(f"Error sending alert: {e}")

# Fire Detection Function
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 150, 150])  # Adjusted fire color range
    upper_fire = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    return cv2.countNonZero(mask) > 1000  # Fire detection threshold

# Surveillance Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    fire_detected = detect_fire(frame)

    for detection in detections:
        scores = detection[5:]
        
        if len(scores) > 0:
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                object_name = "Gun" if class_id == 0 else "Knife"  # Adjust class IDs
                print(f"⚠️ Detected: {object_name} - Confidence: {confidence:.2f}")

                # Send alert for weapon detection
                send_alert(f"Warning! {object_name} detected in the surveillance area.")

    if fire_detected:
        print("🔥 Fire detected!")
        send_alert("Emergency! Fire detected in the surveillance area. Please take immediate action.")

    cv2.imshow("Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
