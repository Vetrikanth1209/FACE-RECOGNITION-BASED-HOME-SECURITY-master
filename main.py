import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import vonage
import time

# Path to the directory containing student images
path = 'Student_image'

# Load images and corresponding class names
images = []
classNames = []
for cl in os.listdir(path):
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert image to RGB format (required by face_recognition library)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encode face
        encoded_face = face_recognition.face_encodings(imgRGB)[0]
        encodeList.append(encoded_face)
    return encodeList

# Encode faces in the dataset
encoded_face_train = findEncodings(images)

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S %p')
            date = now.strftime('%d-%B-%Y')
            f.write(f'{name}, {time}, {date}\n')

# Vonage configuration
client = vonage.Client(key="9f59589d", secret="qEnoYqxxusjv4i1V")
sms = vonage.Sms(client)
recipient_phone_number = "916379790184"  # Replace with recipient's phone number

# Function to send SMS using Vonage API
def send_sms(message):
    responseData = sms.send_message(
        {
            "from": "Vonage APIs",
            "to": recipient_phone_number,
            "text": message,
        }
    )
    if responseData["messages"][0]["status"] == "0":
        print("Message sent successfully.")
    else:
        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")

# Capture video from webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    current_time = time.time()
    if current_time - start_time >= 10:  # Send SMS every 15 seconds
        start_time = time.time()  # Reset tixcmer
        send_sms("Unknown face detected!")  # Send SMS
    success, img = cap.read()
    # Resize input frame for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # Locate faces in the frame
    faces_in_frame = face_recognition.face_locations(imgS)
    # Encode faces in the frame
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        # Compare the encoded face with the encoded faces in the dataset
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
        else:
            name = "Unknown"
        # Extract face location coordinates
        y1, x2, y2, x1 = faceloc
        # Scale coordinates back to original size
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        # Draw rectangle around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        # Display name of the recognized person
        cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        # Mark attendance if recognized
        if name != "Unknown":
            markAttendance(name)
    # Display the output frame
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()