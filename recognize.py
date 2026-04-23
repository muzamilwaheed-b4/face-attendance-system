import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime

def get_student_name(student_id):
    dataset_path = "dataset"
    for student_name in os.listdir(dataset_path):
        if abs(hash(student_name)) % 1000 == student_id:
            return student_name
    return "Unknown"

def mark_attendance(name):
    file_exists = os.path.exists("attendance.csv")
    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")
        writer.writerow([name, date, time_now])
        print(f"Attendance marked: {name} at {time_now}")

def recognize_faces():
    if not os.path.exists("classifier.xml"):
        print("classifier.xml nahi mila! Pehle train.py chalao.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")

    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml'
    )

    cam = cv2.VideoCapture(0)
    time.sleep(2)

    if not cam.isOpened():
        print("Camera nahi khula!")
        return

    # Camera warm up
    for i in range(5):
        cam.read()

    marked_today = []
    print("Camera starting — press Q to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Frame nahi aaya — retry...")
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            student_id, confidence = recognizer.predict(face_region)

            if confidence < 70:
                name = get_student_name(student_id)
                label = f"{name} ({int(confidence)}%)"
                color = (0, 255, 0)
                if name not in marked_today:
                    mark_attendance(name)
                    marked_today.append(name)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("System closed.")

recognize_faces()