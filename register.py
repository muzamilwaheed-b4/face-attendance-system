import cv2
import os

def register_student():
    name = input("Student ka naam likho: ")
    
    save_path = f"dataset/{name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    
    count = 0
    print("Camera khul raha hai — apna chehra dikhao. 30 photos lenge.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Photo: {count}/30", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Register", frame)
        
        if count >= 30 or cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"{name} ka registration complete! {count} photos li gayin.")

register_student()