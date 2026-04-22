import cv2
import numpy as np
import os
from PIL import Image

def train_model():
    # Dataset folder ka path
    dataset_path = "dataset"
    
    # Haar cascade load karo face detect karne ke liye
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml'
    )
    
    # LBPH Face Recognizer banao
    # LBPH = Local Binary Patterns Histograms
    # Yeh har chehra ek unique pattern se seekhta hai
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []   # Photos store hongi yahan
    ids = []     # Har photo ka ID store hoga
    
    print("Training started — reading photos...")
    
    # Har student ke folder mein jao
    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        
        if not os.path.isdir(student_folder):
            continue
            
        # Student ko ek unique number do
        # Kyunki model numbers samajhta hai, names nahi
        student_id = abs(hash(student_name)) % 1000
        
        print(f"Reading photos of: {student_name} (ID: {student_id})")
        
        # Us student ki har photo padho
        for photo_file in os.listdir(student_folder):
            photo_path = os.path.join(student_folder, photo_file)
            
            # Photo ko grayscale mein kholo
            # Grayscale = black & white — model ke liye better
            img = Image.open(photo_path).convert('L')
            img_array = np.array(img, 'uint8')
            
            faces.append(img_array)
            ids.append(student_id)
    
    if len(faces) == 0:
        print("No photos found! Pehle register.py chalao.")
        return
    
    # Model train karo
    ids_array = np.array(ids)
    recognizer.train(faces, ids_array)
    
    # Trained model save karo
    recognizer.save("classifier.xml")
    print(f"Training complete! {len(faces)} photos se model trained.")
    print("classifier.xml saved!")

train_model()