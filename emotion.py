import cv2
from tkinter import Tk, Label, Button, Entry, StringVar
from deepface import DeepFace
import os
import face_recognition
import pyttsx3
import numpy as np

# Global Variables
user_name = ""
images_captured = 0
dataset_path = "user_datasets"
known_face_encodings = []
known_face_names = []

# Initialize text-to-speech and dataset directory
engine = pyttsx3.init()
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def load_known_faces():
    """Load all known face encodings from the dataset."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    for user in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user)
        if not os.path.isdir(user_folder):
            continue
            
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(user)
                    print(f"Loaded face data: {user} - {image_name}")
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")

def home_page():
    """Create the home page."""
    home = Tk()
    home.title("Face Recognizer")
    home.geometry("400x300")
    Label(home, text="Face Recognizer", font=("Arial", 20)).pack(pady=20)
    Button(home, text="Add a User", width=20, command=lambda: add_user_page(home)).pack(pady=10)
    Button(home, text="Check a User", width=20, command=lambda: check_user_page(home)).pack(pady=10)
    home.mainloop()

def add_user_page(home):
    """Create the add user page."""
    home.destroy()
    add_user = Tk()
    add_user.title("Add a User")
    add_user.geometry("400x300")
    Label(add_user, text="Enter User Name:", font=("Arial", 14)).pack(pady=10)
    name_var = StringVar()
    Entry(add_user, textvariable=name_var, font=("Arial", 14)).pack(pady=10)
    Button(add_user, text="Next", width=20, command=lambda: capture_dataset_page(add_user, name_var.get())).pack(pady=10)
    Button(add_user, text="Back", width=20, command=lambda: (add_user.destroy(), home_page())).pack(pady=10)
    add_user.mainloop()

def capture_dataset_page(add_user, name):
    """Create the capture dataset page."""
    global user_name, images_captured
    add_user.destroy()
    user_name = name.strip()
    images_captured = 0
    
    capture_page = Tk()
    capture_page.title("Capture Dataset")
    capture_page.geometry("400x300")
    Label(capture_page, text=f"User: {user_name}", font=("Arial", 14)).pack(pady=10)
    count_label = Label(capture_page, text=f"Images captured: {images_captured}", font=("Arial", 14))
    count_label.pack(pady=10)
    Button(capture_page, text="Capture Dataset", width=20, command=lambda: capture_images(count_label)).pack(pady=10)
    Button(capture_page, text="Train Model", width=20, command=lambda: train_model()).pack(pady=10)
    Button(capture_page, text="Home", width=20, command=lambda: (capture_page.destroy(), home_page())).pack(pady=10)
    capture_page.mainloop()

def capture_images(count_label):
    """Capture images of the user."""
    global images_captured, user_name
    if not user_name:
        print("Error: User name is empty")
        return

    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while images_captured < 20:  # Reduced for faster processing
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Ensure minimum face size
            if w < 100 or h < 100:
                continue
                
            image_path = os.path.join(user_folder, f"{images_captured}.jpg")
            cv2.imwrite(image_path, face_roi)

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                
                engine.say(f"{user_name} is {emotion}")
                engine.runAndWait()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                images_captured += 1
                count_label.config(text=f"Images captured: {images_captured}")
            except Exception as e:
                print(f"Error analyzing face: {str(e)}")

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """Train the face recognition model."""
    load_known_faces()
    print("Model trained successfully!")
    engine.say("Training completed")
    engine.runAndWait()

def check_user_page(home):
    """Create the check user page."""
    home.destroy()
    check_user = Tk()
    check_user.title("Check a User")
    check_user.geometry("400x300")
    Label(check_user, text="Check User", font=("Arial", 14)).pack(pady=10)
    Button(check_user, text="Recognize User", width=20, command=recognize_user).pack(pady=10)
    Button(check_user, text="Home", width=20, command=lambda: (check_user.destroy(), home_page())).pack(pady=10)
    check_user.mainloop()

def recognize_user():
    """Recognize the user from the webcam."""
    if not known_face_encodings:
        load_known_faces()
        if not known_face_encodings:
            print("No face data available")
            engine.say("No users found in database")
            engine.runAndWait()
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                name = known_face_names[matches.index(True)]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    home_page()
