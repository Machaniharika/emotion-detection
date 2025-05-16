
import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import pathlib

# --- Function to reconstruct model from parts ---
def merge_model_parts(output_file='emotion_model.pth', part_prefix='emotion_model.pth.part'):
    index = 1
    try:
        with open(output_file, 'wb') as output:
            while True:
                part_file = f"{part_prefix}{index}"
                try:
                    with open(part_file, 'rb') as pf:
                        output.write(pf.read())
                        st.info(f"Merged: {part_file}")
                except FileNotFoundError:
                    break
                index += 1
        st.success("Model reconstruction complete.")
    except Exception as e:
        st.error(f"Error merging model parts: {e}")

# --- Check if model file exists ---
model_path = "emotion_model.pth"
if not pathlib.Path(model_path).exists():
    st.warning("Model file not found. Attempting to reconstruct from parts...")
    merge_model_parts()

if not pathlib.Path(model_path).exists():
    st.error("Model still missing after reconstruction. Please upload all .part files.")
    st.stop()

# --- Load model ---
@st.cache_resource(show_spinner=True)
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 4)  # 4 emotion classes
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Preprocessing and labels ---
emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

# --- Load Haar Cascade ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Streamlit UI ---
st.title("🧠 Real-time Emotion Detection via Webcam")
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

motion_threshold = 800
prev_gray = None
cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Anti-spoofing: motion detection
        motion = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.sum(diff)
            if score > motion_threshold:
                motion = True
        prev_gray = gray

        # Face Detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not motion:
            cv2.putText(frame, "Spoof detected! No motion", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_img)
                input_tensor = transform(pil_img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion = emotion_labels[predicted.item()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()
    FRAME_WINDOW.image([])

else:
    st.info("Click the checkbox above to start the webcam.")

