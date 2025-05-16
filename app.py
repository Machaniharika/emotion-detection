# Save as app.py
import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import pathlib

# --- Reconstruct model from parts if needed ---
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

# --- Check and merge model ---
if not pathlib.Path("emotion_model.pth").exists():
    st.warning("Model file not found. Attempting to reconstruct from parts...")
    merge_model_parts()

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 4)
    model.load_state_dict(torch.load("emotion_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- Labels and preprocessing ---
emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

# --- Face Detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Streamlit UI ---
st.title("Emotion Detection from Webcam (PyTorch, No TensorFlow)")
start_webcam = st.checkbox('Start Webcam')

frame_window = st.image([])
motion_threshold = 800
prev_gray = None
cap = None

# --- Webcam Logic ---
try:
    if start_webcam:
        if cap is None:
            cap = cv2.VideoCapture(0)

        stop_webcam = False

        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Anti-spoof: motion detection
            motion = False
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = np.sum(diff)
                if score > motion_threshold:
                    motion = True
            prev_gray = gray

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif not motion:
                cv2.putText(frame, "Spoof detected! No motion", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

            # This will stop loop on user action outside loop
            stop_webcam = not st.session_state.get("continue_stream", True)

        cap.release()
        cap = None
except Exception as e:
    st.error(f"Error: {e}")
finally:
    if cap is not None:
        cap.release()

# --- Toggle to stop stream (must be outside loop to avoid key conflict) ---
if start_webcam:
    stop_button = st.button("Stop Webcam")
    if stop_button:
        st.session_state["continue_stream"] = False
    else:
        st.session_state["continue_stream"] = True
