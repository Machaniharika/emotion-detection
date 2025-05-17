import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os

# --- Emotion Labels ---
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# --- Function to reconstruct SVM model from parts ---
def merge_model_parts(output_file='emotion_svm.pkl', part_prefix='emotion_svm.pkl.part'):
    index = 1
    try:
        with open(output_file, 'wb') as output:
            while True:
                part_file = f"{part_prefix}{index}"
                if not os.path.exists(part_file):
                    break
                with open(part_file, 'rb') as pf:
                    output.write(pf.read())
                    st.info(f"Merged: {part_file}")
                index += 1
        st.success("✅ Model reconstruction complete.")
    except Exception as e:
        st.error(f"❌ Error merging model parts: {e}")

# --- Ensure model file exists ---
model_path = "emotion_svm.pkl"
if not os.path.exists(model_path):
    st.warning("⚠️ Model file not found. Attempting to reconstruct...")
    merge_model_parts()

if not os.path.exists(model_path):
    st.error("❌ Model still missing after reconstruction. Please upload all .part files.")
    st.stop()

# --- Load Haar Cascade for Face Detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load the trained SVM model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(model_path)
        st.success("✅ SVM model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load SVM model: {e}")
        st.stop()

model = load_model()

# --- Streamlit UI ---
st.title("😊 Real-time Emotion Detection (SVM Model)")
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
motion_threshold = 800
prev_gray = None

if run:
    # Try multiple webcam indexes to ensure access
    for cam_index in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            break
    else:
        st.error("❌ Could not open any webcam (tried indexes 0, 1, 2).")
        st.stop()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame from webcam.")
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
                face_img = gray[y:y+h, x:x+w]
                try:
                    resized_face = cv2.resize(face_img, (48, 48))
                    input_vec = resized_face.flatten().reshape(1, -1) / 255.0

                    pred_idx = model.predict(input_vec)[0]
                    emotion = emotion_labels[pred_idx]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frame, "Face error", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.error(f"Face processing error: {e}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()
    FRAME_WINDOW.image([])

else:
    st.info("👆 Click the checkbox above to start the webcam.")

    # --- Browser-compatible fallback: camera input ---
    st.subheader("📷 Browser Camera Snapshot (for deployed apps)")
    image_data = st.camera_input("Take a picture using your webcam")

    if image_data is not None:
        st.success("✅ Image captured. Processing...")

        img = Image.open(image_data).convert("L")  # Convert to grayscale
        img = img.resize((48, 48))  # Match the SVM input format
        input_vec = np.array(img).flatten().reshape(1, -1) / 255.0

        try:
            pred_idx = model.predict(input_vec)[0]
            emotion = emotion_labels[pred_idx]
            st.markdown(f"### 🧠 Predicted Emotion: **{emotion}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

