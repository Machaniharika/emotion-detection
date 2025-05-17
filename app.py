import streamlit as st
import cv2
import numpy as np
import joblib
import os

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

# --- Load the trained SVM model and label encoder ---
@st.cache_resource
def load_model_and_labels():
    try:
        model = joblib.load(model_path)
        st.success("✅ SVM model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load SVM model: {e}")
        st.stop()

    try:
        encoder = joblib.load("label_encoder.pkl")
        emotion_labels = encoder.classes_
        st.success("✅ Label encoder loaded.")
    except Exception as e:
        st.warning("⚠️ Label encoder not found. Using default label order (may be inaccurate).")
        emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

    return model, emotion_labels

model, emotion_labels = load_model_and_labels()

# --- Streamlit UI ---
st.title("😊 Emotion Detection from Webcam (Deployed)")

# Use browser webcam input widget (single image capture)
img_file_buffer = st.camera_input("Take a selfie")

motion_threshold = 800
prev_gray = None

if img_file_buffer is not None:
    # Convert uploaded image to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Anti-spoofing: motion detection — 
    # NOTE: since this is a single image snapshot, motion detection is tricky or impossible.
    # You can either skip or implement it by comparing with previous snapshot if you keep state.

    # Face Detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please try again.")
    else:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                resized_face = cv2.resize(face_img, (48, 48))
                input_vec = resized_face.flatten().reshape(1, -1) / 255.0

                pred_idx = model.predict(input_vec)[0]
                emotion = emotion_labels[pred_idx]

                # Draw rectangle and label on image
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Face processing error: {e}")

        # Show the image with rectangle and emotion label
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Emotion detection result")

else:
    st.info("👆 Use the above button to take a selfie and detect emotion.")
