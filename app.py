# import streamlit as st
# import cv2
# import torch
# import numpy as np
# from torchvision import models, transforms
# from PIL import Image
# import pathlib

# # --- Reconstruct model from parts ---
# def merge_model_parts(output_file='emotion_model.pth', part_prefix='emotion_model.pth.part'):
#     index = 1
#     try:
#         with open(output_file, 'wb') as output:
#             while True:
#                 part_file = f"{part_prefix}{index}"
#                 try:
#                     with open(part_file, 'rb') as pf:
#                         output.write(pf.read())
#                         st.info(f"Merged: {part_file}")
#                 except FileNotFoundError:
#                     break
#                 index += 1
#         st.success("Model reconstruction complete.")
#     except Exception as e:
#         st.error(f"Error merging model parts: {e}")

# # --- Check and merge model ---
# if not pathlib.Path("emotion_model.pth").exists():
#     st.warning("Model file not found. Attempting to reconstruct from parts...")
#     merge_model_parts()

# if not pathlib.Path("emotion_model.pth").exists():
#     st.error("Model file still missing after reconstruction. Please upload all .part files.")
#     st.stop()

# # --- Load model ---
# @st.cache_resource
# def load_model():
#     model = models.mobilenet_v2(weights=None)
#     model.classifier[1] = torch.nn.Linear(model.last_channel, 4)
#     model.load_state_dict(torch.load("emotion_model.pth", map_location='cpu'))
#     model.eval()
#     return model

# model = load_model()

# # --- Labels and preprocessing ---
# emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
# transform = transforms.Compose([
#     transforms.Resize((96, 96)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485]*3, [0.229]*3)
# ])

# # --- Face Detection ---
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # --- Streamlit UI ---
# st.title("Emotion Detection from Webcam (PyTorch)")
# run = st.toggle("Start Webcam")
# FRAME_WINDOW = st.image([])
# motion_threshold = 800
# prev_gray = None

# # --- Webcam loop ---
# cap = None
# if run:
#     cap = cv2.VideoCapture(0)

# while run:
#     if cap is None:
#         break

#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Failed to access webcam")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Anti-spoofing motion detection
#     motion = False
#     if prev_gray is not None:
#         diff = cv2.absdiff(prev_gray, gray)
#         score = np.sum(diff)
#         if score > motion_threshold:
#             motion = True
#     prev_gray = gray

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     elif not motion:
#         cv2.putText(frame, "Spoof detected! No motion", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     else:
#         for (x, y, w, h) in faces:
#             face_img = frame[y:y+h, x:x+w]
#             face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(face_img)
#             input_tensor = transform(pil_img).unsqueeze(0)

#             with torch.no_grad():
#                 outputs = model(input_tensor)
#                 _, predicted = torch.max(outputs, 1)
#                 emotion = emotion_labels[predicted.item()]

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)

# if cap is not None:
#     cap.release()

# 

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import joblib
# import os

# # --- Emotion Labels ---
# emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# # --- Function to reconstruct SVM model from parts ---
# def merge_model_parts(output_file='emotion_svm.pkl', part_prefix='emotion_svm.pkl.part'):
#     index = 1
#     try:
#         with open(output_file, 'wb') as output:
#             while True:
#                 part_file = f"{part_prefix}{index}"
#                 if not os.path.exists(part_file):
#                     break
#                 with open(part_file, 'rb') as pf:
#                     output.write(pf.read())
#                     st.info(f"Merged: {part_file}")
#                 index += 1
#         st.success("✅ Model reconstruction complete.")
#     except Exception as e:
#         st.error(f"❌ Error merging model parts: {e}")

# # --- Ensure model file exists ---
# model_path = "emotion_svm.pkl"
# if not os.path.exists(model_path):
#     st.warning("⚠️ Model file not found. Attempting to reconstruct...")
#     merge_model_parts()

# if not os.path.exists(model_path):
#     st.error("❌ Model still missing after reconstruction. Please upload all .part files.")
#     st.stop()

# # --- Load Haar Cascade for Face Detection ---
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # --- Load the trained SVM model ---
# @st.cache_resource
# def load_model():
#     try:
#         model = joblib.load(model_path)
#         st.success("✅ SVM model loaded successfully.")
#         return model
#     except Exception as e:
#         st.error(f"❌ Failed to load SVM model: {e}")
#         st.stop()

# model = load_model()

# # --- Streamlit UI ---
# st.title("😊 Real-time Emotion Detection (SVM Model)")
# run = st.checkbox("Start Webcam")

# FRAME_WINDOW = st.image([])
# motion_threshold = 800
# prev_gray = None

# if run:
#     # Try multiple webcam indexes to ensure access
#     for cam_index in [0, 1, 2]:
#         cap = cv2.VideoCapture(cam_index)
#         if cap.isOpened():
#             break
#     else:
#         st.error("❌ Could not open any webcam (tried indexes 0, 1, 2).")
#         st.stop()

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("⚠️ Failed to read frame from webcam.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Anti-spoofing: motion detection
#         motion = False
#         if prev_gray is not None:
#             diff = cv2.absdiff(prev_gray, gray)
#             score = np.sum(diff)
#             if score > motion_threshold:
#                 motion = True
#         prev_gray = gray

#         # Face Detection
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         if len(faces) == 0:
#             cv2.putText(frame, "No face detected", (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         elif not motion:
#             cv2.putText(frame, "Spoof detected! No motion", (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             for (x, y, w, h) in faces:
#                 face_img = gray[y:y+h, x:x+w]
#                 try:
#                     resized_face = cv2.resize(face_img, (48, 48))
#                     input_vec = resized_face.flatten().reshape(1, -1) / 255.0

#                     pred_idx = model.predict(input_vec)[0]
#                     emotion = emotion_labels[pred_idx]

#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     cv2.putText(frame, emotion, (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 except Exception as e:
#                     cv2.putText(frame, "Face error", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     st.error(f"Face processing error: {e}")

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)

#     cap.release()
#     FRAME_WINDOW.image([])

# else:
#     st.info("👆 Click the checkbox above to start the webcam.")

#     # --- Browser-compatible fallback: camera input ---
#     st.subheader("📷 Browser Camera Snapshot (for deployed apps)")
#     image_data = st.camera_input("Take a picture using your webcam")

#     if image_data is not None:
#         st.success("✅ Image captured. Processing...")

#         img = Image.open(image_data).convert("L")  # Convert to grayscale
#         img = img.resize((48, 48))  # Match the SVM input format
#         input_vec = np.array(img).flatten().reshape(1, -1) / 255.0

#         try:
#             pred_idx = model.predict(input_vec)[0]
#             emotion = emotion_labels[pred_idx]
#             st.markdown(f"### 🧠 Predicted Emotion: **{emotion}**")
#         except Exception as e:
# #             st.error(f"Prediction error: {e}")
# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import joblib
# import os

# # --- Emotion Labels ---
# emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# # --- Function to reconstruct SVM model from parts ---
# def merge_model_parts(output_file='emotion_svm.pkl', part_prefix='emotion_svm.pkl.part'):
#     index = 1
#     try:
#         with open(output_file, 'wb') as output:
#             while True:
#                 part_file = f"{part_prefix}{index}"
#                 if not os.path.exists(part_file):
#                     break
#                 with open(part_file, 'rb') as pf:
#                     output.write(pf.read())
#                     st.info(f"Merged: {part_file}")
#                 index += 1
#         st.success("✅ Model reconstruction complete.")
#     except Exception as e:
#         st.error(f"❌ Error merging model parts: {e}")

# # --- Ensure model file exists ---
# model_path = "emotion_svm.pkl"
# if not os.path.exists(model_path):
#     st.warning("⚠️ Model file not found. Attempting to reconstruct...")
#     merge_model_parts()

# if not os.path.exists(model_path):
#     st.error("❌ Model still missing after reconstruction. Please upload all .part files.")
#     st.stop()

# # --- Load Haar Cascade for Face Detection ---
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # --- Load the trained SVM model ---
# @st.cache_resource
# def load_model():
#     try:
#         model = joblib.load(model_path)
#         st.success("✅ SVM model loaded successfully.")
#         return model
#     except Exception as e:
#         st.error(f"❌ Failed to load SVM model: {e}")
#         st.stop()

# model = load_model()

# # --- Streamlit UI ---
# st.title("😊 Real-time Emotion Detection (SVM Model)")
# run = st.checkbox("Start Webcam")

# FRAME_WINDOW = st.image([])
# motion_threshold = 800
# prev_gray = None

# if run:
#     # Try multiple webcam indexes to ensure access
#     for cam_index in [0, 1, 2]:
#         cap = cv2.VideoCapture(cam_index)
#         if cap.isOpened():
#             break
#     else:
#         st.error("❌ Could not open any webcam (tried indexes 0, 1, 2).")
#         st.stop()

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("⚠️ Failed to read frame from webcam.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Anti-spoofing: motion detection
#         motion = False
#         if prev_gray is not None:
#             diff = cv2.absdiff(prev_gray, gray)
#             score = np.sum(diff)
#             if score > motion_threshold:
#                 motion = True
#         prev_gray = gray

#         # Face Detection
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         if len(faces) == 0:
#             cv2.putText(frame, "No face detected", (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         elif not motion:
#             cv2.putText(frame, "Spoof detected! No motion", (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             for (x, y, w, h) in faces:
#                 face_img = gray[y:y+h, x:x+w]
#                 try:
#                     resized_face = cv2.resize(face_img, (48, 48))
#                     input_vec = resized_face.flatten().reshape(1, -1) / 255.0

#                     pred_idx = model.predict(input_vec)[0]
#                     emotion = emotion_labels[pred_idx]

#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     cv2.putText(frame, emotion, (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 except Exception as e:
#                     cv2.putText(frame, "Face error", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                     st.error(f"Face processing error: {e}")

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)

#     cap.release()
#     FRAME_WINDOW.image([])

# else:
#     st.info("👆 Click the checkbox above to start the webcam for real-time emotion detection.")
# 

# 

import streamlit as st
import cv2
import numpy as np
import os
import joblib
from PIL import Image

# --- Emotion Labels ---
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# --- Load Haar Cascade (for face detection) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Reconstruct SVM model if split ---
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

# --- Ensure model exists ---
model_path = "emotion_svm.pkl"
if not os.path.exists(model_path):
    st.warning("⚠️ Model file not found. Attempting to reconstruct...")
    merge_model_parts()

if not os.path.exists(model_path):
    st.error("❌ Model still missing after reconstruction. Please upload all .part files.")
    st.stop()

# --- Load SVM model ---
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

# --- Webcam Support Table ---
st.markdown("### 📊 Webcam Support Matrix")
st.markdown("""
| Deployment Mode      | Webcam Method       | Works? |
|----------------------|---------------------|--------|
| **Local Streamlit**  | `cv2.VideoCapture`  | ✅ Yes |
| **Deployed (Cloud)** | `cv2.VideoCapture`  | ❌ No  |
| **Deployed (Cloud)** | `st.camera_input()` | ✅ Yes |
""")

# --- Webcam Toggle ---
run = st.toggle("▶ Start Webcam")
FRAME_WINDOW = st.image([])
motion_threshold = 800
prev_gray = None

# --- Try OpenCV webcam (LOCAL) ---
webcam_available = False
cap = None

if run:
    for cam_index in [0, 1, 2]:
        st.write(f"🔍 Trying camera index {cam_index}...")
        backend = cv2.CAP_DSHOW if os.name == 'nt' else 0
        temp_cap = cv2.VideoCapture(cam_index, backend)
        if temp_cap.isOpened():
            cap = temp_cap
            webcam_available = True
            st.success(f"✅ Webcam opened at index {cam_index}")
            break
        else:
            temp_cap.release()
            st.warning(f"⚠️ Camera index {cam_index} not available.")

# --- Local webcam loop using OpenCV ---
if run and webcam_available:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.sum(diff)
            if score > motion_threshold:
                motion = True
        prev_gray = gray

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not motion:
            cv2.putText(frame, "Spoof detected (no motion)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                try:
                    resized = cv2.resize(face_img, (48, 48))
                    input_vec = resized.flatten().reshape(1, -1) / 255.0
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

# --- Fallback: Browser snapshot for cloud deployment ---
elif run and not webcam_available:
    st.warning("⚠️ Webcam access not available. Using browser-based snapshot as fallback.")
    snapshot = st.camera_input("📸 Take a picture using your webcam")

    if snapshot is not None:
        st.success("✅ Image captured! Analyzing...")

        img = Image.open(snapshot).convert("L")
        img = img.resize((48, 48))
        input_vec = np.array(img).flatten().reshape(1, -1) / 255.0

        try:
            pred_idx = model.predict(input_vec)[0]
            emotion = emotion_labels[pred_idx]
            st.markdown(f"### 🧠 Detected Emotion: **{emotion}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
