import streamlit as st
import cv2
import numpy as np
import joblib

# Load SVM model
@st.cache_resource
def load_model():
    try:
        return joblib.load("emotion_svm.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad']

def is_real_face(face):
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Check for brightness
        brightness = np.mean(gray)
        if brightness < 50 or brightness > 220:
            return False

        # Check for blur
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 80:
            return False

        return True
    except:
        return False

def extract_features(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    return resized.flatten().astype('float32') / 255.0

def detect_and_predict(frame, model):
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if is_real_face(roi):
                features = extract_features(roi).reshape(1, -1)
                pred = model.predict(features)[0]
                label = EMOTIONS[pred]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Spoof Detected", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame
    except Exception as e:
        st.warning(f"Error: {e}")
        return frame

def main():
    st.title("Real-Time Emotion Detection (Webcam Only)")
    model = load_model()
    if not model:
        return

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
        return

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Frame not available.")
            break
        frame = cv2.flip(frame, 1)
        processed = detect_and_predict(frame, model)
        FRAME_WINDOW.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
