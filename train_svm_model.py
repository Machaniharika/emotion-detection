import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ✅ Set your dataset path here
data_dir = r"D:/emotion detection/train" # raw string for Windows path
emotions = ['angry', 'happy', 'neutral', 'sad']  # Make sure this matches your folder names

X, y = [], []

print("📥 Loading images from:", data_dir)

for idx, emotion in enumerate(emotions):
    folder = os.path.join(data_dir, emotion)
    print(f"Looking in folder: {folder}")  # Debug: Print the folder path
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue
    
    # List files in the folder
    files = os.listdir(folder)
    # print("Files in folder:", files)  # Debug: Print files in the folder
    
    for img_name in files:
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Image not loaded properly")
            img = cv2.resize(img, (48, 48))
            X.append(img.flatten() / 255.0)
            y.append(idx)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {len(emotions)} emotions.")

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train SVM Model
print("🚀 Training SVM model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# ✅ Evaluate
print("📊 Evaluation Report:")
print(classification_report(y_test, model.predict(X_test), target_names=emotions))

# ✅ Save model
joblib.dump(model, "emotion_svm.pkl")
print("✅ Model saved as 'emotion_svm.pkl'")
