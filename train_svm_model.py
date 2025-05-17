# import os
# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib

# # ✅ Set your dataset path here
# data_dir = r"/detection/train" # raw string for Windows path
# emotions = ['angry', 'happy', 'neutral', 'sad']  # Make sure this matches your folder names

# X, y = [], []

# print("📥 Loading images from:", data_dir)

# for idx, emotion in enumerate(emotions):
#     folder = os.path.join(data_dir, emotion)
#     print(f"Looking in folder: {folder}")  # Debug: Print the folder path
#     if not os.path.exists(folder):
#         print(f"⚠️ Folder not found: {folder}")
#         continue
    
#     # List files in the folder
#     files = os.listdir(folder)
#     # print("Files in folder:", files)  # Debug: Print files in the folder
    
#     for img_name in files:
#         img_path = os.path.join(folder, img_name)
#         try:
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 raise ValueError("Image not loaded properly")
#             img = cv2.resize(img, (48, 48))
#             X.append(img.flatten() / 255.0)
#             y.append(idx)
#         except Exception as e:
#             print(f"⚠️ Skipping {img_path}: {e}")

# X = np.array(X)
# y = np.array(y)

# print(f"✅ Loaded {len(X)} samples across {len(emotions)} emotions.")

# # ✅ Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Train SVM Model
# print("🚀 Training SVM model...")
# model = SVC(kernel='linear', probability=True)
# model.fit(X_train, y_train)

# # ✅ Evaluate
# print("📊 Evaluation Report:")
# print(classification_report(y_test, model.predict(X_test), target_names=emotions))

# # ✅ Save model
# joblib.dump(model, "emotion_svm.pkl")
# print("✅ Model saved as 'emotion_svm.pkl'")


# def split_file(input_file, part_size_mb=25):
#     part_size = part_size_mb * 1024 * 1024  # bytes
#     with open(input_file, "rb") as f:
#         i = 1
#         while True:
#             chunk = f.read(part_size)
#             if not chunk:
#                 break
#             with open(f"emotion_svm_part_{i}.pkl", "wb") as part_file:
#                 part_file.write(chunk)
#             print(f"✅ Saved part {i}")
#             i += 1

# split_file("emotion_svm.pkl", part_size_mb=25)

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ✅ Set your dataset path here
data_dir = "D:/emotion detection/train"  # Make sure this folder contains angry/, happy/, neutral/, sad/
emotions = ['angry', 'happy', 'neutral', 'sad']

X, y = [], []

print("📥 Loading images from:", data_dir)

for idx, emotion in enumerate(emotions):
    folder = os.path.join(data_dir, emotion)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):
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
print("✅ Model saved as emotion_svm.pkl")

# ✅ Split model into 25MB parts
def split_file(input_file, part_size_mb=25):
    part_size = part_size_mb * 1024 * 1024
    with open(input_file, "rb") as f:
        i = 1
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            part_filename = f"emotion_svm.pkl.part{i}"
            with open(part_filename, "wb") as part_file:
                part_file.write(chunk)
            print(f"✅ Saved: {part_filename}")
            i += 1

split_file("emotion_svm.pkl", part_size_mb=25)
