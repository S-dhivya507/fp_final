import os
import numpy as np
import librosa
import pickle

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "speech")

# -----------------------------
# Extract Emotion from Filename
# -----------------------------
def extract_label(filename):
    emotion_char = filename[3]  # F03S02.wav -> S

    mapping = {
        "A": "Anger",
        "F": "Fear",
        "H": "Happy",
        "S": "Sad",
        "W": "Surprise",
        "N": "Neutral"
    }

    return mapping.get(emotion_char)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    combined = np.vstack((mfcc, chroma, mel))

    max_len = 200
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(combined, ((0,0),(0,pad_width)), mode='constant')
    else:
        combined = combined[:, :max_len]

    return combined

# -----------------------------
# Load All Files
# -----------------------------
X = []
y = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            label = extract_label(file)
            if label:
                path = os.path.join(root, file)
                features = extract_features(path)
                X.append(features)
                y.append(label)

print("Dataset Distribution:", Counter(y))

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=7, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save("shemo_emotion_model.h5")

print("\n✅ SHemo Training Completed Successfully!")