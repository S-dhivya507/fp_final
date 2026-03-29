import os
import re
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical

RAVDESS_ROOT = r"C:\Users\DELL\Downloads\STREES_DETECTION-main\STREES_DETECTION-main\dataset\audio_speech_actors_01-24"
SHEMO_ROOT = r"C:\Users\DELL\Downloads\STREES_DETECTION-main\STREES_DETECTION-main\dataset\speech"
OUTPUT_MODEL = r"C:\Users\DELL\Downloads\STREES_DETECTION-main\STREES_DETECTION-main\voice_module\voice_emotion_model.h5"
OUTPUT_ENCODER = r"C:\Users\DELL\Downloads\STREES_DETECTION-main\STREES_DETECTION-main\voice_module\label_encoder.pkl"

# RAVDESS emotion mapping
# 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Neutral",
    "03": "Happy",
    "04": "Sad",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}

# ShEMO file emotion codes (4th character in filename, e.g., F01A01.wav)
SHEMO_MAP = {
    "A": "Anger",
    "N": "Neutral",
    "S": "Sad",
    "W": "Surprise",
    "H": "Happy",
    "F": "Fear",
}

N_MFCC = 40
TARGET_FRAMES = 174
SAMPLE_RATE = 22050


def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < TARGET_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TARGET_FRAMES]
    return mfcc


def iter_ravdess_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith('.wav'):
                continue
            yield os.path.join(dirpath, name)


def label_from_filename(filename):
    # RAVDESS file pattern: 03-01-05-02-02-02-12.wav
    parts = os.path.basename(filename).split('-')
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code)


def iter_shemo_files(root):
    for gender in ["female", "male"]:
        gpath = os.path.join(root, gender)
        if not os.path.isdir(gpath):
            continue
        for name in os.listdir(gpath):
            if not name.lower().endswith(".wav"):
                continue
            yield os.path.join(gpath, name)


def shemo_label_from_filename(filename):
    base = os.path.basename(filename)
    if len(base) < 4:
        return None
    code = base[3].upper()
    return SHEMO_MAP.get(code)


def main():
    X = []
    y = []

    # RAVDESS
    for path in iter_ravdess_files(RAVDESS_ROOT):
        label = label_from_filename(path)
        if not label:
            continue
        try:
            mfcc = extract_mfcc(path)
        except Exception:
            continue
        X.append(mfcc)
        y.append(label)

    # ShEMO
    for path in iter_shemo_files(SHEMO_ROOT):
        label = shemo_label_from_filename(path)
        if not label:
            continue
        try:
            mfcc = extract_mfcc(path)
        except Exception:
            continue
        X.append(mfcc)
        y.append(label)

    if not X:
        raise RuntimeError("No audio files found. Check RAVDESS_ROOT path.")

    X = np.array(X, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, TARGET_FRAMES, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(y_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save(OUTPUT_MODEL)
    with open(OUTPUT_ENCODER, 'wb') as f:
        pickle.dump(le, f)

    print("Saved model to:", OUTPUT_MODEL)
    print("Saved label encoder to:", OUTPUT_ENCODER)
    print("Classes:", list(le.classes_))


if __name__ == '__main__':
    main()
