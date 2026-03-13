import numpy as np
import librosa
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import time
import os

print("🔄 Loading SHemo Model...")
model = load_model("shemo_emotion_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("✅ Model Ready!\n")

DURATION = 1 * 60
SAMPLE_RATE = 22050

print("🎤 Speak Now...")

recording = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)

for i in range(DURATION):
    print("Recording" + "." * (i % 3 + 1))
    time.sleep(1)

sd.wait()

print("\n✅ Recording Completed!")

filename = f"recorded_{int(time.time())}.wav"
write(filename, SAMPLE_RATE, recording)

print(f"📁 Saved As: {os.path.abspath(filename)}")

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

print("\n🔍 Extracting Features...")
features = extract_features(filename)
features = features.reshape(1, features.shape[0], features.shape[1], 1)

print("🧠 Predicting Emotion...\n")

predictions = model.predict(features)[0]

print("🎯 Emotion Probabilities:\n")

for i, prob in enumerate(predictions):
    emotion = encoder.classes_[i]
    bar = "█" * int(prob * 25)
    print(f"{emotion:10s} : {prob*100:6.2f}%  {bar}")

max_index = np.argmax(predictions)
final_emotion = encoder.classes_[max_index]
confidence = predictions[max_index] * 100

print("\n======================================")
print(f"🎉 Final Predicted Emotion : {final_emotion}")
print(f"🔥 Confidence Level        : {confidence:.2f}%")
print("======================================")