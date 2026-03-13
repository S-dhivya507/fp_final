import cv2
import numpy as np
import time
from collections import Counter
from tensorflow.keras.models import load_model

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("../models/face_emotion_model.h5")

# Emotion Labels (same order as training folders)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                  'Neutral', 'Sad', 'Surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

start_time = time.time()
duration = 1 * 60   # 15 minutes in seconds

emotion_counter = Counter()
total_predictions = 0

print("🎥 Recording Started for 15 Minutes...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        predictions = model.predict(face, verbose=0)
        max_index = np.argmax(predictions[0])
        emotion = emotion_labels[max_index]
        confidence = predictions[0][max_index] * 100

        # Count emotion
        emotion_counter[emotion] += 1
        total_predictions += 1

        # Draw Rectangle + Emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame,
                    f"{emotion} ({confidence:.2f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Stop after 15 minutes
    if time.time() - start_time > duration:
        print("\n⏰ 15 Minutes Completed!")
        break

    # Press Q to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Overall Emotion Calculation
# -----------------------------
print("\n📊 Overall Emotion Analysis:\n")

if total_predictions > 0:
    for emotion in emotion_labels:
        percent = (emotion_counter[emotion] / total_predictions) * 100
        print(f"{emotion} : {percent:.2f}%")

    overall_emotion = emotion_counter.most_common(1)[0][0]
    overall_percent = (emotion_counter[overall_emotion] / total_predictions) * 100

    print("\n===================================")
    print(f"🎯 Overall Emotion: {overall_emotion}")
    print(f"📈 Confidence: {overall_percent:.2f}%")
else:
    print("No faces detected during the session.")