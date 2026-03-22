"""
Real-Time Stress & Emotion Detection Web Application
Flask Web Dashboard for Final Year Project
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import pickle
import librosa
import sounddevice as sd
from scipy.io.wavfile import write, read
from tensorflow.keras.models import load_model
from collections import Counter
import time
from datetime import datetime
import shutil
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Serve logo at root as a fallback to avoid 404s from cached/legacy URLs.
@app.route("/logo.png")
def logo_png():
    return send_from_directory(os.path.join(app.root_path, "static", "img"), "logo.png")

@app.route("/logo.svg")
def logo_svg():
    return send_from_directory(os.path.join(app.root_path, "static", "img"), "logo.svg")

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    face_model = load_model(os.path.join(BASE_DIR, "models", "face_emotion_model.h5"))
    voice_model = load_model(os.path.join(BASE_DIR, "voice_module", "shemo_emotion_model.h5"))
    with open(os.path.join(BASE_DIR, "voice_module", "label_encoder.pkl"), "rb") as f:
        voice_encoder = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    models_loaded = True
    print("Models loaded")
except Exception as e:
    print("Model loading error:", e)
    models_loaded = False
    voice_encoder = None

# ================= CONFIG =================
SAMPLE_RATE = 22050
AUDIO_DURATION = 60
VIDEO_DURATION = 60

face_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ================= UTILITY =================
def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    return obj


# ================= VIDEO =================
def detect_faces_multi(gray):
    gray_eq = clahe.apply(gray)
    passes = [
        (gray_eq, 1.05, 3),
        (gray_eq, 1.1, 4),
        (gray, 1.05, 3),
    ]

    h, w = gray.shape[:2]
    min_w = max(int(w * 0.2), 80)
    min_h = max(int(h * 0.2), 80)
    max_w = int(w * 0.8)
    max_h = int(h * 0.8)

    for img, scale, neighbors in passes:
        faces = face_cascade.detectMultiScale(
            img,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(min_w, min_h),
            maxSize=(max_w, max_h)
        )
        if len(faces) > 0:
            return faces, img

    return (), gray_eq


def update_face_counter(frame, face_counter):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, gray_used = detect_faces_multi(gray)

    emotions = detect_face_emotions_from_faces(gray_used, faces)
    for item in emotions:
        face_counter[item["emotion"]] += 1
    return len(emotions)


def detect_face_emotions_from_faces(gray_used, faces):
    # Always label only one face (largest) to avoid multiple boxes
    if len(faces) == 0:
        return []
    (x, y, w, h) = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray_used.shape[1], x + w + pad_x)
    y2 = min(gray_used.shape[0], y + h + pad_y)

    face = gray_used[y1:y2, x1:x2]
    face = cv2.resize(face, (48, 48)) / 255.0
    face = face.reshape(1, 48, 48, 1)

    pred = face_model.predict(face, verbose=0)
    emotion = face_labels[np.argmax(pred)]

    return [{"box": (x1, y1, x2 - x1, y2 - y1), "emotion": emotion}]


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(0)


def detect_emotions_from_video(duration=VIDEO_DURATION):
    cap = open_camera()
    if not cap.isOpened():
        raise Exception("Camera not accessible")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 24)

    face_counter = Counter()
    face_frames = 0
    start_time = time.time()

    for _ in range(10):
        cap.read()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        if update_face_counter(frame, face_counter) > 0:
            face_frames += 1

    cap.release()
    return dict(face_counter), face_frames


def detect_emotions_from_video_file(video_path, duration=VIDEO_DURATION):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Video file not accessible")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0

    writer = None
    labeled_path = None
    base = os.path.splitext(os.path.basename(video_path))[0]
    labeled_name = f"{base}_labeled.mp4"
    labeled_path = os.path.join(app.config["UPLOAD_FOLDER"], labeled_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(labeled_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer = None
        labeled_path = None

    face_counter = Counter()
    face_frames = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, gray_used = detect_faces_multi(gray)
        emotions = detect_face_emotions_from_faces(gray_used, faces)
        if emotions:
            face_frames += 1
            for item in emotions:
                face_counter[item["emotion"]] += 1
                (x, y, w, h) = item["box"]
                if writer:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (59, 130, 246), 2)
                    cv2.putText(
                        frame,
                        item["emotion"],
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (59, 130, 246),
                        2
                    )

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    return dict(face_counter), face_frames, labeled_path


def capture_video_with_emotions(duration=VIDEO_DURATION, output_base=None):
    cap = open_camera()
    if not cap.isOpened():
        raise Exception("Camera not accessible")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0

    writer = None
    video_name = None
    if output_base:
        video_name = f"{output_base}.mp4"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = None
            video_name = None
            print("Video writer not available. Skipping video file save.")

    face_counter = Counter()
    face_frames = 0
    start_time = time.time()

    for _ in range(10):
        cap.read()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, gray_used = detect_faces_multi(gray)
        emotions = detect_face_emotions_from_faces(gray_used, faces)
        if emotions:
            face_frames += 1
            for item in emotions:
                face_counter[item["emotion"]] += 1
                (x, y, w, h) = item["box"]
                if writer:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (59, 130, 246), 2)
                    cv2.putText(
                        frame,
                        item["emotion"],
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (59, 130, 246),
                        2
                    )

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    return dict(face_counter), video_name, face_frames


def convert_video_for_browser(input_path, output_dir, base_name):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    targets = [
        ("mp4", "libx264"),
        ("webm", "libvpx"),
    ]

    for ext, codec in targets:
        output_name = f"{base_name}_browser.{ext}"
        output_path = os.path.join(output_dir, output_name)
        command = [
            ffmpeg,
            "-y",
            "-i", input_path,
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
        ]
        if ext == "mp4":
            command += ["-movflags", "+faststart"]
        command.append(output_path)

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_name
        except Exception as e:
            print("FFmpeg conversion error:", e)
            return None

    return None


def convert_video_for_analysis(input_path, output_dir, base_name):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    output_name = f"{base_name}_analysis.mp4"
    output_path = os.path.join(output_dir, output_name)
    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        print("FFmpeg analysis conversion error:", e)
    return None


def convert_audio_to_wav(input_path, output_dir, base_name):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    output_name = f"{base_name}_audio.wav"
    output_path = os.path.join(output_dir, output_name)
    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        output_path
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        print("FFmpeg audio conversion error:", e)
    return None


# ================= AUDIO =================
def detect_emotions_from_audio(audio_file):
    try:
        ext = os.path.splitext(audio_file)[1].lower()
        audio = None
        sr = SAMPLE_RATE

        if ext == ".wav":
            try:
                sr_native, audio = read(audio_file)
                sr = sr_native
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                    sr = SAMPLE_RATE
            except Exception:
                audio = None

        if audio is None:
            audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
            audio = librosa.util.normalize(audio)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        combined = np.vstack((mfcc, chroma, mel))

        if combined.shape[1] < 200:
            combined = np.pad(combined, ((0, 0), (0, 200 - combined.shape[1])))
        else:
            combined = combined[:, :200]

        combined = combined.reshape(1, combined.shape[0], combined.shape[1], 1)

        probs = voice_model.predict(combined, verbose=0)[0]
        return probs

    except Exception as e:
        print("Audio error:", e)
        return None


# ================= FUSION =================
def fuse_emotions(face_data, voice_probs):
    total_face = sum(face_data.values()) if face_data else 0
    results = {}

    for emotion in face_labels:
        face_val = (face_data.get(emotion, 0) / total_face * 100) if total_face else 0

        if voice_probs is not None and voice_encoder and emotion in voice_encoder.classes_:
            idx = list(voice_encoder.classes_).index(emotion)
            voice_val = float(voice_probs[idx] * 100)
        else:
            voice_val = 0

        fused = face_val * 0.6 + voice_val * 0.4
        results[emotion] = round(fused, 2)

    return results


def calculate_stress_level(scores):
    stress_emotions = ["Angry", "Disgust", "Fear", "Sad"]
    stress_score = sum(scores.get(e, 0) for e in stress_emotions)

    if stress_score > 40:
        return {"level": "High", "color": "#dc2626", "score": round(stress_score, 2)}
    if stress_score > 20:
        return {"level": "Medium", "color": "#ea580c", "score": round(stress_score, 2)}
    return {"level": "Low", "color": "#16a34a", "score": round(stress_score, 2)}


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/.well-known/appspecific/<path:filename>")
def app_specific(filename):
    well_known_dir = os.path.join(os.path.dirname(__file__), ".well-known", "appspecific")
    return send_from_directory(well_known_dir, filename)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico")


@app.route("/Roboto-Regular.ttf")
def roboto_font():
    return send_from_directory("C:\\Windows\\Fonts", "arial.ttf")


@app.route("/api/start-recording", methods=["POST"])
def start_recording():
    try:
        print("Starting video...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_emotions, video_name, face_frames = capture_video_with_emotions(5, f"{timestamp}_capture")
        video_preview_supported = False
        video_error = None

        if video_name:
            captured_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)
            converted_name = convert_video_for_browser(captured_path, app.config["UPLOAD_FOLDER"], f"{timestamp}_capture")
            if converted_name:
                video_name = converted_name
                video_preview_supported = True
            else:
                video_name = None
                video_error = "Install FFmpeg to convert the capture into a browser-supported format."

        print("Video finished, recording audio...")

        audio_name = f"{timestamp}_audio.wav"
        audio_file = os.path.join(app.config["UPLOAD_FOLDER"], audio_name)

        recording = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        write(audio_file, SAMPLE_RATE, recording)

        print("Audio saved:", audio_file)

        voice_probs = detect_emotions_from_audio(audio_file)

        print("Voice probs:", voice_probs)

        fused_emotions = fuse_emotions(face_emotions, voice_probs)
        stress_level = calculate_stress_level(fused_emotions)

        response_data = {
            "success": True,
            "face_emotions": face_emotions,
            "voice_probs": voice_probs.tolist() if voice_probs is not None else None,
            "fused_emotions": fused_emotions,
            "stress_level": stress_level,
            "video_file": f"/uploads/{video_name}" if video_name else None,
            "audio_file": f"/uploads/{audio_name}",
            "face_detected": face_frames > 0,
            "face_frames": face_frames,
            "video_preview_supported": video_preview_supported,
            "video_error": video_error,
        }

        response_data = convert_numpy_to_python(response_data)
        return jsonify(response_data)

    except Exception as e:
        print("FULL ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/quick-analysis", methods=["POST"])
def quick_analysis():
    try:
        fused_emotions = {
            "Angry": 10,
            "Disgust": 5,
            "Fear": 15,
            "Happy": 30,
            "Neutral": 25,
            "Sad": 10,
            "Surprise": 5,
        }

        stress_level = calculate_stress_level(fused_emotions)

        response_data = {
            "success": True,
            "face_emotions": fused_emotions,
            "voice_probs": list(fused_emotions.values()),
            "fused_emotions": fused_emotions,
            "stress_level": stress_level,
            "video_file": None,
            "audio_file": None,
            "face_detected": True,
            "face_frames": 0,
        }

        return jsonify(convert_numpy_to_python(response_data))

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/upload-analysis", methods=["POST"])
def upload_analysis():
    try:
        if not models_loaded:
            return jsonify({"success": False, "error": "Models not loaded. Check server logs."}), 500

        video_file = request.files.get("video")
        audio_file = request.files.get("audio")

        if not video_file and not audio_file:
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        face_emotions = {}
        voice_probs = None
        video_url = None
        audio_url = None
        face_frames = 0
        video_preview_supported = False
        video_error = None

        if video_file:
            video_name = secure_filename(video_file.filename) or f"video_{timestamp}.mp4"
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{video_name}")
            video_file.save(video_path)
            analysis_path = video_path
            ext = os.path.splitext(video_path)[1].lower()
            if ext != ".mp4":
                converted_path = convert_video_for_analysis(video_path, app.config["UPLOAD_FOLDER"], f"{timestamp}_upload")
                if converted_path:
                    analysis_path = converted_path

            labeled_path = None
            try:
                face_emotions, face_frames, labeled_path = detect_emotions_from_video_file(analysis_path, VIDEO_DURATION)
            except Exception as e:
                print("Upload video decode error:", e)
                face_emotions = {}
                face_frames = 0
                video_error = f"Video decode failed: {e}"

            preview_source = labeled_path if labeled_path else video_path
            converted_name = convert_video_for_browser(preview_source, app.config["UPLOAD_FOLDER"], f"{timestamp}_upload")
            if converted_name:
                video_url = f"/uploads/{converted_name}"
                video_preview_supported = True
            else:
                video_url = None
                if not video_error:
                    video_error = "Install FFmpeg to convert uploads into a browser-supported format."

        if audio_file:
            audio_name = secure_filename(audio_file.filename) or f"audio_{timestamp}.wav"
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{audio_name}")
            audio_file.save(audio_path)
            audio_url = f"/uploads/{timestamp}_{audio_name}"

            audio_ext = os.path.splitext(audio_path)[1].lower()
            if audio_ext != ".wav":
                converted_audio = convert_audio_to_wav(audio_path, app.config["UPLOAD_FOLDER"], f"{timestamp}_upload")
                if converted_audio:
                    audio_path = converted_audio
                    audio_url = f"/uploads/{os.path.basename(converted_audio)}"

            voice_probs = detect_emotions_from_audio(audio_path)
        elif video_file:
            extracted_audio = convert_audio_to_wav(video_path, app.config["UPLOAD_FOLDER"], f"{timestamp}_upload_from_video")
            if extracted_audio:
                voice_probs = detect_emotions_from_audio(extracted_audio)
                audio_url = f"/uploads/{os.path.basename(extracted_audio)}"

        fused_emotions = fuse_emotions(face_emotions, voice_probs)
        stress_level = calculate_stress_level(fused_emotions)

        response_data = {
            "success": True,
            "face_emotions": face_emotions,
            "voice_probs": voice_probs.tolist() if voice_probs is not None else None,
            "fused_emotions": fused_emotions,
            "stress_level": stress_level,
            "video_file": video_url,
            "audio_file": audio_url,
            "face_detected": face_frames > 0 if video_file else False,
            "face_frames": face_frames,
            "video_preview_supported": video_preview_supported,
            "video_error": video_error,
        }

        return jsonify(convert_numpy_to_python(response_data))

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ================= MAIN =================
if __name__ == "__main__":
    print("Server running on http://localhost:5000")
    app.run(debug=True)
