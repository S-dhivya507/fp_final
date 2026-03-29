"""
Real-Time Stress & Emotion Detection Web Application
Flask Web Dashboard for Final Year Project
"""

import os

from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pickle
import librosa
from scipy.io.wavfile import write, read
from tensorflow import keras
from collections import Counter
import time
from datetime import datetime
import shutil
import subprocess

try:
    import sounddevice as sd
except Exception:
    sd = None

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
IS_RENDER = bool(os.getenv("RENDER")) or bool(os.getenv("RENDER_SERVICE_ID"))

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
    face_model = keras.models.load_model(os.path.join(BASE_DIR, "models", "face_emotion_model.h5"), compile=False)
    voice_model = keras.models.load_model(os.path.join(BASE_DIR, "voice_module", "voice_emotion_model.h5"), compile=False)
    with open(os.path.join(BASE_DIR, "voice_module", "label_encoder.pkl"), "rb") as f:
        voice_encoder = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    # Capture expected voice input shape if available (e.g., (None, 40, 174, 1))
    voice_input_shape = getattr(voice_model, "input_shape", None)
    voice_expected_mfcc = None
    voice_expected_frames = None
    if voice_input_shape and len(voice_input_shape) >= 3:
        voice_expected_mfcc = voice_input_shape[1]
        voice_expected_frames = voice_input_shape[2]
    models_loaded = True
    print("Models loaded")
except Exception as e:
    print("Model loading error:", e)
    models_loaded = False
    voice_encoder = None

# ================= CONFIG =================
SAMPLE_RATE = 22050
AUDIO_DURATION = 10
VIDEO_DURATION = 10

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


def face_counts_to_percentages(face_counts):
    total = sum(face_counts.values()) if face_counts else 0
    if not total:
        return {label: 0 for label in face_labels}
    return {label: round((face_counts.get(label, 0) / total) * 100, 2) for label in face_labels}


def cleanup_uploads(keep_names):
    keep = set(keep_names or [])
    for name in os.listdir(app.config["UPLOAD_FOLDER"]):
        if name in keep:
            continue
        path = os.path.join(app.config["UPLOAD_FOLDER"], name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass


# ================= VIDEO =================
def detect_faces_multi(gray):
    if not models_loaded or face_cascade is None or face_cascade.empty():
        return (), gray

    gray_eq = clahe.apply(gray)
    passes = [
        (gray_eq, 1.05, 3),
        (gray_eq, 1.1, 4),
        (gray, 1.05, 3),
    ]

    h, w = gray.shape[:2]
    if h < 60 or w < 60:
        return (), gray_eq
    min_w = max(int(w * 0.2), 80)
    min_h = max(int(h * 0.2), 80)
    max_w = int(w * 0.8)
    max_h = int(h * 0.8)
    min_w = min(min_w, w)
    min_h = min(min_h, h)
    max_w = min(max_w, w)
    max_h = min(max_h, h)

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


def detect_emotions_from_video_file(video_path, duration=VIDEO_DURATION, frame_step=1, allow_label_video=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Video file not accessible")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
    max_frames = int(duration * fps) if duration else total_frames
    if total_frames and max_frames > total_frames:
        max_frames = total_frames

    writer = None
    labeled_path = None
    base = os.path.splitext(os.path.basename(video_path))[0]
    labeled_name = f"{base}_labeled.mp4"
    labeled_path = os.path.join(app.config["UPLOAD_FOLDER"], labeled_name)
    if allow_label_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(labeled_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = None
            labeled_path = None

    face_counter = Counter()
    face_frames = 0
    frame_idx = 0

    while True:
        if max_frames and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_step > 1 and (frame_idx % frame_step) != 0:
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

        # Use MFCC-only features to match the model input shape.
        n_mfcc = voice_expected_mfcc or 40
        target_frames = voice_expected_frames or 174

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < target_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :target_frames]

        features = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)

        probs = voice_model.predict(features, verbose=0)[0]
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


def voice_probs_to_percentages(voice_probs):
    if voice_probs is None or voice_encoder is None:
        return {label: 0 for label in face_labels}
    results = {label: 0 for label in face_labels}
    for idx, label in enumerate(voice_encoder.classes_):
        if idx < len(voice_probs) and label in results:
            results[label] = round(float(voice_probs[idx] * 100), 2)
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

@app.route("/models/<path:filename>")
def face_api_models(filename):
    return send_from_directory(os.path.join(app.root_path, "face_api_models"), filename)


@app.route("/Roboto-Regular.ttf")
def roboto_font():
    return send_from_directory("C:\\Windows\\Fonts", "arial.ttf")


@app.route("/api/start-recording", methods=["POST"])
def start_recording():
    try:
        if sd is None:
            return jsonify({
                "success": False,
                "error": "Live server audio capture is not available in deployment. Use the browser live capture."
            }), 400
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

        if voice_probs is None:
            if voice_encoder is not None:
                voice_probs = np.zeros(len(voice_encoder.classes_), dtype=np.float32)
            else:
                voice_probs = np.zeros(len(face_labels), dtype=np.float32)

        fused_emotions = fuse_emotions(face_emotions, voice_probs)
        stress_level = calculate_stress_level(fused_emotions)
        face_emotions_percent = face_counts_to_percentages(face_emotions)
        voice_emotions_percent = voice_probs_to_percentages(voice_probs)

        response_data = {
            "success": True,
            "face_emotions": face_emotions_percent,
            "voice_probs": voice_probs.tolist() if voice_probs is not None else None,
            "voice_emotions": voice_emotions_percent,
            "voice_labels": list(voice_encoder.classes_) if voice_encoder is not None else [],
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
        keep_files = [audio_name]
        if video_name:
            keep_files.append(video_name)
        cleanup_uploads(keep_files)
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
            if IS_RENDER:
                # Skip heavy face analysis on Render free tier to avoid timeouts.
                face_emotions = {}
                face_frames = 0
            else:
                try:
                    face_emotions, face_frames, labeled_path = detect_emotions_from_video_file(
                        analysis_path,
                        VIDEO_DURATION,
                        frame_step=1,
                        allow_label_video=True
                    )
                except Exception as e:
                    print("Upload video decode error:", e)
                    face_emotions = {}
                    face_frames = 0
                    video_error = f"Video decode failed: {e}"

            if IS_RENDER:
                video_url = None
            else:
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

        if voice_probs is None:
            if voice_encoder is not None:
                voice_probs = np.zeros(len(voice_encoder.classes_), dtype=np.float32)
            else:
                voice_probs = np.zeros(len(face_labels), dtype=np.float32)

        fused_emotions = fuse_emotions(face_emotions, voice_probs)
        stress_level = calculate_stress_level(fused_emotions)
        face_emotions_percent = face_counts_to_percentages(face_emotions)
        voice_emotions_percent = voice_probs_to_percentages(voice_probs)

        response_data = {
            "success": True,
            "face_emotions": face_emotions_percent,
            "voice_probs": voice_probs.tolist() if voice_probs is not None else None,
            "voice_emotions": voice_emotions_percent,
            "voice_labels": list(voice_encoder.classes_) if voice_encoder is not None else [],
            "fused_emotions": fused_emotions,
            "stress_level": stress_level,
            "video_file": video_url,
            "audio_file": audio_url,
            "face_detected": face_frames > 0 if video_file else False,
            "face_frames": face_frames,
            "video_preview_supported": video_preview_supported,
            "video_error": video_error,
        }

        keep_files = []
        if video_url:
            keep_files.append(os.path.basename(video_url))
        if audio_url:
            keep_files.append(os.path.basename(audio_url))
        cleanup_uploads(keep_files)
        return jsonify(convert_numpy_to_python(response_data))

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ================= MAIN =================
if __name__ == "__main__":
    print("Server running on http://localhost:5000")
    app.run(debug=True)
