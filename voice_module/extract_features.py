import librosa
import numpy as np

IMG_SIZE = (128, 128)

def extract_spectrogram(audio, sr=22050):

    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        log_mel = librosa.power_to_db(mel_spec)

        # Resize to CNN input shape
        log_mel = np.resize(log_mel, IMG_SIZE)

        return log_mel

    except:
        return None