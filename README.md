# Stress Detection System

This repository contains the web application and source code for the Stress & Emotion Detection project.

## Large model files (download separately)
GitHub does not allow files larger than 100 MB. The trained model files are **not** included in this repo.

Place these files in the following locations:
- `models/face_emotion_model.h5`
- `voice_module/shemo_emotion_model.h5`
- `voice_module/label_encoder.pkl`

After downloading, verify that the files exist in those folders before running the app.

## Run locally (quick)
1. Create and activate a Python environment.
2. Install dependencies:
   ```
   pip install -r web_app/requirements.txt
   ```
3. Run the app:
   ```
   python web_app/app.py
   ```

## Notes
If you want the model files available in GitHub, use Git LFS or upload them to a separate storage (Google Drive / Release assets).
