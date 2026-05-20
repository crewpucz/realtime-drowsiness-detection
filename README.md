# Drowsiness Detection System

Real-time driver drowsiness detection using CNN + MediaPipe facial landmarks, built with Streamlit.

## How It Works

1. MediaPipe detects 468 facial landmarks in real-time
2. Eye regions are automatically cropped from the landmarks
3. A MobileNetV2 CNN classifies each eye as Open or Closed
4. Alert is triggered if both eyes remain closed for 10 consecutive frames

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `drowsiness_model.h5` | Trained MobileNetV2 model |
| `face_landmarker.task` | MediaPipe face landmark model |
| `class_indices.json` | Class label mapping |
| `style.css` | Custom UI styles |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
