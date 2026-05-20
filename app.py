import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import json
import time
import threading
import io
import wave
import struct
import math
import base64
import numpy as np
import av
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide",
)

# ============================================================
# LOAD CSS dari file terpisah
# ============================================================

def load_css(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

st.markdown(f"<style>{load_css('style.css')}</style>", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (cache supaya tidak reload setiap frame)
# ============================================================

@st.cache_resource
def load_resources():
    model = load_model('drowsiness_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

try:
    model, idx_to_class = load_resources()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ============================================================
# KONSTANTA
# ============================================================

IMG_SIZE = (64, 64)
CLOSED_FRAME_THRESHOLD = 10
CONFIDENCE_THRESHOLD = 0.6

LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390,
                    249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)


# ============================================================
# AUDIO ALERT
# ============================================================

def _generate_beep_b64(freq=880, duration=0.6, sample_rate=44100, volume=0.8):
    """Generate beep WAV in-memory dan return sebagai base64 string."""
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = []
        for i in range(num_samples):
            t = i / sample_rate
            # Envelope: fade in/out untuk menghindari klik
            env = min(1.0, t * 20, (duration - t) * 20)
            val = int(volume * env * 32767 * math.sin(2 * math.pi * freq * t))
            frames.append(struct.pack('<h', val))
        wf.writeframes(b''.join(frames))
    return base64.b64encode(buf.getvalue()).decode()

ALERT_BEEP_B64 = _generate_beep_b64(freq=880, duration=0.6)

def play_alert():
    """Inject audio element ke halaman untuk trigger beep di browser."""
    audio_html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/wav;base64,{ALERT_BEEP_B64}" type="audio/wav">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.closed_frames = 0
        self.status = 'AWAKE'
        self.left_state = 'open'
        self.right_state = 'open'
        self.left_conf = 1.0
        self.right_conf = 1.0
        self.face_detected = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._skip_frames = 2  # Proses setiap 3 frame (skip 2)

    def _get_eye_bbox(self, landmarks, indices, h, w, pad=10):
        pts = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ])
        x1 = max(0, pts[:, 0].min() - pad)
        x2 = min(w, pts[:, 0].max() + pad)
        y1 = max(0, pts[:, 1].min() - pad)
        y2 = min(h, pts[:, 1].max() + pad)
        return x1, y1, x2, y2

    def _predict_batch(self, left_eye_img, right_eye_img):
        """Prediksi kedua mata sekaligus dalam 1 batch — jauh lebih cepat."""
        left_resized = cv2.resize(left_eye_img, IMG_SIZE)
        left_rgb = cv2.cvtColor(left_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        right_resized = cv2.resize(right_eye_img, IMG_SIZE)
        right_rgb = cv2.cvtColor(right_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Batch 2 gambar sekaligus — 1x inference saja
        batch = np.stack([left_rgb, right_rgb], axis=0)
        probs = model(batch, training=False).numpy()  # Lebih cepat dari .predict()

        results = []
        for prob_val in probs:
            p = float(prob_val[0])
            if p > 0.5:
                results.append((idx_to_class.get(1, 'open'), p))
            else:
                results.append((idx_to_class.get(0, 'closed'), 1.0 - p))
        return results[0], results[1]

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # Frame skipping — hanya proses setiap N frame untuk performa
        self._frame_count += 1
        process_this_frame = (self._frame_count % (self._skip_frames + 1) == 0)

        if process_this_frame:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.detector.detect(mp_image)

            with self._lock:
                if results.face_landmarks:
                    self.face_detected = True
                    lms = results.face_landmarks[0]

                    # Crop mata
                    lx1, ly1, lx2, ly2 = self._get_eye_bbox(lms, LEFT_EYE_INDICES, h, w)
                    rx1, ry1, rx2, ry2 = self._get_eye_bbox(lms, RIGHT_EYE_INDICES, h, w)

                    left_eye = img[ly1:ly2, lx1:lx2]
                    right_eye = img[ry1:ry2, rx1:rx2]

                    if left_eye.size > 0 and right_eye.size > 0:
                        # Batch prediction — 1x inference untuk 2 mata
                        (self.left_state, self.left_conf), (self.right_state, self.right_conf) = \
                            self._predict_batch(left_eye, right_eye)

                        both_closed = (
                            self.left_state == 'closed' and self.left_conf >= CONFIDENCE_THRESHOLD and
                            self.right_state == 'closed' and self.right_conf >= CONFIDENCE_THRESHOLD
                        )

                        if both_closed:
                            self.closed_frames += 1
                        else:
                            self.closed_frames = max(0, self.closed_frames - 2)

                        if self.closed_frames >= CLOSED_FRAME_THRESHOLD:
                            self.status = 'ALERT'
                        elif self.closed_frames >= CLOSED_FRAME_THRESHOLD // 2:
                            self.status = 'WARNING'
                        else:
                            self.status = 'AWAKE'

                    # Simpan koordinat mata untuk drawing
                    self._last_eye_coords = (lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2)
                else:
                    self.face_detected = False

        # Drawing — selalu dilakukan (pakai hasil terakhir)
        with self._lock:
            if self.face_detected and hasattr(self, '_last_eye_coords'):
                lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = self._last_eye_coords

                lc = COLOR_GREEN if self.left_state == 'open' else COLOR_RED
                rc = COLOR_GREEN if self.right_state == 'open' else COLOR_RED
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), lc, 2)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), rc, 2)

                # Corner accents
                corner_len = 8
                for (x1, y1, x2, y2, c) in [(lx1, ly1, lx2, ly2, lc), (rx1, ry1, rx2, ry2, rc)]:
                    cv2.line(img, (x1, y1), (x1 + corner_len, y1), c, 3)
                    cv2.line(img, (x1, y1), (x1, y1 + corner_len), c, 3)
                    cv2.line(img, (x2, y1), (x2 - corner_len, y1), c, 3)
                    cv2.line(img, (x2, y1), (x2, y1 + corner_len), c, 3)
                    cv2.line(img, (x1, y2), (x1 + corner_len, y2), c, 3)
                    cv2.line(img, (x1, y2), (x1, y2 - corner_len), c, 3)
                    cv2.line(img, (x2, y2), (x2 - corner_len, y2), c, 3)
                    cv2.line(img, (x2, y2), (x2, y2 - corner_len), c, 3)

                # Status overlay
                if self.status == 'ALERT':
                    cv2.rectangle(img, (0, 0), (w, h), COLOR_RED, 6)
                    cv2.putText(img, 'DROWSY DETECTED', (w // 2 - 120, h - 25),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                elif self.status == 'WARNING':
                    cv2.putText(img, 'Eyes closing...', (w // 2 - 80, h - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)
            elif not self.face_detected:
                cv2.putText(img, 'No face detected', (w // 2 - 90, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ============================================================
# LAYOUT UTAMA
# ============================================================

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🚗 Drowsiness Detection System</div>
    <div class="header-subtitle">Real-time driver monitoring menggunakan CNN + MediaPipe facial landmarks</div>
    <div class="header-badge">
        <span>●</span> AI-Powered · MobileNetV2 · 99.5% Accuracy
    </div>
</div>
""", unsafe_allow_html=True)

# ── Cek model ──────────────────────────────────────────────
if not MODEL_LOADED:
    st.error(f"❌ Model tidak bisa diload: `{MODEL_ERROR}`")
    st.info("Pastikan file `drowsiness_model.h5`, `class_indices.json`, dan `face_landmarker.task` ada di folder yang sama dengan `app.py`")
    st.stop()

# ── Layout: video | panel kanan ────────────────────────────
col_video, col_panel = st.columns([3, 2], gap="large")

with col_video:
    st.markdown("""
    <div class="camera-label">
        <div class="camera-dot"></div>
        Live Camera Feed
    </div>
    """, unsafe_allow_html=True)

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
            {
                "urls": ["turn:a.relay.metered.ca:80"],
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            },
            {
                "urls": ["turn:a.relay.metered.ca:80?transport=tcp"],
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            },
            {
                "urls": ["turn:a.relay.metered.ca:443"],
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            },
            {
                "urls": ["turns:a.relay.metered.ca:443?transport=tcp"],
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            },
        ]
    })

    ctx = webrtc_streamer(
        key="drowsiness",
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_panel:
    st.markdown('<div class="section-title">📊 Detection Status</div>', unsafe_allow_html=True)

    # Placeholder untuk update real-time
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress_placeholder = st.empty()
    audio_placeholder = st.empty()




# ── Real-time status update ────────────────────────────────
if ctx.state.playing and ctx.video_processor:
    proc = ctx.video_processor
    last_alert_time = 0  # cooldown tracker

    while ctx.state.playing:
        with proc._lock:
            status = proc.status
            closed_frames = proc.closed_frames
            left_state = proc.left_state
            right_state = proc.right_state
            left_conf = proc.left_conf
            right_conf = proc.right_conf
            face_detected = proc.face_detected

        # Progress bar %
        pct = min(100, int(closed_frames / CLOSED_FRAME_THRESHOLD * 100))
        bar_class = 'progress-green' if pct < 40 else ('progress-yellow' if pct < 80 else 'progress-red')

        # Status card
        if not face_detected:
            status_html = """
            <div class="status-card status-inactive">
                <div class="status-icon">🔍</div>
                <div class="status-text" style="color:rgba(255,255,255,0.7)">No Face Detected</div>
                <div class="status-sub" style="color:rgba(255,255,255,0.4)">Posisikan wajah ke depan kamera</div>
            </div>"""
        elif status == 'ALERT':
            status_html = """
            <div class="status-card status-alert">
                <div class="status-icon">🚨</div>
                <div class="status-text" style="color:#fca5a5">DROWSY DETECTED!</div>
                <div class="status-sub" style="color:#fca5a5">Segera menepi dan istirahat</div>
            </div>"""
        elif status == 'WARNING':
            status_html = """
            <div class="status-card status-warning">
                <div class="status-icon">⚠️</div>
                <div class="status-text" style="color:#fcd34d">Eyes Closing...</div>
                <div class="status-sub" style="color:#fbbf24">Mulai terdeteksi mengantuk</div>
            </div>"""
        else:
            status_html = """
            <div class="status-card status-awake">
                <div class="status-icon">✅</div>
                <div class="status-text" style="color:#86efac">AWAKE</div>
                <div class="status-sub" style="color:#4ade80">Kondisi normal — tetap fokus</div>
            </div>"""

        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # Audio alert — beep setiap 2 detik saat ALERT
        now = time.time()
        if status == 'ALERT' and (now - last_alert_time) >= 2.0:
            with audio_placeholder:
                play_alert()
            last_alert_time = now
        elif status != 'ALERT':
            audio_placeholder.empty()

        # Stat cards — tukar label karena kamera mirror (flip)
        # LEFT_EYE_INDICES di MediaPipe = mata kanan user (setelah flip)
        # RIGHT_EYE_INDICES di MediaPipe = mata kiri user (setelah flip)
        r_color = "#4ade80" if left_state == 'open' else "#f87171"
        l_color = "#4ade80" if right_state == 'open' else "#f87171"
        r_icon = "🟢" if left_state == 'open' else "🔴"
        l_icon = "🟢" if right_state == 'open' else "🔴"

        stats_placeholder.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Mata Kiri</div>
                <div class="stat-value">{l_icon}</div>
                <div class="stat-sub" style="color:{l_color}">{right_state.upper()} · {right_conf:.0%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mata Kanan</div>
                <div class="stat-value">{r_icon}</div>
                <div class="stat-sub" style="color:{r_color}">{left_state.upper()} · {left_conf:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        progress_placeholder.markdown(f"""
        <div class="progress-container">
            <div class="progress-header">
                <span class="progress-label">Drowsiness Level</span>
                <span class="progress-value">{closed_frames} / {CLOSED_FRAME_THRESHOLD}</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill {bar_class}" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.1)

else:
    # Sebelum kamera aktif
    status_placeholder.markdown("""
    <div class="status-card status-inactive">
        <div class="status-icon">📷</div>
        <div class="status-text" style="color:rgba(255,255,255,0.7)">Kamera Belum Aktif</div>
        <div class="status-sub" style="color:rgba(255,255,255,0.4)">Klik START untuk memulai deteksi</div>
    </div>
    """, unsafe_allow_html=True)



