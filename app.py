# Buat ulang app.py dengan UI yang lebih menarik dan mendukung dua metode input:
# 1. Kamera real-time via WebRTC
# 2. Upload gambar langsung untuk deteksi mata mengantuk

enhanced_app_py = '''
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model dan classifier
model = load_model("models/cnncat2.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

st.set_page_config(page_title="Drowsiness Detection App", layout="wide")
st.title("😴 Drowsiness Detection App")
st.markdown("Pilih metode input untuk mendeteksi kantuk: Kamera atau Gambar Upload")

tabs = st.tabs(["📷 Kamera Real-Time", "🖼️ Upload Gambar"])

# ========================
# KAMERA REAL-TIME
# ========================
with tabs[0]:
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.score = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye = roi_color[ey:ey + eh, ex:ex + ew]
                    eye = cv2.resize(eye, (80, 80))
                    eye = eye / 255.0
                    eye = np.expand_dims(eye, axis=0)
                    prediction = model.predict(eye)
                    if prediction[0][0] > 0.5:
                        cv2.putText(img, "Mengantuk", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Tidak Mengantuk", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break
                break
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="drowsy-stream",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": []})
    )

# ========================
# UPLOAD GAMBAR
# ========================
with tabs[1]:
    uploaded_file = st.file_uploader("Upload gambar wajah untuk deteksi kantuk:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((80, 80))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Gambar yang Diunggah", use_column_width=True)
        with col2:
            if prediction[0][0] > 0.5:
                st.error("❌ Deteksi: Mengantuk")
            else:
                st.success("✅ Deteksi: Tidak Mengantuk")
'''
