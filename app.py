
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import requests 


def download_model_if_needed(model_url, model_path):
    """Mengecek jika model ada, jika tidak, unduh dari URL."""
    if not os.path.exists(model_path):
        st.info("Model tidak ditemukan. Mengunduh model... (mungkin perlu beberapa saat)")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            with st.spinner("Mengunduh..."):
                r = requests.get(model_url, stream=True)
                r.raise_for_status() 
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Model berhasil diunduh!")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            st.stop()


st.set_page_config(
    page_title="Detektor Jerawat Real-Time",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_URL = "https://github.com/glendod/acne-detection-project/releases/download/v1.0/best.pt" 
MODEL_PATH = "weights/best.pt"


download_model_if_needed(MODEL_URL, MODEL_PATH)


st.title("🔬 Aplikasi Deteksi Jerawat Real-Time")
st.warning("**Disclaimer:** Ini adalah proyek edukasi dan bukan alat diagnosis medis. Hasilnya mungkin tidak 100% akurat dan tidak dapat menggantikan konsultasi profesional dengan dokter kulit.", icon="⚠️")

with st.sidebar:
    st.header("Pengaturan")
    confidence_threshold = st.slider(
        "Tingkat Kepercayaan (Confidence Threshold)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25,
        step=0.05
    )


@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

try:
    model = load_yolo_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

class YOLOAcneTransformer(VideoTransformerBase):
    def __init__(self, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.confidence_threshold)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

tab1, tab2 = st.tabs(["🖼️ Deteksi dari Gambar", "📹 Deteksi Real-Time (Webcam)"])
with tab1:
    st.header("Unggah Gambar untuk Dianalisis")
    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        results = model.predict(image_np, conf=confidence_threshold)
        annotated_image_np = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB)
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Hasil Deteksi Model", use_column_width=True)
        detection_count = len(results[0].boxes)
        if detection_count > 0:
            st.success(f"**Terdeteksi {detection_count} objek jerawat.**")
        else:
            st.info("**Tidak ada jerawat yang terdeteksi** dengan tingkat kepercayaan saat ini.")
with tab2:
    st.header("Analisis Langsung dari Webcam")
    st.write("Klik 'START' untuk memulai. Pastikan Anda memberikan izin akses kamera pada browser Anda.")
    webrtc_streamer(
        key="acne-detection-stream",
        video_transformer_factory=lambda: YOLOAcneTransformer(confidence_threshold=confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )