import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Detektor Jerawat Real-Time",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- JUDUL UTAMA ---
st.title("🔬 Aplikasi Deteksi Jerawat Real-Time")
st.warning("**Disclaimer:** Ini adalah proyek edukasi dan bukan alat diagnosis medis. Hasilnya mungkin tidak 100% akurat dan tidak dapat menggantikan konsultasi profesional dengan dokter kulit.", icon="⚠️")

# --- SIDEBAR PENGATURAN ---
with st.sidebar:
    st.header("Pengaturan")
    confidence_threshold = st.slider(
        "Tingkat Kepercayaan (Confidence Threshold)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25,
        step=0.05
    )
    

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

try:
    model = load_yolo_model('weights/best.pt')
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'best.pt' ada di folder 'weights'. Error: {e}")
    st.stop()


# --- KELAS UNTUK TRANSFORMASI VIDEO WEBRTC ---
class YOLOAcneTransformer(VideoTransformerBase):
    def __init__(self, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Mengubah frame video menjadi format yang bisa dibaca OpenCV
        img = frame.to_ndarray(format="bgr24")

        # Menjalankan inferensi YOLO
        results = self.model.predict(img, conf=self.confidence_threshold)
        
        # Menggambar kotak deteksi pada frame
        annotated_frame = results[0].plot()

        # Mengembalikan frame yang sudah diolah
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# --- TAMPILAN TAB ---
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