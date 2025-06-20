import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import pandas as pd
import plotly.express as px
import os

"""Page Configuration"""
# Set page configuration for a wide dashboard layout
st.set_page_config(page_title="Carpark Management Dashboard", layout="wide")

# Title and description
st.title("Carpark Management Dashboard")
st.markdown("This dashboard demonstrates real-time object detection using YOLOv11 on CCTV streams.")


"""Model Loading"""
# Cache YOLO model to avoid reloading
@st.cache_resource
def load_model():
    model_path = "yolo11m.pt" #to be replaced with trained weights
    if not os.path.exists(model_path):
        st.warning("Downloading YOLO model...")
        st.error(f"Model {model_path} not found. Please upload to the repository. ")        
    return YOLO(model_path)

model = load_model()

"""Tracks Detection History"""
# Initialize session state for detection history
if "detections" not in st.session_state:
    st.session_state.detections = []

# Create tabs for dashboard organization
tab1, tab2, tab3 = st.tabs(["Live Stream", "Image Upload", "Analytics"])


"""Dashboard Structure"""
# Tab 1: Live CCTV Stream
with tab1:
    st.header("Live CCTV Stream")
    st.markdown("Enter your CCTV's RTSP URL to view real-time car park space detections.")
    rtsp_url = st.text_input("RTSP URL", "rtsp://your_camera_url")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        conf_threshold = st.session_state.get("conf_threshold", 0.85)
        results = model.predict(img, conf=conf_threshold)
        img = results[0].plot()  # Draw bounding boxes
        
        # Store detection data
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = model.names[int(box.cls)]
                st.session_state.detections.append({
                    "class": cls,
                    "confidence": float(box.conf),
                    "timestamp": pd.Timestamp.now()
                })
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    if rtsp_url and rtsp_url != "rtsp://your_camera_url":
        try:
            webrtc_streamer(
                key="yolo-cctv",
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False}
            )
        except Exception as e:
            st.error(f"Error accessing RTSP stream: {e}. Try a valid RTSP URL or use the Image Upload tab.")
    else:
        st.info("Please enter a valid RTSP URL (e.g., rtsp://username:password@camera_ip/stream).")

#gonna change this for video 
# Tab 2: Image Upload
with tab2:
    st.header("Upload Image for Detection")
    st.markdown("Upload an image to run YOLO detection.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        conf_threshold = st.session_state.get("conf_threshold", 0.5)
        results = model.predict(img, conf=conf_threshold)
        st.image(results[0].plot(), channels="BGR", caption="Detection Results")
        
        # Store detection data
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = model.names[int(box.cls)]
                st.session_state.detections.append({
                    "class": cls,
                    "confidence": float(box.conf),
                    "timestamp": pd.Timestamp.now()
                })

# Tab 3: Analytics
with tab3:
    st.header("Detection Analytics")
    st.markdown("View detection statistics and adjust the confidence threshold.")
    st.slider("Confidence Threshold", 0.1, 1.0, 0.5, key="conf_threshold")
    
    if st.session_state.detections:
        # Create DataFrame for detections
        df = pd.DataFrame(st.session_state.detections)
        
        # Display detection table
        st.subheader("Detection Summary")
        st.dataframe(df)
        
        # Bar chart of detection counts by class
        st.subheader("Detection Counts by Class")
        class_counts = df["class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        fig = px.bar(class_counts, x="Class", y="Count", title="Objects Detected by Class")
        st.plotly_chart(fig)
        
        # Histogram of confidence scores
        st.subheader("Confidence Score Distribution")
        fig = px.histogram(df, x="confidence", title="Confidence Score Distribution")
        st.plotly_chart(fig)
    else:
        st.info("No detections yet. Run a detection in the Live Stream or Image Upload tab.")