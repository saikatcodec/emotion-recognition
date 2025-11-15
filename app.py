import sys
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.utils.process_frame import process_video

sys.path.append(str(Path(__file__).parent))

st.title("Face Emotion Recognition")
st.subheader("Recognize the face emotion to video", divider="gray")

real_time = st.checkbox("Real-time process(Camera)")
if real_time:
    st.write("Real-time process")
    webrtc_streamer(
        key="emotion-detect-camera-access",
        sendback_audio=False,
        video_frame_callback=process_video,
    )
else:
    st.text_input("Video path", placeholder="path/to/video.mp4")
