import streamlit as st
import whisper
import pandas as pd
import os
import subprocess

st.set_page_config(page_title="AI Audio Search Engine", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #121212;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        color: #00e6e6;
        text-shadow: 3px 3px 10px #00e6e6;
        padding: 20px 0;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00e6e6;
        text-shadow: 2px 2px 8px #00e6e6;
        padding: 10px 0;
    }
    .upload-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #00e6e6;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 255, 255, 0.2);
        color: #ffffff;
    }
    .upload-box label {
        color: #ffcc00;
        font-weight: bold;
    }
    .search-bar input {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 2px solid #00e6e6;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00e6e6, #0073e6);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 20px;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff007f, #ff4d00);
        color: white;
        transform: scale(1.05);
    }
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #00e6e6, #0073e6);
        border-radius: 5px;
    }
    .uploaded-file {
        font-size: 1rem;
        color: #ffcc00;
        font-weight: bold;
        padding-top: 10px;
    }
    .search-input {
        color: #ffcc00 !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="main-title">🎤 AI-Powered Audio Search Engine</h1>', unsafe_allow_html=True)

# Centered Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Upload Audio File with Color Modification
    st.markdown("""
        <div class="upload-box">
            <label style="color: #ffcc00; font-size: 1.1rem; font-weight: bold;">🔊 Upload an audio file</label>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file
    audio_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.markdown(f'<p class="uploaded-file">{uploaded_file.name}  {round(uploaded_file.size / 1024, 1)}KB</p>', unsafe_allow_html=True)

    # Show progress
    st.markdown('<div class="progress-bar" style="width: 100%"></div>', unsafe_allow_html=True)

    model = whisper.load_model("base")

    # Transcribe Audio
    result = model.transcribe(audio_path, temperature=0.0)
    segments = result.get("segments", [])

    if segments:
        df = pd.DataFrame(segments)
        df.to_csv("transcription.csv", index=False)

    # Display Transcription
    st.markdown('<h3 class="section-title">📜 Transcribed Text:</h3>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: #00e6e6; 
                    font-size: 1.2rem;
                    font-weight: bold;">
            {result['text']}
        </div>
    """, unsafe_allow_html=True)

    # Search Bar for Keyword Search
    st.markdown('<h3 class="section-title">🔍 Search in Audio:</h3>', unsafe_allow_html=True)
    keyword = st.text_input("Enter a keyword to search...", key="search", help="Find specific words in the transcription")

    if st.button("Search Keyword"):
        if keyword:
            df = pd.read_csv("transcription.csv")
            matches = df[df['text'].str.contains(keyword, case=False, na=False)]

            if not matches.empty:
                st.success(f"✅ Found '{keyword}' at timestamps:")
                st.dataframe(matches[["start", "end", "text"]])

                # Play from First Occurrence
                start_time = matches.iloc[0]["start"]
                st.info(f"Jumping to {start_time} seconds in audio...")

                cmd = ["ffplay", "-ss", str(start_time), "-i", audio_path, "-autoexit"]
                try:
                    subprocess.run(cmd, check=True)
                except FileNotFoundError:
                    st.error("FFmpeg is not installed. Please install FFmpeg to play audio from timestamps.")
            else:
                st.warning("❌ Keyword not found in audio.")
        else:
            st.warning("⚠️ Please enter a keyword.")
