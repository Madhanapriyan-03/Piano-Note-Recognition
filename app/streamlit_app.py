"""
Streamlit Web Application: Piano Note Recognition & Automatic Piano Transcription.
"""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure project root is in sys.path when launched directly via `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="Piano Note Recognition & Transcription",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and dark mode styling
CUSTOM_CSS = """
<style>
    /* Main Background and Fonts */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #0B0F19 100%);
        color: #F3F4F6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Header */
    .hero-container {
        padding: 24px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 16px;
    }

    /* Model Status Badge */
    .model-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Note Sequence Flow */
    .sequence-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #38BDF8;
        border-radius: 12px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 1.15rem;
        color: #38BDF8;
        line-height: 1.6;
        word-wrap: break-word;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Lazy loading of modules
@st.cache_resource
def get_transcriber():
    from src.inference.transcriber import PianoTranscriber
    from src.utils.config import load_config

    cfg = load_config()
    return PianoTranscriber(config=cfg, device="cpu")


def main():
    from src.utils.config import load_config
    from src.visualization.piano_keyboard import render_piano_html
    from src.visualization.spectrogram_view import plot_waveform, plot_interactive_spectrogram

    config = load_config()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎹 Piano Note Recognition & Automatic Transcription</div>
            <div class="hero-subtitle">
                High-Resolution Acoustic Transcription powered by Pretrained CRNN with Onset/Offset Regression & Velocity Modeling
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Settings & Model")
        st.markdown('<div class="model-badge">✅ Pretrained CRNN Model Ready</div>', unsafe_allow_html=True)
        st.markdown("**⚡ Engine:** `PianoTranscription (Kong et al.)`")
        st.markdown("**💻 Device:** `CPU`")
        st.markdown("**🎼 Model Accuracy:** Note F1: `0.9677` | Pedal F1: `0.9186`")

        st.markdown("---")
        st.subheader("🎯 Transcription Thresholds")
        onset_thresh = st.slider(
            "Onset Sensitivity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(getattr(config.inference, "onset_threshold", 0.3)),
            step=0.05,
        )
        frame_thresh = st.slider(
            "Frame Sustain Threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(getattr(config.inference, "frame_threshold", 0.1)),
            step=0.05,
        )
        min_duration = st.slider(
            "Min Note Duration (s)",
            min_value=0.02,
            max_value=0.5,
            value=float(getattr(config.inference, "min_note_duration", 0.05)),
            step=0.01,
        )

        st.markdown("---")
        st.caption("Piano Note Recognition Deep Learning System | MAESTRO v3.0")

    # Main Area: Audio Input
    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.markdown("### 1. Upload or Select Audio")
        input_mode = st.radio("Choose Input Source:", ["Upload Audio File", "Use Sample Dataset Audio"], horizontal=True)

        audio_bytes = None
        audio_path_to_process = None
        temp_audio_file = None

        if input_mode == "Upload Audio File":
            uploaded_file = st.file_uploader("Upload Piano Audio (WAV, MP3, FLAC, OGG):", type=["wav", "mp3", "flac", "ogg"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    audio_path_to_process = tmp.name
                    temp_audio_file = tmp.name
        else:
            samples_dir = PROJECT_ROOT / "data" / "maestro"
            sample_wavs = list(samples_dir.rglob("*.wav")) if samples_dir.exists() else []

            if sample_wavs:
                sample_options = {p.name: p for p in sample_wavs}
                selected_sample = st.selectbox("Select Sample Track:", list(sample_options.keys()))
                audio_path_to_process = str(sample_options[selected_sample])
                with open(audio_path_to_process, "rb") as f:
                    audio_bytes = f.read()
            else:
                st.info("No sample tracks found yet in `data/maestro/`.")

    with col_info:
        st.markdown("### 📊 Audio Preview")
        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.info("Upload or select a piano recording to begin.")

    # Execution Section
    if audio_path_to_process is not None:
        transcriber = get_transcriber()

        st.markdown("---")
        col_btn, col_status = st.columns([1, 2])
        run_transcription = col_btn.button("🚀 Transcribe Piano Notes", type="primary", use_container_width=True)

        # Persistent temp MIDI file in session state
        temp_midi_export = Path(tempfile.gettempdir()) / "piano_transcription_active.mid"

        if run_transcription or "transcription_results" in st.session_state:
            if run_transcription:
                with st.spinner("Processing audio and running Pretrained Deep Learning inference..."):
                    t0 = time.time()
                    results = transcriber.transcribe_audio(
                        audio_path_to_process,
                        onset_threshold=onset_thresh,
                        frame_threshold=frame_thresh,
                        min_note_duration=min_duration,
                        midi_path=temp_midi_export,
                    )
                    inference_time = time.time() - t0
                    results["inference_time"] = inference_time
                    results["midi_file"] = str(temp_midi_export)
                    st.session_state["transcription_results"] = results

            results = st.session_state["transcription_results"]
            notes = results["notes"]
            note_events = results["note_events"]
            sequence_chain = results["sequence_chain"]

            st.markdown("### 2. Audio & Signal Analysis")
            tab_wave, tab_spec = st.tabs(["🌊 Waveform", "🌈 Log-Mel Spectrogram"])

            with tab_wave:
                fig_wave = plot_waveform(results["audio"], sr=transcriber.audio_proc.sample_rate)
                st.pyplot(fig_wave)

            with tab_spec:
                fig_spec = plot_interactive_spectrogram(
                    results["mel_spectrogram"],
                    sr=transcriber.audio_proc.sample_rate,
                    duration=results["duration"],
                )
                st.pyplot(fig_spec)

            st.markdown("### 3. Detected Notes & Transcription")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Notes Detected", len(notes))
            m2.metric("Audio Duration", f"{results['duration']:.2f} s")
            unique_pitches = len(set(n["pitch"] for n in notes)) if notes else 0
            m3.metric("Unique Pitches", unique_pitches)
            m4.metric("Inference Time", f"{results.get('inference_time', 0.0):.2f} s")

            st.markdown("#### 🎼 Detected Note Progression Sequence:")
            st.markdown(f'<div class="sequence-box">{sequence_chain}</div>', unsafe_allow_html=True)

            st.markdown("#### 🎹 Virtual 88-Key Piano Keyboard (Detected Notes Highlighted):")
            active_pitches = [n["pitch"] for n in notes]
            piano_html = render_piano_html(active_pitches=active_pitches)
            st.components.v1.html(piano_html, height=180, scrolling=False)

            st.markdown("#### 📋 Timestamped Note Events Table:")
            if notes:
                import pandas as pd
                df = pd.DataFrame(notes)
                df_display = df[["onset_formatted", "offset_formatted", "note", "pitch", "duration", "velocity"]].copy()
                df_display.columns = ["Onset (MM:SS.ss)", "Offset (MM:SS.ss)", "Note", "MIDI Pitch", "Duration (s)", "Velocity"]
                st.dataframe(df_display, use_container_width=True, height=280)
            else:
                st.warning("No notes detected with current threshold settings.")

            st.markdown("#### 💾 Export Transcriptions:")
            col_d1, col_d2 = st.columns(2)

            json_str = json.dumps(notes, indent=2)
            col_d1.download_button(
                label="📥 Download JSON Annotation",
                data=json_str,
                file_name="transcription.json",
                mime="application/json",
                use_container_width=True,
            )

            midi_data = None
            if temp_midi_export.exists():
                with open(temp_midi_export, "rb") as mf:
                    midi_data = mf.read()
            elif note_events:
                transcriber.export_midi(note_events, temp_midi_export, pedal_events=results.get("pedal_events"))
                if temp_midi_export.exists():
                    with open(temp_midi_export, "rb") as mf:
                        midi_data = mf.read()

            if midi_data is not None:
                col_d2.download_button(
                    label="🎹 Download MIDI (.mid) File",
                    data=midi_data,
                    file_name="transcription.mid",
                    mime="audio/midi",
                    use_container_width=True,
                )

        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except Exception:
                pass


if __name__ == "__main__":
    main()
