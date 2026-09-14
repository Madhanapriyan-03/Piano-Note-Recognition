# 🎹 Piano Note Recognition & Automatic Piano Transcription

A high-resolution acoustic piano transcription and note recognition web application and deep learning pipeline. The system transcribes polyphonic piano solo audio into timestamped musical notes, interactive visual piano roll representations, an interactive 88-key virtual piano keyboard, and downloadable Standard MIDI (`.mid`) and JSON files.

---

## 🚀 Live Streamlit Cloud Deployment

The application is fully configured for deployment on **Streamlit Community Cloud** with automatic model weight resolution and system-level audio library decoders.

### Deploying to Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository (`Piano-Note-Recognition`), branch (`main`).
5. Set **Main file path** to:
   ```text
   app/streamlit_app.py
   ```
   *(or `app.py`)*
6. Click **Deploy!**
7. The app will automatically install dependencies from `requirements.txt`, system decoders from `packages.txt`, and download/cache the high-accuracy pretrained CRNN transcription model (~165 MB) on first launch.

---

## 💻 Local Setup & Execution

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/<your-username>/Piano-Note-Recognition.git
cd Piano-Note-Recognition

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
Launch the interactive Streamlit UI using either entrypoint:
```bash
streamlit run app/streamlit_app.py
```
*or:*
```bash
streamlit run app.py
```

### 4. Run CLI Inference
Transcribe any local audio file directly via command line:
```bash
python infer.py --audio path/to/piano_audio.wav --out-midi transcription.mid --out-json transcription.json
```

### 5. Run Test Suite
```bash
python -m pytest
```

---

## 🏗️ Architecture & Features

- **Pretrained CRNN Neural Engine**: High-resolution time-frequency convolutional recurrent neural network with onset/offset regression and velocity modeling.
- **Dynamic Model Cache Manager**: Automatically verifies, downloads, and caches pretrained weights safely without hardcoded paths.
- **Signal Analysis**: Audio waveform visualization and Log-Mel Spectrogram computation.
- **Interactive 88-Key Piano Keyboard**: Real-time visual feedback highlighting active pitches from the transcribed audio.
- **Multi-Format Export**: One-click download of generated Standard MIDI (`.mid`) and JSON annotations.

---

## 📂 Project Structure

```text
Piano-Note-Recognition/
├── .streamlit/
│   └── config.toml             # Streamlit server and dark theme settings
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Streamlit web application
├── app.py                      # Root Streamlit entrypoint
├── configs/
│   └── config.yaml             # System configuration
├── packages.txt                # Linux system dependencies (libsndfile1, ffmpeg)
├── requirements.txt            # Python dependencies
├── infer.py                    # Standalone CLI inference script
├── train.py                    # Training pipeline entrypoint
├── evaluate.py                 # Evaluation benchmark script
├── src/
│   ├── dataset/                # Dataset loaders & MAESTRO preparation
│   ├── evaluation/             # Metrics and evaluation plotter
│   ├── inference/              # Transcriber engine & checkpoint manager
│   ├── models/                 # CRNN architecture definitions
│   ├── preprocessing/          # Audio & label processors
│   ├── training/               # Training loop & scheduler
│   ├── utils/                  # Audio I/O, MIDI export, and config loaders
│   └── visualization/          # Piano keyboard & spectrogram rendering
└── tests/                      # Unit and integration test suite
```
