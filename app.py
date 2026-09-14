"""
Root Streamlit Entrypoint for Piano Note Recognition and Automatic Piano Transcription.
Delegates directly to app/streamlit_app.py.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_app import main

if __name__ == "__main__" or "streamlit" in sys.modules:
    main()
