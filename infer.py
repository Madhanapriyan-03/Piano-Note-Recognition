"""
CLI Entry point for transcribing an arbitrary audio file using Pretrained CRNN.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.transcriber import PianoTranscriber
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Piano Audio to Timestamped Notes using Pretrained CRNN"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to input audio file (WAV, MP3, FLAC, OGG)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional custom path to pretrained .pth checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for inference (default: 'cpu')"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--out-midi", type=str, default=None, help="Optional output path to export transcribed MIDI (.mid)"
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="Optional output path to save JSON notes"
    )
    parser.add_argument(
        "--onset-thresh", type=float, default=None, help="Onset sensitivity threshold"
    )
    parser.add_argument(
        "--frame-thresh", type=float, default=None, help="Frame sustain threshold"
    )
    parser.add_argument(
        "--min-duration", type=float, default=None, help="Minimum note duration in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    config = load_config(args.config) if args.config else None
    transcriber = PianoTranscriber(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )

    print(f"\n[Transcribing] Processing audio: {audio_path.name}...")
    results = transcriber.transcribe_audio(
        audio_path,
        onset_threshold=args.onset_thresh,
        frame_threshold=args.frame_thresh,
        min_note_duration=args.min_duration,
        midi_path=args.out_midi,
    )

    notes = results["notes"]
    seq_chain = results["sequence_chain"]

    print("\n=======================================================")
    print(f" DETECTED PIANO NOTES ({len(notes)} notes detected in {results['duration']:.2f}s)")
    print("=======================================================")
    print(f"{'ONSET':<12} {'OFFSET':<12} {'NOTE':<8} {'PITCH':<8} {'VELOCITY':<8}")
    print("-" * 55)

    for n in notes:
        vel = n.get("velocity", 100)
        print(
            f"{n['onset_formatted']:<12} {n['offset_formatted']:<12} "
            f"{n['note']:<8} {n['pitch']:<8} {vel:<8}"
        )

    print("\n-------------------------------------------------------")
    print("Detected Note Sequence:")
    print(f"  {seq_chain}")
    print("=======================================================\n")

    if args.out_midi:
        out_mid = Path(args.out_midi).resolve()
        if not out_mid.exists():
            transcriber.export_midi(results["note_events"], out_mid, pedal_events=results.get("pedal_events"))
        print(f"[Export] Saved MIDI file to: {out_mid}")

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        print(f"[Export] Saved JSON annotations to: {out_json}")


if __name__ == "__main__":
    main()
