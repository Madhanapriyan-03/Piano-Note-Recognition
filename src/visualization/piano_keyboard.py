"""
88-Key Virtual Piano Keyboard visualization (HTML/SVG and Matplotlib).
"""

from typing import List, Optional, Set
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils.midi_utils import midi_to_note_name


def is_black_key(pitch: int) -> bool:
    return (pitch % 12) in [1, 3, 6, 8, 10]


def render_piano_html(active_pitches: Optional[List[int]] = None, width: int = 900, height: int = 140) -> str:
    active_set: Set[int] = set(active_pitches or [])
    min_midi = 21  # A0
    max_midi = 108 # C8

    white_keys = [p for p in range(min_midi, max_midi + 1) if not is_black_key(p)]
    num_white = len(white_keys)
    key_w = width / num_white
    key_h = height
    black_w = key_w * 0.65
    black_h = height * 0.62

    svg_white_keys = []
    svg_black_keys = []
    svg_labels = []

    white_idx = 0
    pitch_to_x = {}

    for p in range(min_midi, max_midi + 1):
        if not is_black_key(p):
            x = white_idx * key_w
            pitch_to_x[p] = x
            is_active = p in active_set
            fill = "#38BDF8" if is_active else "#F8FAFC"
            stroke = "#0F172A"
            glow = 'filter="url(#glow)"' if is_active else ""

            svg_white_keys.append(
                f'<rect x="{x:.1f}" y="0" width="{key_w:.1f}" height="{key_h:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" rx="3" ry="3" {glow} '
                f'class="piano-key" data-pitch="{p}" data-note="{midi_to_note_name(p)}"/>'
            )

            if p % 12 == 0:
                note_lbl = midi_to_note_name(p)
                svg_labels.append(
                    f'<text x="{x + key_w/2:.1f}" y="{key_h - 10:.1f}" font-size="10" '
                    f'font-family="sans-serif" font-weight="bold" fill="#64748B" '
                    f'text-anchor="middle">{note_lbl}</text>'
                )

            white_idx += 1

    for p in range(min_midi, max_midi + 1):
        if is_black_key(p):
            prev_white = p - 1
            if prev_white in pitch_to_x:
                x = pitch_to_x[prev_white] + key_w - (black_w / 2)
            else:
                x = 0
            is_active = p in active_set
            fill = "#EC4899" if is_active else "#1E293B"
            stroke = "#0F172A"
            glow = 'filter="url(#glow-pink)"' if is_active else ""

            svg_black_keys.append(
                f'<rect x="{x:.1f}" y="0" width="{black_w:.1f}" height="{black_h:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="2" ry="2" {glow} '
                f'class="piano-key" data-pitch="{p}" data-note="{midi_to_note_name(p)}"/>'
            )

    html = f"""
    <div style="width: 100%; overflow-x: auto; background: #0F172A; padding: 16px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <svg viewBox="0 0 {width} {height}" style="width: 100%; height: auto; display: block;">
            <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="glow-pink" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <!-- White Keys -->
            {''.join(svg_white_keys)}
            <!-- Labels -->
            {''.join(svg_labels)}
            <!-- Black Keys -->
            {''.join(svg_black_keys)}
        </svg>
    </div>
    """
    return html


def render_piano_matplotlib(active_pitches: Optional[List[int]] = None) -> plt.Figure:
    active_set = set(active_pitches or [])
    min_midi = 21
    max_midi = 108

    white_keys = [p for p in range(min_midi, max_midi + 1) if not is_black_key(p)]
    fig, ax = plt.subplots(figsize=(14, 2.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    white_idx = 0
    pitch_to_x = {}

    for p in range(min_midi, max_midi + 1):
        if not is_black_key(p):
            x = white_idx
            pitch_to_x[p] = x
            color = "#38BDF8" if p in active_set else "#F8FAFC"
            rect = plt.Rectangle((x, 0), 1.0, 1.0, facecolor=color, edgecolor="#0F172A", linewidth=1.5)
            ax.add_patch(rect)
            if p % 12 == 0:
                ax.text(x + 0.5, 0.1, midi_to_note_name(p), ha="center", va="center", color="#64748B", fontsize=8, fontweight="bold")
            white_idx += 1

    for p in range(min_midi, max_midi + 1):
        if is_black_key(p):
            prev_white = p - 1
            x = pitch_to_x[prev_white] + 0.65
            color = "#EC4899" if p in active_set else "#1E293B"
            rect = plt.Rectangle((x, 0.38), 0.7, 0.62, facecolor=color, edgecolor="#0F172A", linewidth=1.0, zorder=3)
            ax.add_patch(rect)

    ax.set_xlim(0, len(white_keys))
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    plt.tight_layout()
    return fig
