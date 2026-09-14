"""
Transcription evaluation metrics: Frame-level and Note-level Precision, Recall, F1.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from src.preprocessing.label_processor import LabelProcessor
from src.utils.midi_utils import NoteEvent


def evaluate_transcription(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    onset_tolerance: float = 0.05,
    offset_ratio: Optional[float] = 0.20,
    offset_min_tolerance: float = 0.05,
) -> Dict[str, float]:
    if not ref_notes and not est_notes:
        return {
            "onset_precision": 1.0, "onset_recall": 1.0, "onset_f1": 1.0,
            "ref_notes_count": 0, "est_notes_count": 0, "matched_onsets": 0,
        }
    if not ref_notes:
        return {
            "onset_precision": 0.0, "onset_recall": 1.0, "onset_f1": 0.0,
            "ref_notes_count": 0, "est_notes_count": len(est_notes), "matched_onsets": 0,
        }
    if not est_notes:
        return {
            "onset_precision": 1.0, "onset_recall": 0.0, "onset_f1": 0.0,
            "ref_notes_count": len(ref_notes), "est_notes_count": 0, "matched_onsets": 0,
        }

    matched_ref = set()
    matched_est = set()

    for e_idx, est in enumerate(est_notes):
        best_ref_idx = None
        min_onset_diff = float("inf")

        for r_idx, ref in enumerate(ref_notes):
            if r_idx in matched_ref:
                continue

            if est.pitch != ref.pitch:
                continue

            diff = abs(est.onset - ref.onset)
            if diff <= onset_tolerance and diff < min_onset_diff:
                min_onset_diff = diff
                best_ref_idx = r_idx

        if best_ref_idx is not None:
            matched_ref.add(best_ref_idx)
            matched_est.add(e_idx)

    tp_onset = len(matched_ref)
    fp_onset = len(est_notes) - tp_onset
    fn_onset = len(ref_notes) - tp_onset

    onset_prec = tp_onset / (tp_onset + fp_onset) if (tp_onset + fp_onset) > 0 else 0.0
    onset_rec = tp_onset / (tp_onset + fn_onset) if (tp_onset + fn_onset) > 0 else 0.0
    onset_f1 = (2 * onset_prec * onset_rec / (onset_prec + onset_rec)) if (onset_prec + onset_rec) > 0 else 0.0

    return {
        "onset_precision": round(float(onset_prec), 4),
        "onset_recall": round(float(onset_rec), 4),
        "onset_f1": round(float(onset_f1), 4),
        "ref_notes_count": len(ref_notes),
        "est_notes_count": len(est_notes),
        "matched_onsets": tp_onset,
    }


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_processor: LabelProcessor,
    frame_rate: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.4,
    onset_tolerance: float = 0.05,
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    total_tp_frame = 0
    total_fp_frame = 0
    total_fn_frame = 0

    all_note_metrics = []

    with torch.no_grad():
        for mels, frames_target, onsets_target, metadata_batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            mels = mels.to(device)
            outputs = model(mels)

            frame_probs = outputs["frame_probs"].cpu().numpy()
            onset_probs = outputs["onset_probs"].cpu().numpy()
            frames_gt = frames_target.cpu().numpy()

            b_size = mels.size(0)
            for b in range(b_size):
                f_pred = (frame_probs[b] >= frame_threshold).astype(int)
                f_true = (frames_gt[b] >= 0.5).astype(int)

                total_tp_frame += int(np.sum((f_pred == 1) & (f_true == 1)))
                total_fp_frame += int(np.sum((f_pred == 1) & (f_true == 0)))
                total_fn_frame += int(np.sum((f_pred == 0) & (f_true == 1)))

                est_notes = label_processor.predictions_to_notes(
                    frame_probs=frame_probs[b],
                    onset_probs=onset_probs[b],
                    frame_rate=frame_rate,
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                )
                ref_notes = metadata_batch[b].get("notes", [])

                metrics = evaluate_transcription(
                    ref_notes=ref_notes,
                    est_notes=est_notes,
                    onset_tolerance=onset_tolerance,
                )
                all_note_metrics.append(metrics)

    f_prec = total_tp_frame / (total_tp_frame + total_fp_frame) if (total_tp_frame + total_fp_frame) > 0 else 0.0
    f_rec = total_tp_frame / (total_tp_frame + total_fn_frame) if (total_tp_frame + total_fn_frame) > 0 else 0.0
    f_f1 = (2 * f_prec * f_rec / (f_prec + f_rec)) if (f_prec + f_rec) > 0 else 0.0

    avg_onset_prec = float(np.mean([m["onset_precision"] for m in all_note_metrics])) if all_note_metrics else 0.0
    avg_onset_rec = float(np.mean([m["onset_recall"] for m in all_note_metrics])) if all_note_metrics else 0.0
    avg_onset_f1 = float(np.mean([m["onset_f1"] for m in all_note_metrics])) if all_note_metrics else 0.0

    return {
        "frame_precision": round(float(f_prec), 4),
        "frame_recall": round(float(f_rec), 4),
        "frame_f1": round(float(f_f1), 4),
        "note_onset_precision": round(avg_onset_prec, 4),
        "note_onset_recall": round(avg_onset_rec, 4),
        "note_onset_f1": round(avg_onset_f1, 4),
        "evaluated_segments": len(all_note_metrics),
    }
