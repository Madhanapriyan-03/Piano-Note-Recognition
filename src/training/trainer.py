"""
Deep Learning Training loop with multi-task loss, validation, checkpointing, and metric tracking.
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.crnn_transcriber import CRNNTranscriber, build_model
from src.utils.config import get_device


class TranscriptionTrainer:
    """
    Handles model training, validation, multi-task loss calculation, and checkpoint management.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_onset: nn.Module,
        criterion_frame: nn.Module,
        device: torch.device,
        epochs: int = 20,
        onset_loss_weight: float = 1.0,
        frame_loss_weight: float = 1.0,
        clip_grad_norm: float = 3.0,
        early_stopping_patience: int = 7,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        model_save_dir: Union[str, Path] = "models",
        best_model_name: str = "best_model.pt",
        latest_checkpoint_name: str = "checkpoint_latest.pt",
        logs_dir: Union[str, Path] = "outputs/logs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion_onset = criterion_onset
        self.criterion_frame = criterion_frame
        self.device = device
        self.epochs = epochs
        self.onset_loss_weight = onset_loss_weight
        self.frame_loss_weight = frame_loss_weight
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler = lr_scheduler

        self.model_save_dir = Path(model_save_dir).resolve()
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_save_dir / best_model_name
        self.latest_checkpoint_path = self.model_save_dir / latest_checkpoint_name

        self.logs_dir = Path(logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_onset_loss": [],
            "train_frame_loss": [],
            "val_loss": [],
            "val_onset_loss": [],
            "val_frame_loss": [],
            "val_f1_frame": [],
            "val_f1_onset": [],
            "learning_rate": [],
        }

    def train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_onset_loss = 0.0
        total_frame_loss = 0.0
        batch_count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx + 1}/{self.epochs} [Train]",
            leave=False,
            ncols=100,
        )

        for mels, frames_target, onsets_target, _ in pbar:
            mels = mels.to(self.device)
            frames_target = frames_target.to(self.device)
            onsets_target = onsets_target.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(mels)
            onset_logits = outputs["onset_logits"]
            frame_logits = outputs["frame_logits"]

            loss_onset = self.criterion_onset(onset_logits, onsets_target)
            loss_frame = self.criterion_frame(frame_logits, frames_target)
            loss = self.onset_loss_weight * loss_onset + self.frame_loss_weight * loss_frame

            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_onset_loss += loss_onset.item()
            total_frame_loss += loss_frame.item()
            batch_count += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, batch_count)
        avg_onset = total_onset_loss / max(1, batch_count)
        avg_frame = total_frame_loss / max(1, batch_count)
        return avg_loss, avg_onset, avg_frame

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_onset_loss = 0.0
        total_frame_loss = 0.0
        batch_count = 0

        all_frame_preds = []
        all_frame_targets = []
        all_onset_preds = []
        all_onset_targets = []

        for mels, frames_target, onsets_target, _ in self.val_loader:
            mels = mels.to(self.device)
            frames_target = frames_target.to(self.device)
            onsets_target = onsets_target.to(self.device)

            outputs = self.model(mels)
            onset_logits = outputs["onset_logits"]
            frame_logits = outputs["frame_logits"]
            onset_probs = outputs["onset_probs"]
            frame_probs = outputs["frame_probs"]

            loss_onset = self.criterion_onset(onset_logits, onsets_target)
            loss_frame = self.criterion_frame(frame_logits, frames_target)
            loss = self.onset_loss_weight * loss_onset + self.frame_loss_weight * loss_frame

            total_loss += loss.item()
            total_onset_loss += loss_onset.item()
            total_frame_loss += loss_frame.item()
            batch_count += 1

            all_frame_preds.append((frame_probs >= 0.5).cpu().numpy().reshape(-1))
            all_frame_targets.append((frames_target >= 0.5).cpu().numpy().reshape(-1))
            all_onset_preds.append((onset_probs >= 0.5).cpu().numpy().reshape(-1))
            all_onset_targets.append((onsets_target >= 0.5).cpu().numpy().reshape(-1))

        avg_loss = total_loss / max(1, batch_count)
        avg_onset = total_onset_loss / max(1, batch_count)
        avg_frame = total_frame_loss / max(1, batch_count)

        f1_frame = self._compute_f1(
            np.concatenate(all_frame_preds) if all_frame_preds else np.array([]),
            np.concatenate(all_frame_targets) if all_frame_targets else np.array([]),
        )
        f1_onset = self._compute_f1(
            np.concatenate(all_onset_preds) if all_onset_preds else np.array([]),
            np.concatenate(all_onset_targets) if all_onset_targets else np.array([]),
        )

        return avg_loss, avg_onset, avg_frame, f1_frame, f1_onset

    def _compute_f1(self, preds: np.ndarray, targets: np.ndarray) -> float:
        if len(preds) == 0:
            return 0.0
        tp = np.sum((preds == 1) & (targets == 1))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return float(f1)

    def train(self) -> Dict[str, List[float]]:
        best_val_metric = -1.0
        patience_counter = 0
        start_time = time.time()

        print(f"\n=======================================================")
        print(f" Starting CRNN Piano Transcription Model Training")
        print(f" Device: {self.device} | Epochs: {self.epochs} | Train batches: {len(self.train_loader)}")
        print(f"=======================================================\n")

        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss, train_onset, train_frame = self.train_epoch(epoch)
            val_loss, val_onset, val_frame, val_f1_frame, val_f1_onset = self.validate()
            epoch_time = time.time() - t0

            curr_lr = self.optimizer.param_groups[0]["lr"]
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_onset_loss"].append(train_onset)
            self.history["train_frame_loss"].append(train_frame)
            self.history["val_loss"].append(val_loss)
            self.history["val_onset_loss"].append(val_onset)
            self.history["val_frame_loss"].append(val_frame)
            self.history["val_f1_frame"].append(val_f1_frame)
            self.history["val_f1_onset"].append(val_f1_onset)
            self.history["learning_rate"].append(curr_lr)

            composite_metric = (val_f1_frame + val_f1_onset) / 2.0
            if composite_metric == 0.0:
                composite_metric = -val_loss

            is_best = composite_metric > best_val_metric
            if is_best:
                best_val_metric = composite_metric
                patience_counter = 0
                self._save_checkpoint(self.best_model_path, epoch, is_best=True)
                best_indicator = " * [BEST]"
            else:
                patience_counter += 1
                best_indicator = ""

            self._save_checkpoint(self.latest_checkpoint_path, epoch, is_best=False)

            print(
                f"Epoch {epoch + 1:02d}/{self.epochs:02d} [{epoch_time:.1f}s] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val F1 (Frame): {val_f1_frame:.4f} | Val F1 (Onset): {val_f1_onset:.4f}{best_indicator}"
            )

            if patience_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement in validation score for {self.early_stopping_patience} epochs. Stopping.")
                break

        total_time = time.time() - start_time
        print(f"\n[Training Complete] Total Duration: {total_time:.2f}s | Best Model Saved: {self.best_model_path}")

        history_file = self.logs_dir / "training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "is_best": is_best,
        }
        torch.save(checkpoint, str(path))


def train_model(config, device: Optional[torch.device] = None) -> Tuple[CRNNTranscriber, dict]:
    if device is None:
        device = get_device(config.training.device, verbose=True)

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    from src.dataset.maestro_dataset import create_dataloaders
    train_loader, val_loader, _ = create_dataloaders(config)

    model = build_model(config)

    pos_weight = torch.tensor([5.0], device=device)
    criterion_onset = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_frame = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.training.epochs),
        eta_min=float(config.training.min_lr),
    )

    trainer = TranscriptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion_onset=criterion_onset,
        criterion_frame=criterion_frame,
        device=device,
        epochs=int(config.training.epochs),
        onset_loss_weight=float(config.training.onset_loss_weight),
        frame_loss_weight=float(config.training.frame_loss_weight),
        clip_grad_norm=float(config.training.clip_grad_norm),
        early_stopping_patience=int(config.training.early_stopping_patience),
        lr_scheduler=scheduler,
        model_save_dir=config.paths.model_dir,
        best_model_name=config.paths.best_model_name,
        latest_checkpoint_name=config.paths.latest_checkpoint_name,
        logs_dir=config.paths.logs_dir,
    )

    history = trainer.train()
    return model, history
