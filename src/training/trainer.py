"""
Training Engine — Signal-Level and TCAFE End-to-End Training.

Provides a unified Trainer class that handles:
  - Individual signal model training (NLP, Temporal, Voice, Network, Geo)
  - TCAFE fusion engine end-to-end training
  - Checkpoint management, early stopping, metric logging
  - Train/validation/test split management
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from src.training.losses import (
    WeightedBCELoss,
    PhaseTransitionLoss,
    TripletCampaignLoss,
    MultiSignalFusionLoss,
)


# ════════════════════════════════════════════════════════
# PyTorch Datasets
# ════════════════════════════════════════════════════════

class NLPDataset(Dataset):
    """Dataset for training the ConversationSemanticsAnalyzer."""
    
    def __init__(self, conversations: List[Dict], labels: Dict):
        self.texts = []
        self.nlp_labels = []
        self.scam_labels = []
        
        for i, conv in enumerate(conversations):
            for msg in conv["messages"]:
                if msg["role"] in ("scammer", "caller"):
                    self.texts.append(msg["text"])
                    scores = msg["nlp_labels"]
                    self.nlp_labels.append([
                        scores.get("authority", 0),
                        scores.get("urgency", 0),
                        scores.get("isolation", 0),
                        scores.get("coercion", 0),
                        scores.get("legal", 0),
                    ])
                    self.scam_labels.append(1.0 if conv["is_scam"] else 0.0)
        
        self.nlp_labels = torch.tensor(self.nlp_labels, dtype=torch.float32)
        self.scam_labels = torch.tensor(self.scam_labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "nlp_labels": self.nlp_labels[idx],
            "scam_label": self.scam_labels[idx],
        }


class AudioDataset(Dataset):
    """Dataset for training the VoiceAnalysisAnalyzer."""
    
    def __init__(self, mel_specs: torch.Tensor, labels: Dict):
        self.mel_specs = mel_specs
        self.deepfake_labels = labels["deepfake_labels"]
    
    def __len__(self):
        return len(self.mel_specs)
    
    def __getitem__(self, idx):
        return {
            "mel_spectrogram": self.mel_specs[idx],
            "deepfake_label": self.deepfake_labels[idx],
        }


class FusionDataset(Dataset):
    """Dataset for end-to-end TCAFE training.
    
    Each sample bundles pre-computed signal embeddings with fusion labels.
    """
    
    def __init__(
        self,
        signal_embeddings: Dict[str, torch.Tensor],
        signal_confidences: Dict[str, torch.Tensor],
        threat_scores: torch.Tensor,
        scam_labels: torch.Tensor,
        campaign_ids: torch.Tensor,
    ):
        self.signal_embeddings = signal_embeddings
        self.signal_confidences = signal_confidences
        self.threat_scores = threat_scores
        self.scam_labels = scam_labels
        self.campaign_ids = campaign_ids
    
    def __len__(self):
        return len(self.threat_scores)
    
    def __getitem__(self, idx):
        embeddings = {k: v[idx] for k, v in self.signal_embeddings.items()}
        confidences = {k: v[idx].item() for k, v in self.signal_confidences.items()}
        return {
            "signal_embeddings": embeddings,
            "signal_confidences": confidences,
            "threat_score": self.threat_scores[idx],
            "scam_label": self.scam_labels[idx],
            "campaign_id": self.campaign_ids[idx],
        }


# ════════════════════════════════════════════════════════
# Training History
# ════════════════════════════════════════════════════════

@dataclass
class TrainingHistory:
    """Tracks training metrics across epochs."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: List[Dict[str, float]] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }


# ════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════

class Trainer:
    """Unified training engine for all DASI models.
    
    Usage:
        trainer = Trainer(
            model=my_model,
            loss_fn=my_loss,
            config=training_config,
            checkpoint_dir="checkpoints/nlp"
        )
        history = trainer.train(train_loader, val_loader, epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        lr = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 0.01)
        optimizer_name = config.get("optimizer", "adamw").lower()
        
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        scheduler_name = config.get("scheduler", "cosine").lower()
        epochs = config.get("epochs", 50)
        warmup_steps = config.get("warmup_steps", 500)
        
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Gradient clipping
        self.gradient_clip = config.get("gradient_clip", 1.0)
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        
        # History
        self.history = TrainingHistory()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        forward_fn: Optional[callable] = None,
    ) -> TrainingHistory:
        """Run the full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (overrides config).
            forward_fn: Custom forward function that takes (model, batch, device)
                       and returns (loss, metrics_dict). If None, uses default.
        
        Returns:
            TrainingHistory with all metrics.
        """
        epochs = epochs or self.config.get("epochs", 50)
        
        print(f"\n{'='*60}")
        print(f"  DASI Training Engine")
        print(f"{'='*60}")
        print(f"  Model:     {self.model.__class__.__name__}")
        print(f"  Device:    {self.device}")
        print(f"  Epochs:    {epochs}")
        print(f"  LR:        {self.optimizer.param_groups[0]['lr']}")
        print(f"  Params:    {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(
                train_loader, forward_fn,
            )
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(
                val_loader, forward_fn,
            )
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Record history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_metrics.append(train_metrics)
            self.history.val_metrics.append(val_metrics)
            
            elapsed = time.time() - epoch_start
            
            # Check for improvement
            improved = ""
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self.history.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
                improved = " ★ BEST"
            else:
                self.history.epochs_without_improvement += 1
            
            # Print progress
            lr = self.optimizer.param_groups[0]["lr"]
            acc_str = ""
            if "accuracy" in val_metrics:
                acc_str = f"  acc={val_metrics['accuracy']:.4f}"
            
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f}{acc_str} | "
                f"lr={lr:.2e} | "
                f"{elapsed:.1f}s{improved}"
            )
            
            # Early stopping
            if self.history.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {self.early_stopping_patience} epochs)")
                break
        
        # Save final model and history
        self._save_checkpoint(epoch, val_loss, is_best=False)
        self._save_history()
        
        print(f"\n  Training complete. Best epoch: {self.history.best_epoch} (val_loss={self.history.best_val_loss:.4f})")
        print(f"  Checkpoints saved to {self.checkpoint_dir}/")
        
        return self.history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        forward_fn: Optional[callable],
    ) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        
        for batch in loader:
            self.optimizer.zero_grad()
            
            if forward_fn is not None:
                loss, metrics = forward_fn(self.model, batch, self.device)
            else:
                loss, metrics = self._default_forward(batch)
            
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def _validate_epoch(
        self,
        loader: DataLoader,
        forward_fn: Optional[callable],
    ) -> Tuple[float, Dict[str, float]]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        num_batches = 0
        
        for batch in loader:
            if forward_fn is not None:
                loss, metrics = forward_fn(self.model, batch, self.device)
            else:
                loss, metrics = self._default_forward(batch)
            
            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0) + v
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in all_metrics.items()}
        return avg_loss, avg_metrics
    
    def _default_forward(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Default forward pass for simple classification models."""
        # Move data to device
        inputs = batch.get("input", batch.get("mel_spectrogram"))
        labels = batch.get("label", batch.get("scam_label", batch.get("deepfake_label")))
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)
        
        outputs = self.model(inputs)
        
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("predictions"))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = self.loss_fn(logits.squeeze(), labels)
        
        # Compute accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            accuracy = (preds == labels).float().mean().item()
        
        return loss, {"accuracy": accuracy}
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
    
    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
        return checkpoint["epoch"]
