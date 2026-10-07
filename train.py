"""
DASI Training CLI — Train signal models and the TCAFE fusion engine.

Usage:
    # Generate training data first
    python -m src.data.dataset_generator --num-scam 500 --num-legit 500 --output-dir data/synthetic

    # Train individual signal models
    python train.py --signal conversation_semantics --epochs 30
    python train.py --signal temporal_sequence --epochs 30
    python train.py --signal voice_analysis --epochs 30

    # Train TCAFE fusion engine (requires signal models to be trained first)
    python train.py --signal tcafe --epochs 20

    # Train all sequentially
    python train.py --all --epochs 30
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any

import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


SIGNAL_NAMES = [
    "conversation_semantics",
    "temporal_sequence",
    "voice_analysis",
    "network_graph",
    "geo_temporal",
    "tcafe",
]


def load_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load model configuration."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(data_dir: str) -> Dict[str, Any]:
    """Load synthetic training data.
    
    Args:
        data_dir: Path to synthetic data directory.
        
    Returns:
        Dictionary with conversations, audio features, network data, etc.
    """
    data = {}
    
    # Load conversations
    conv_path = os.path.join(data_dir, "conversations.json")
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            data["conversations"] = json.load(f)
        logger.info(f"Loaded {len(data['conversations'])} conversations.")
    else:
        logger.warning(f"No conversations found at {conv_path}")
        data["conversations"] = []
    
    # Load audio features
    audio_path = os.path.join(data_dir, "audio_features.json")
    if os.path.exists(audio_path):
        with open(audio_path, "r", encoding="utf-8") as f:
            data["audio_features"] = json.load(f)
        logger.info(f"Loaded audio features.")
    
    # Load network data
    network_path = os.path.join(data_dir, "network_data.json")
    if os.path.exists(network_path):
        with open(network_path, "r", encoding="utf-8") as f:
            data["network_data"] = json.load(f)
        logger.info(f"Loaded network data.")
    
    # Load geo data
    geo_path = os.path.join(data_dir, "geo_data.json")
    if os.path.exists(geo_path):
        with open(geo_path, "r", encoding="utf-8") as f:
            data["geo_data"] = json.load(f)
        logger.info(f"Loaded geo data.")
    
    # Load metadata
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)
    
    return data


def train_signal(
    signal_name: str,
    config: Dict[str, Any],
    training_data: Dict[str, Any],
    epochs: int,
    batch_size: int,
    checkpoint_dir: str,
    device: str,
):
    """Train a single signal model.
    
    Args:
        signal_name: Name of the signal to train.
        config: Full model configuration.
        training_data: Loaded training data.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to use ('cpu', 'cuda', 'auto').
    """
    logger.info("=" * 60)
    logger.info(f"  Training: {signal_name}")
    logger.info("=" * 60)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    signal_config = config.get(signal_name, {})
    training_config = config.get("training", {})
    
    # Override with CLI args
    training_config["epochs"] = epochs
    training_config["batch_size"] = batch_size
    
    try:
        trainer = Trainer(
            signal_name=signal_name,
            signal_config=signal_config,
            training_config=training_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        
        # Prepare data based on signal type
        if signal_name == "conversation_semantics":
            conversations = training_data.get("conversations", [])
            if not conversations:
                logger.error("No conversation data available for NLP training.")
                return
            trainer.train_nlp(conversations)
        
        elif signal_name == "temporal_sequence":
            conversations = training_data.get("conversations", [])
            if not conversations:
                logger.error("No conversation data available for temporal training.")
                return
            trainer.train_temporal(conversations)
        
        elif signal_name == "voice_analysis":
            audio = training_data.get("audio_features", {})
            if not audio:
                logger.warning("No audio features. Generating synthetic audio data...")
                # Generate placeholder audio data
                n_samples = len(training_data.get("conversations", []))
                if n_samples == 0:
                    n_samples = 100
                mel_specs = torch.randn(n_samples, 1, 128, 128)
                labels = {
                    "deepfake_labels": torch.randint(0, 2, (n_samples,)).float()
                }
                trainer.train_audio(mel_specs, labels)
            else:
                trainer.train_audio_from_features(audio)
        
        elif signal_name == "network_graph":
            network_data = training_data.get("network_data", {})
            if not network_data:
                logger.warning("No network data available. Skipping.")
                return
            trainer.train_network(network_data)
        
        elif signal_name == "geo_temporal":
            geo_data = training_data.get("geo_data", {})
            if not geo_data:
                logger.warning("No geo data available. Skipping.")
                return
            trainer.train_geo(geo_data)
        
        elif signal_name == "tcafe":
            # TCAFE training requires all signals
            conversations = training_data.get("conversations", [])
            if not conversations:
                logger.error("No data available for TCAFE training.")
                return
            trainer.train_tcafe(conversations, training_data)
        
        logger.info(f"Training complete for {signal_name}.")
        
    except Exception as e:
        logger.error(f"Training failed for {signal_name}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train DASI signal models and TCAFE fusion engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --signal conversation_semantics --epochs 30
  python train.py --signal tcafe --epochs 20
  python train.py --all --epochs 30
  python train.py --all --data-dir data/synthetic --checkpoint-dir data/checkpoints
        """,
    )
    
    parser.add_argument(
        "--signal", type=str, choices=SIGNAL_NAMES,
        help="Signal model to train",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all signal models sequentially, then TCAFE",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/synthetic",
        help="Directory containing training data (default: data/synthetic)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="data/checkpoints",
        help="Directory for model checkpoints (default: data/checkpoints)",
    )
    parser.add_argument(
        "--config", type=str, default="config/model_config.yaml",
        help="Model configuration file (default: config/model_config.yaml)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )
    
    args = parser.parse_args()
    
    if not args.signal and not args.all:
        parser.error("Either --signal or --all is required.")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load training data
    logger.info(f"Loading training data from {args.data_dir}...")
    training_data = load_training_data(args.data_dir)
    
    if not training_data.get("conversations"):
        logger.warning(
            "No training data found. Generate data first:\n"
            "  python -m src.data.dataset_generator --num-scam 500 --num-legit 500 "
            f"--output-dir {args.data_dir}"
        )
    
    start_time = time.time()
    
    if args.all:
        # Train all signals, then TCAFE
        for signal in SIGNAL_NAMES:
            train_signal(
                signal_name=signal,
                config=config,
                training_data=training_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                device=args.device,
            )
    else:
        train_signal(
            signal_name=args.signal,
            config=config,
            training_data=training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
