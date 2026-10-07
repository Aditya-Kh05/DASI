"""
Master Dataset Generator — Orchestrates all sub-generators.

Creates complete, multi-modal scam scenario datasets where each sample
has aligned text, audio features, network graph, and geo-temporal data,
ready for training the 5 signal analyzers and the TCAFE fusion engine.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.text_generator import TextGenerator, GeneratedConversation
from src.data.audio_generator import AudioFeatureGenerator, GeneratedAudioFeatures
from src.data.network_generator import NetworkGraphGenerator, GeneratedGraph
from src.data.geo_generator import GeoTemporalGenerator, GeneratedGeoDataset


@dataclass
class ScenarioSample:
    """A complete multi-modal training sample.
    
    Combines aligned data from all 5 modalities for one scam/legitimate scenario.
    """
    sample_id: str
    is_scam: bool
    
    # Signal 1: Text
    conversation: GeneratedConversation
    
    # Signal 2: Temporal (derived from conversation phases)
    phase_sequence: List[int]
    phase_labels: List[str]
    
    # Signal 3: Audio
    audio_features: GeneratedAudioFeatures
    
    # Signal 4: Network
    network_graph: GeneratedGraph
    
    # Signal 5: Geo-temporal (index into shared geo dataset)
    geo_event_indices: List[int]
    
    # Fusion labels
    threat_score_label: float        # Ground truth [0, 1]
    threat_level_label: str          # MINIMAL/LOW/MEDIUM/HIGH/CRITICAL
    campaign_id_label: Optional[int]
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "is_scam": self.is_scam,
            "num_messages": len(self.conversation.messages),
            "phases": self.phase_sequence,
            "audio_is_deepfake": self.audio_features.is_deepfake,
            "network_is_syndicate": self.network_graph.is_syndicate,
            "threat_score_label": self.threat_score_label,
            "threat_level_label": self.threat_level_label,
            "campaign_id_label": self.campaign_id_label,
        }


class DatasetGenerator:
    """Master generator that creates aligned multi-modal datasets.
    
    Usage:
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate(
            num_scam=2500,
            num_legitimate=2500,
            output_dir="data/synthetic"
        )
    """
    
    THREAT_LEVELS = [
        (0.9, "CRITICAL"),
        (0.7, "HIGH"),
        (0.5, "MEDIUM"),
        (0.3, "LOW"),
        (0.0, "MINIMAL"),
    ]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.text_gen = TextGenerator(seed=seed)
        self.audio_gen = AudioFeatureGenerator(seed=seed + 1)
        self.network_gen = NetworkGraphGenerator(seed=seed + 2)
        self.geo_gen = GeoTemporalGenerator(seed=seed + 3)
        self._sample_counter = 0
    
    def _compute_threat_score(self, sample_data: Dict) -> float:
        """Compute ground-truth threat score from multi-modal signals.
        
        The threat score is a weighted combination of:
          - Text phase progression (how far into the scam)
          - Deepfake probability
          - Network syndicate topology
          - Geographic clustering
        """
        if not sample_data["is_scam"]:
            # Legitimate: very low threat with some noise
            return max(0.0, min(0.15, np.random.normal(0.05, 0.03)))
        
        # Scam threat increases with phase progression
        max_phase = max(sample_data["phases"]) if sample_data["phases"] else 0
        phase_score = min(1.0, max_phase / 5.0)  # Normalize to [0, 1]
        
        # Deepfake adds to threat
        deepfake_score = 0.8 if sample_data["is_deepfake"] else 0.3
        
        # Syndicate network adds to threat
        network_score = 0.7 if sample_data["is_syndicate"] else 0.2
        
        # Weighted combination
        threat = (
            0.40 * phase_score +
            0.20 * deepfake_score +
            0.25 * network_score +
            0.15 * np.random.uniform(0.3, 0.8)  # Campaign/geo contribution
        )
        
        # Add noise
        threat += np.random.normal(0, 0.03)
        return max(0.1, min(1.0, threat))
    
    def _get_threat_level(self, score: float) -> str:
        """Map threat score to categorical level."""
        for threshold, level in self.THREAT_LEVELS:
            if score >= threshold:
                return level
        return "MINIMAL"
    
    def _generate_scam_sample(self, campaign_id: int) -> ScenarioSample:
        """Generate one complete scam scenario sample."""
        self._sample_counter += 1
        sample_id = f"sample_{self._sample_counter:05d}"
        
        # Generate aligned modalities
        conversation = self.text_gen.generate_scam_conversation()
        
        # Extract phase sequence from conversation
        phase_sequence = [m.phase for m in conversation.messages if m.role == "scammer"]
        phase_labels = [m.phase_name for m in conversation.messages if m.role == "scammer"]
        
        # Generate audio (higher deepfake probability for scam calls)
        is_deepfake = np.random.random() < 0.35
        audio = self.audio_gen.generate_sample(
            is_deepfake=is_deepfake,
            environment=np.random.choice(["CALL_CENTER", "VOIP", "QUIET_ROOM"], p=[0.4, 0.4, 0.2]),
        )
        
        # Generate network (syndicate topology)
        topology = np.random.choice(["star", "chain", "cluster"], p=[0.5, 0.2, 0.3])
        network = self.network_gen.generate_sample(is_syndicate=True, topology=topology)
        
        # Compute ground-truth labels
        sample_data = {
            "is_scam": True,
            "phases": phase_sequence,
            "is_deepfake": is_deepfake,
            "is_syndicate": True,
        }
        threat_score = self._compute_threat_score(sample_data)
        threat_level = self._get_threat_level(threat_score)
        
        return ScenarioSample(
            sample_id=sample_id,
            is_scam=True,
            conversation=conversation,
            phase_sequence=phase_sequence,
            phase_labels=phase_labels,
            audio_features=audio,
            network_graph=network,
            geo_event_indices=[],  # Will be linked after geo dataset generation
            threat_score_label=threat_score,
            threat_level_label=threat_level,
            campaign_id_label=campaign_id,
        )
    
    def _generate_legitimate_sample(self) -> ScenarioSample:
        """Generate one complete legitimate scenario sample."""
        self._sample_counter += 1
        sample_id = f"sample_{self._sample_counter:05d}"
        
        conversation = self.text_gen.generate_legitimate_conversation()
        
        phase_sequence = [0] * len(conversation.messages)
        phase_labels = ["LEGITIMATE"] * len(conversation.messages)
        
        audio = self.audio_gen.generate_sample(
            is_deepfake=False,
            environment=np.random.choice(["QUIET_ROOM", "STREET", "CAR"], p=[0.5, 0.3, 0.2]),
        )
        
        network = self.network_gen.generate_sample(is_syndicate=False)
        
        sample_data = {
            "is_scam": False,
            "phases": phase_sequence,
            "is_deepfake": False,
            "is_syndicate": False,
        }
        threat_score = self._compute_threat_score(sample_data)
        threat_level = self._get_threat_level(threat_score)
        
        return ScenarioSample(
            sample_id=sample_id,
            is_scam=False,
            conversation=conversation,
            phase_sequence=phase_sequence,
            phase_labels=phase_labels,
            audio_features=audio,
            network_graph=network,
            geo_event_indices=[],
            threat_score_label=threat_score,
            threat_level_label=threat_level,
            campaign_id_label=None,
        )
    
    def generate(
        self,
        num_scam: int = 2500,
        num_legitimate: int = 2500,
        num_campaigns: int = 50,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete multi-modal dataset.
        
        Args:
            num_scam: Number of scam scenarios.
            num_legitimate: Number of legitimate scenarios.
            num_campaigns: Number of distinct scam campaigns.
            output_dir: If provided, save dataset to this directory.
        
        Returns:
            Dict with "samples", "geo_dataset", and "statistics".
        """
        print(f"\n{'='*60}")
        print(f"  DASI Synthetic Dataset Generator")
        print(f"{'='*60}")
        print(f"  Scam samples:       {num_scam}")
        print(f"  Legitimate samples: {num_legitimate}")
        print(f"  Campaigns:          {num_campaigns}")
        print()
        
        start_time = time.time()
        samples = []
        
        # Generate scam samples distributed across campaigns
        scams_per_campaign = max(1, num_scam // num_campaigns)
        print(f"  Generating {num_scam} scam samples across {num_campaigns} campaigns...")
        for campaign_id in range(num_campaigns):
            n = scams_per_campaign
            if campaign_id < num_scam % num_campaigns:
                n += 1
            for _ in range(n):
                sample = self._generate_scam_sample(campaign_id)
                samples.append(sample)
        
        # Generate legitimate samples
        print(f"  Generating {num_legitimate} legitimate samples...")
        for _ in range(num_legitimate):
            sample = self._generate_legitimate_sample()
            samples.append(sample)
        
        # Generate geo-temporal dataset
        print(f"  Generating geo-temporal dataset...")
        geo_dataset = self.geo_gen.generate_dataset(
            num_campaigns=num_campaigns,
            events_per_campaign=max(5, num_scam // num_campaigns),
            num_legitimate=num_legitimate,
        )
        
        # Shuffle samples
        np.random.shuffle(samples)
        
        elapsed = time.time() - start_time
        
        # Statistics
        stats = {
            "total_samples": len(samples),
            "num_scam": sum(1 for s in samples if s.is_scam),
            "num_legitimate": sum(1 for s in samples if not s.is_scam),
            "num_campaigns": num_campaigns,
            "num_deepfake_audio": sum(1 for s in samples if s.audio_features.is_deepfake),
            "num_syndicate_graphs": sum(1 for s in samples if s.network_graph.is_syndicate),
            "geo_events": len(geo_dataset.events),
            "generation_time_seconds": round(elapsed, 2),
            "avg_threat_score_scam": round(np.mean([s.threat_score_label for s in samples if s.is_scam]), 4),
            "avg_threat_score_legit": round(np.mean([s.threat_score_label for s in samples if not s.is_scam]), 4),
        }
        
        print(f"\n  Generation complete in {elapsed:.1f}s")
        print(f"  Total samples:     {stats['total_samples']}")
        print(f"  Deepfake audio:    {stats['num_deepfake_audio']}")
        print(f"  Syndicate graphs:  {stats['num_syndicate_graphs']}")
        print(f"  Avg threat (scam): {stats['avg_threat_score_scam']}")
        print(f"  Avg threat (legit):{stats['avg_threat_score_legit']}")
        
        result = {
            "samples": samples,
            "geo_dataset": geo_dataset,
            "statistics": stats,
        }
        
        # Save if output_dir provided
        if output_dir:
            self._save_dataset(result, output_dir)
        
        return result
    
    def _save_dataset(self, result: Dict, output_dir: str):
        """Save dataset to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(result["statistics"], f, indent=2)
        
        # Save sample metadata (without tensors)
        metadata_path = os.path.join(output_dir, "samples_metadata.json")
        metadata = [s.to_dict() for s in result["samples"]]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save text conversations
        conversations_path = os.path.join(output_dir, "conversations.json")
        conversations = [s.conversation.to_dict() for s in result["samples"]]
        with open(conversations_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        # Save tensors as PyTorch files
        tensors_dir = os.path.join(output_dir, "tensors")
        os.makedirs(tensors_dir, exist_ok=True)
        
        # Batch save audio features
        mel_specs = torch.stack([s.audio_features.mel_spectrogram for s in result["samples"]])
        torch.save(mel_specs, os.path.join(tensors_dir, "mel_spectrograms.pt"))
        
        mfccs = torch.stack([s.audio_features.mfcc for s in result["samples"]])
        torch.save(mfccs, os.path.join(tensors_dir, "mfccs.pt"))
        
        # Save labels
        labels = {
            "is_scam": torch.tensor([s.is_scam for s in result["samples"]], dtype=torch.float32),
            "threat_scores": torch.tensor([s.threat_score_label for s in result["samples"]], dtype=torch.float32),
            "deepfake_labels": torch.tensor(
                [s.audio_features.is_deepfake for s in result["samples"]], dtype=torch.float32
            ),
            "phase_sequences": [s.phase_sequence for s in result["samples"]],
        }
        torch.save(labels, os.path.join(tensors_dir, "labels.pt"))
        
        # Save geo dataset
        geo_path = os.path.join(output_dir, "geo_dataset.json")
        with open(geo_path, "w", encoding="utf-8") as f:
            json.dump(result["geo_dataset"].to_dict(), f, indent=2)
        
        geo_tensors = result["geo_dataset"].get_feature_tensors()
        torch.save(geo_tensors, os.path.join(tensors_dir, "geo_tensors.pt"))
        
        # Save network graphs individually (variable sizes)
        graphs_dir = os.path.join(tensors_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        for i, sample in enumerate(result["samples"]):
            graph_data = {
                "node_features": sample.network_graph.node_features,
                "edge_index": sample.network_graph.edge_index,
                "edge_type": sample.network_graph.edge_type,
                "node_labels": sample.network_graph.node_labels,
            }
            torch.save(graph_data, os.path.join(graphs_dir, f"graph_{i:05d}.pt"))
        
        print(f"\n  Dataset saved to {output_dir}/")
        print(f"    - statistics.json")
        print(f"    - samples_metadata.json")
        print(f"    - conversations.json")
        print(f"    - tensors/ (mel_spectrograms.pt, mfccs.pt, labels.pt, geo_tensors.pt)")
        print(f"    - tensors/graphs/ ({len(result['samples'])} graph files)")


# ════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DASI Synthetic Dataset Generator")
    parser.add_argument("--num-scam", type=int, default=2500, help="Number of scam samples")
    parser.add_argument("--num-legit", type=int, default=2500, help="Number of legitimate samples")
    parser.add_argument("--num-campaigns", type=int, default=50, help="Number of scam campaigns")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(seed=args.seed)
    generator.generate(
        num_scam=args.num_scam,
        num_legitimate=args.num_legit,
        num_campaigns=args.num_campaigns,
        output_dir=args.output_dir,
    )
