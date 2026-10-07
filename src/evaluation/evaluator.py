"""
Evaluator — Comprehensive evaluation metrics for the DASI system.

Computes:
  - Classification metrics: Precision, Recall, F1, Accuracy
  - Ranking metrics: ROC-AUC, PR-AUC
  - Early detection metrics: Phase-aware detection rate, time-to-detection
  - Campaign metrics: Campaign matching accuracy, fingerprint clustering quality
  - Signal contribution analysis: Per-signal attribution breakdown
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class DASIEvaluator:
    """Evaluates the DASI system across multiple dimensions.
    
    Collects predictions and ground truth labels during evaluation,
    then computes comprehensive metrics.
    
    Usage:
        evaluator = DASIEvaluator()
        for sample in test_set:
            result = orchestrator.process(sample)
            evaluator.record(
                prediction=result.threat_assessment.threat_score,
                ground_truth=sample["is_scam"],
                phase=sample.get("scam_phase"),
                campaign_pred=result.campaign_id,
                campaign_true=sample.get("campaign_label"),
                signal_contributions=result.threat_assessment.signal_contributions,
            )
        report = evaluator.compute_metrics()
    """
    
    def __init__(self, name: str = "DASI Evaluation"):
        self.name = name
        self.reset()
    
    def reset(self):
        """Clear all collected data."""
        self.predictions: List[float] = []
        self.ground_truths: List[int] = []
        self.threat_levels: List[str] = []
        self.phases: List[Optional[int]] = []
        self.campaign_preds: List[int] = []
        self.campaign_trues: List[Optional[str]] = []
        self.signal_contributions_list: List[Dict[str, float]] = []
        self.processing_times: List[float] = []
        self.escalation_speeds: List[float] = []
    
    def record(
        self,
        prediction: float,
        ground_truth: int,
        threat_level: str = "",
        phase: Optional[int] = None,
        campaign_pred: int = -1,
        campaign_true: Optional[str] = None,
        signal_contributions: Optional[Dict[str, float]] = None,
        processing_time_ms: float = 0.0,
        escalation_speed: float = 0.0,
    ):
        """Record a single prediction for evaluation."""
        self.predictions.append(float(prediction))
        self.ground_truths.append(int(ground_truth))
        self.threat_levels.append(threat_level)
        self.phases.append(phase)
        self.campaign_preds.append(campaign_pred)
        self.campaign_trues.append(campaign_true)
        self.signal_contributions_list.append(signal_contributions or {})
        self.processing_times.append(processing_time_ms)
        self.escalation_speeds.append(escalation_speed)
    
    def compute_metrics(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Compute all evaluation metrics.
        
        Args:
            threshold: Classification threshold for binary prediction.
            
        Returns:
            Dictionary with all computed metrics.
        """
        if not self.predictions:
            return {"error": "No predictions recorded."}
        
        preds = np.array(self.predictions)
        truths = np.array(self.ground_truths)
        binary_preds = (preds >= threshold).astype(int)
        
        report = {
            "name": self.name,
            "num_samples": len(preds),
            "threshold": threshold,
            "timestamp": time.time(),
        }
        
        # 1. Classification metrics
        report["classification"] = self._compute_classification_metrics(
            binary_preds, truths
        )
        
        # 2. Ranking metrics (ROC-AUC, PR-AUC)
        report["ranking"] = self._compute_ranking_metrics(preds, truths)
        
        # 3. Early detection metrics
        report["early_detection"] = self._compute_early_detection_metrics(
            preds, truths, threshold
        )
        
        # 4. Campaign metrics
        report["campaign"] = self._compute_campaign_metrics()
        
        # 5. Signal contribution analysis
        report["signal_analysis"] = self._compute_signal_analysis()
        
        # 6. Performance metrics
        report["performance"] = self._compute_performance_metrics()
        
        # 7. Threshold analysis
        report["threshold_analysis"] = self._compute_threshold_analysis(preds, truths)
        
        return report
    
    def _compute_classification_metrics(
        self, preds: np.ndarray, truths: np.ndarray
    ) -> Dict:
        """Standard classification metrics."""
        tp = int(np.sum((preds == 1) & (truths == 1)))
        tn = int(np.sum((preds == 0) & (truths == 0)))
        fp = int(np.sum((preds == 1) & (truths == 0)))
        fn = int(np.sum((preds == 0) & (truths == 1)))
        
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        specificity = tn / max(1, tn + fp)
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
        }
    
    def _compute_ranking_metrics(
        self, preds: np.ndarray, truths: np.ndarray
    ) -> Dict:
        """ROC-AUC and PR-AUC computed manually (no sklearn dependency)."""
        # Sort by prediction score descending
        sorted_indices = np.argsort(-preds)
        sorted_truths = truths[sorted_indices]
        sorted_preds = preds[sorted_indices]
        
        # ROC curve
        n_pos = int(np.sum(truths == 1))
        n_neg = int(np.sum(truths == 0))
        
        if n_pos == 0 or n_neg == 0:
            return {"roc_auc": 0.0, "pr_auc": 0.0, "note": "Single class only."}
        
        # Compute ROC-AUC using trapezoid rule
        tpr_list = [0.0]
        fpr_list = [0.0]
        tp_count = 0
        fp_count = 0
        
        for t in sorted_truths:
            if t == 1:
                tp_count += 1
            else:
                fp_count += 1
            tpr_list.append(tp_count / n_pos)
            fpr_list.append(fp_count / n_neg)
        
        # Trapezoid integration for AUC
        roc_auc = 0.0
        for i in range(1, len(fpr_list)):
            roc_auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        
        # PR-AUC
        precision_list = []
        recall_list = []
        tp_count = 0
        
        for i, t in enumerate(sorted_truths, 1):
            if t == 1:
                tp_count += 1
            precision_list.append(tp_count / i)
            recall_list.append(tp_count / n_pos)
        
        # Compute PR-AUC
        pr_auc = 0.0
        for i in range(1, len(recall_list)):
            pr_auc += (recall_list[i] - recall_list[i-1]) * precision_list[i]
        
        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
        }
    
    def _compute_early_detection_metrics(
        self, preds: np.ndarray, truths: np.ndarray, threshold: float
    ) -> Dict:
        """Metrics specific to early scam detection.
        
        Measures how early in the scam progression the system can detect threats.
        """
        phase_detection = defaultdict(lambda: {"total": 0, "detected": 0})
        
        for i, phase in enumerate(self.phases):
            if phase is not None and truths[i] == 1:
                phase_detection[phase]["total"] += 1
                if preds[i] >= threshold:
                    phase_detection[phase]["detected"] += 1
        
        phase_rates = {}
        for phase in sorted(phase_detection.keys()):
            data = phase_detection[phase]
            rate = data["detected"] / max(1, data["total"])
            phase_rates[f"phase_{phase}"] = {
                "detection_rate": round(rate, 4),
                "total": data["total"],
                "detected": data["detected"],
            }
        
        # Early detection rate — detection in phases 1-3 (before financial loss)
        early_phases = [1, 2, 3]
        early_total = sum(
            phase_detection[p]["total"] for p in early_phases if p in phase_detection
        )
        early_detected = sum(
            phase_detection[p]["detected"] for p in early_phases if p in phase_detection
        )
        early_rate = early_detected / max(1, early_total)
        
        return {
            "early_detection_rate": round(early_rate, 4),
            "early_total": early_total,
            "early_detected": early_detected,
            "per_phase": phase_rates,
        }
    
    def _compute_campaign_metrics(self) -> Dict:
        """Campaign clustering and matching metrics."""
        # Filter samples with campaign labels
        labeled = [
            (p, t) for p, t in zip(self.campaign_preds, self.campaign_trues)
            if t is not None
        ]
        
        if not labeled:
            return {"note": "No campaign labels available."}
        
        # Count unique predicted and true campaigns
        unique_pred = len(set(p for p, _ in labeled))
        unique_true = len(set(t for _, t in labeled))
        
        # Purity score: what fraction of predicted clusters are pure?
        cluster_contents = defaultdict(list)
        for pred_id, true_label in labeled:
            cluster_contents[pred_id].append(true_label)
        
        purity = 0.0
        total_in_clusters = 0
        for cid, labels in cluster_contents.items():
            if labels:
                most_common = max(set(labels), key=labels.count)
                purity += labels.count(most_common)
                total_in_clusters += len(labels)
        
        purity_score = purity / max(1, total_in_clusters)
        
        return {
            "purity_score": round(purity_score, 4),
            "predicted_campaigns": unique_pred,
            "true_campaigns": unique_true,
            "labeled_samples": len(labeled),
        }
    
    def _compute_signal_analysis(self) -> Dict:
        """Analyze per-signal contribution patterns."""
        if not self.signal_contributions_list:
            return {}
        
        # Aggregate signal contributions
        signal_stats = defaultdict(list)
        
        for contributions in self.signal_contributions_list:
            for signal, weight in contributions.items():
                signal_stats[signal].append(weight)
        
        analysis = {}
        for signal, weights in signal_stats.items():
            w = np.array(weights)
            analysis[signal] = {
                "mean_contribution": round(float(np.mean(w)), 4),
                "std_contribution": round(float(np.std(w)), 4),
                "min_contribution": round(float(np.min(w)), 4),
                "max_contribution": round(float(np.max(w)), 4),
            }
        
        return analysis
    
    def _compute_performance_metrics(self) -> Dict:
        """Processing time statistics."""
        if not self.processing_times:
            return {}
        
        times = np.array(self.processing_times)
        return {
            "avg_processing_time_ms": round(float(np.mean(times)), 2),
            "p50_processing_time_ms": round(float(np.percentile(times, 50)), 2),
            "p95_processing_time_ms": round(float(np.percentile(times, 95)), 2),
            "p99_processing_time_ms": round(float(np.percentile(times, 99)), 2),
            "max_processing_time_ms": round(float(np.max(times)), 2),
        }
    
    def _compute_threshold_analysis(
        self, preds: np.ndarray, truths: np.ndarray
    ) -> Dict:
        """Analyze performance at different thresholds."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = {}
        
        for t in thresholds:
            binary = (preds >= t).astype(int)
            metrics = self._compute_classification_metrics(binary, truths)
            results[f"threshold_{t}"] = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        
        return results
    
    def generate_report(self, threshold: float = 0.5) -> str:
        """Generate a human-readable evaluation report.
        
        Returns:
            Formatted string report.
        """
        metrics = self.compute_metrics(threshold)
        
        lines = [
            "=" * 60,
            f"  DASI EVALUATION REPORT — {self.name}",
            "=" * 60,
            "",
            f"  Samples: {metrics['num_samples']}",
            f"  Threshold: {metrics['threshold']}",
            "",
            "  ── Classification Metrics ──",
        ]
        
        cls = metrics.get("classification", {})
        lines.extend([
            f"    Accuracy:    {cls.get('accuracy', 0):.4f}",
            f"    Precision:   {cls.get('precision', 0):.4f}",
            f"    Recall:      {cls.get('recall', 0):.4f}",
            f"    F1 Score:    {cls.get('f1_score', 0):.4f}",
            f"    Specificity: {cls.get('specificity', 0):.4f}",
            "",
        ])
        
        cm = cls.get("confusion_matrix", {})
        lines.extend([
            "    Confusion Matrix:",
            f"      TP={cm.get('true_positive', 0)}  FP={cm.get('false_positive', 0)}",
            f"      FN={cm.get('false_negative', 0)}  TN={cm.get('true_negative', 0)}",
            "",
        ])
        
        rank = metrics.get("ranking", {})
        lines.extend([
            "  ── Ranking Metrics ──",
            f"    ROC-AUC:  {rank.get('roc_auc', 0):.4f}",
            f"    PR-AUC:   {rank.get('pr_auc', 0):.4f}",
            "",
        ])
        
        early = metrics.get("early_detection", {})
        lines.extend([
            "  ── Early Detection ──",
            f"    Early Detection Rate (Phase 1-3): {early.get('early_detection_rate', 0):.4f}",
            f"    Early Detected: {early.get('early_detected', 0)}/{early.get('early_total', 0)}",
            "",
        ])
        
        campaign = metrics.get("campaign", {})
        lines.extend([
            "  ── Campaign Intelligence ──",
            f"    Purity Score: {campaign.get('purity_score', 'N/A')}",
            f"    Predicted Campaigns: {campaign.get('predicted_campaigns', 'N/A')}",
            f"    True Campaigns: {campaign.get('true_campaigns', 'N/A')}",
            "",
        ])
        
        perf = metrics.get("performance", {})
        if perf:
            lines.extend([
                "  ── Performance ──",
                f"    Avg Processing Time: {perf.get('avg_processing_time_ms', 0):.2f} ms",
                f"    P95 Processing Time: {perf.get('p95_processing_time_ms', 0):.2f} ms",
                "",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: str, threshold: float = 0.5):
        """Save metrics to a JSON file."""
        metrics = self.compute_metrics(threshold)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
