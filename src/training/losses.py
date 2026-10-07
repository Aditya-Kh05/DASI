"""
Custom Loss Functions for DASI Training.

Includes:
  - WeightedBCELoss: Handles class imbalance between scam/legitimate
  - PhaseTransitionLoss: Penalizes incorrect phase ordering
  - TripletCampaignLoss: Brings same-campaign samples closer in embedding space
  - MultiSignalFusionLoss: Combined loss for TCAFE end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with class weights for imbalanced scam detection.
    
    Since scam events are rarer than legitimate ones in production,
    we weight scam (positive) samples higher to improve recall.
    """
    
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch,) predicted probabilities [0, 1]
            targets: (batch,) ground truth labels {0, 1}
        """
        pos_weight_tensor = torch.tensor([self.pos_weight], device=predictions.device)
        loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=pos_weight_tensor,
        )
        return loss


class PhaseTransitionLoss(nn.Module):
    """Penalizes non-monotonic phase predictions.
    
    Digital arrest scams follow a canonical order:
    Authority → Accusation → Fear → Isolation → Extraction
    
    This loss penalizes the model if it predicts Phase 4 before Phase 2,
    encouraging it to learn the temporal ordering constraint.
    """
    
    def __init__(self, penalty_weight: float = 0.5):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(
        self,
        phase_predictions: torch.Tensor,
        phase_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phase_predictions: (batch, seq_len, num_phases) phase logits
            phase_targets: (batch, seq_len) ground truth phase indices
        """
        # Standard cross-entropy for phase classification
        batch_size, seq_len, num_phases = phase_predictions.shape
        ce_loss = F.cross_entropy(
            phase_predictions.reshape(-1, num_phases),
            phase_targets.reshape(-1),
        )
        
        # Transition penalty: predicted phases should be monotonically non-decreasing
        predicted_phases = phase_predictions.argmax(dim=-1).float()  # (batch, seq_len)
        
        if seq_len > 1:
            # Difference between consecutive predictions
            diffs = predicted_phases[:, 1:] - predicted_phases[:, :-1]
            # Penalize negative transitions (going backwards in phase)
            backward_penalty = F.relu(-diffs).mean()
        else:
            backward_penalty = torch.tensor(0.0, device=phase_predictions.device)
        
        return ce_loss + self.penalty_weight * backward_penalty


class TripletCampaignLoss(nn.Module):
    """Triplet loss for campaign fingerprint learning.
    
    Pulls embeddings of same-campaign samples together and pushes
    different-campaign samples apart. This is the core learning signal
    for the CampaignFingerprintGenerator.
    
    Triplet: (anchor, positive, negative)
      - anchor & positive: same campaign
      - anchor & negative: different campaigns
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        campaign_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) campaign fingerprint embeddings
            campaign_ids: (batch,) campaign ID labels (-1 for no campaign)
        """
        batch_size = embeddings.shape[0]
        
        if batch_size < 3:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Find valid triplets
        anchors, positives, negatives = [], [], []
        
        valid_mask = campaign_ids >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        for i in range(min(len(valid_indices), 32)):  # Limit to 32 triplets per batch
            anchor_idx = valid_indices[i % len(valid_indices)]
            anchor_campaign = campaign_ids[anchor_idx]
            
            # Find a positive (same campaign)
            same_campaign = (campaign_ids == anchor_campaign) & (torch.arange(batch_size, device=embeddings.device) != anchor_idx)
            if same_campaign.any():
                pos_candidates = same_campaign.nonzero(as_tuple=True)[0]
                pos_idx = pos_candidates[torch.randint(len(pos_candidates), (1,))]
            else:
                continue
            
            # Find a negative (different campaign)
            diff_campaign = (campaign_ids != anchor_campaign) & valid_mask
            if diff_campaign.any():
                neg_candidates = diff_campaign.nonzero(as_tuple=True)[0]
                neg_idx = neg_candidates[torch.randint(len(neg_candidates), (1,))]
            else:
                continue
            
            anchors.append(embeddings[anchor_idx])
            positives.append(embeddings[pos_idx.squeeze()])
            negatives.append(embeddings[neg_idx.squeeze()])
        
        if not anchors:
            return torch.tensor(0.0, device=embeddings.device)
        
        anchor_stack = torch.stack(anchors)
        positive_stack = torch.stack(positives)
        negative_stack = torch.stack(negatives)
        
        return self.triplet_loss(anchor_stack, positive_stack, negative_stack)


class MultiSignalFusionLoss(nn.Module):
    """Combined loss for end-to-end TCAFE training.
    
    Components:
      1. Threat classification loss (BCE)
      2. Threat score regression loss (MSE)
      3. Campaign fingerprint triplet loss
      4. Early detection bonus (lower loss when high-threat detected early)
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.5,
        campaign_weight: float = 0.3,
        early_detection_weight: float = 0.2,
        pos_weight: float = 2.0,
        triplet_margin: float = 1.0,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.campaign_weight = campaign_weight
        self.early_detection_weight = early_detection_weight
        
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = TripletCampaignLoss(margin=triplet_margin)
    
    def forward(
        self,
        pred_threat_logits: torch.Tensor,
        pred_threat_scores: torch.Tensor,
        pred_fingerprints: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
        target_campaign_ids: torch.Tensor,
        detection_phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with "total", "classification", "regression",
            "campaign", "early_detection" losses.
        """
        # 1. Classification: is this a scam?
        cls_loss = self.bce_loss(pred_threat_logits, target_labels)
        
        # 2. Regression: how severe is the threat?
        reg_loss = self.mse_loss(pred_threat_scores, target_scores)
        
        # 3. Campaign: cluster same-campaign samples together
        camp_loss = self.triplet_loss(pred_fingerprints, target_campaign_ids)
        
        # 4. Early detection bonus
        if detection_phase is not None:
            # Reward detecting threats at earlier phases
            # detection_phase: (batch,) — phase at which the system triggered alert
            # Lower phase = earlier detection = lower loss
            scam_mask = target_labels > 0.5
            if scam_mask.any():
                early_bonus = (detection_phase[scam_mask] / 5.0).mean()
            else:
                early_bonus = torch.tensor(0.0, device=pred_threat_logits.device)
        else:
            early_bonus = torch.tensor(0.0, device=pred_threat_logits.device)
        
        total = (
            self.classification_weight * cls_loss +
            self.regression_weight * reg_loss +
            self.campaign_weight * camp_loss +
            self.early_detection_weight * early_bonus
        )
        
        return {
            "total": total,
            "classification": cls_loss,
            "regression": reg_loss,
            "campaign": camp_loss,
            "early_detection": early_bonus,
        }
