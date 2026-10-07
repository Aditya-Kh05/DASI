"""
Scam Feature Extractor — Extract 5 scam-indicative sub-scores from text.

The 5 dimensions of a digital arrest scam communication:
  1. authority_impersonation — claiming to be police/CBI/RBI/court
  2. urgency_pressure       — creating time pressure and fear
  3. isolation_instruction   — telling victim to stay silent
  4. financial_coercion      — requesting money transfer
  5. legal_terminology_abuse — misusing legal terms to sound legitimate

Each dimension is scored [0, 1] and combined into a feature vector
that feeds into the conversation semantics embedding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class ScamFeatureExtractor(nn.Module):
    """Neural feature extractor that produces 5 scam sub-scores.
    
    Architecture:
      Input: BERT [CLS] embedding (768-dim)
      → Linear(768, 512) + ReLU + Dropout
      → Linear(512, 256) + ReLU + Dropout
      → 5 parallel heads, each Linear(256, 1) + Sigmoid
      
    The parallel head design ensures each dimension is independently
    scored while sharing a common feature backbone.
    """
    
    SCAM_DIMENSIONS = [
        "authority_impersonation",
        "urgency_pressure",
        "isolation_instruction",
        "financial_coercion",
        "legal_terminology_abuse",
    ]
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_dimensions = len(self.SCAM_DIMENSIONS)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Independent scoring heads — one per scam dimension
        self.scoring_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for dim in self.SCAM_DIMENSIONS
        })
        
        # Embedding projection (from 5 scores → embedding_dim)
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.num_dimensions + hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
    
    def forward(self, cls_embedding: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract scam features from BERT [CLS] embedding.
        
        Args:
            cls_embedding: Shape (batch, 768) — BERT [CLS] token output
            
        Returns:
            Tuple of:
              - embedding: Shape (batch, 256) — dense representation for TCAFE
              - scores: Dict mapping dimension name → score tensor (batch, 1)
        """
        # Shared feature extraction
        shared_features = self.backbone(cls_embedding)
        
        # Independent dimension scores
        scores = {}
        score_values = []
        for dim_name in self.SCAM_DIMENSIONS:
            score = self.scoring_heads[dim_name](shared_features)
            scores[dim_name] = score
            score_values.append(score)
        
        # Concatenate scores with shared features for rich embedding
        score_tensor = torch.cat(score_values, dim=-1)  # (batch, 5)
        combined = torch.cat([score_tensor, shared_features], dim=-1)
        
        # Project to embedding space
        embedding = self.embedding_projection(combined)
        
        return embedding, scores
    
    def get_scam_report(self, cls_embedding: torch.Tensor) -> Dict[str, float]:
        """Generate a human-readable scam analysis report.
        
        Args:
            cls_embedding: Single sample [CLS] embedding (1, 768) or (768,)
            
        Returns:
            Dictionary with dimension scores and overall risk assessment
        """
        if cls_embedding.dim() == 1:
            cls_embedding = cls_embedding.unsqueeze(0)
        
        with torch.no_grad():
            _, scores = self.forward(cls_embedding)
        
        report = {}
        for dim_name, score in scores.items():
            report[dim_name] = round(score.item(), 4)
        
        # Overall risk is max of individual dimensions (any one high = risky)
        report["overall_risk"] = round(max(report.values()), 4)
        
        # Risk level classification
        risk = report["overall_risk"]
        if risk >= 0.8:
            report["risk_level"] = "CRITICAL"
        elif risk >= 0.6:
            report["risk_level"] = "HIGH"
        elif risk >= 0.4:
            report["risk_level"] = "MEDIUM"
        elif risk >= 0.2:
            report["risk_level"] = "LOW"
        else:
            report["risk_level"] = "MINIMAL"
        
        return report
