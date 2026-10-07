"""
Conversation Semantics Analyzer — Signal 1.

Analyzes text/transcribed speech to extract scam-indicative semantic features.
Uses a BERT-based model fine-tuned for digital arrest scam detection with
support for Hindi, English, and Hindi-English code-mixed text.

This signal answers: "What is the caller/message saying?"

Architecture:
  Input Text → Preprocessor → BERT Encoder → Feature Extractor → SignalOutput
                                                  ↓
                                            5 sub-scores:
                                            - authority_impersonation
                                            - urgency_pressure
                                            - isolation_instruction
                                            - financial_coercion
                                            - legal_terminology_abuse
"""

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from src.signals.base_signal import BaseSignalAnalyzer, SignalOutput
from src.signals.conversation_semantics.preprocessor import TextPreprocessor
from src.signals.conversation_semantics.feature_extractor import ScamFeatureExtractor


class ConversationSemanticsModel(nn.Module):
    """BERT-based model for scam semantics analysis.
    
    In prototype mode, uses a lightweight transformer encoder instead of
    full BERT to avoid large model downloads. The architecture mirrors
    BERT's approach: encode text → extract [CLS] → score on 5 dimensions.
    """
    
    def __init__(self, vocab_size: int = 30522, embedding_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3, max_seq_len: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS projection to match feature extractor's expected input
        self.cls_projection = nn.Linear(embedding_dim, 768)
        
        # Feature extractor (produces 5 sub-scores + embedding)
        self.feature_extractor = ScamFeatureExtractor(
            input_dim=768, hidden_dim=512, dropout=dropout
        )
    
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass: tokens → embedding + 5 sub-scores.
        
        Args:
            token_ids: (batch, seq_len) integer token indices
            attention_mask: (batch, seq_len) binary mask
            
        Returns:
            embedding: (batch, 256) dense representation
            scores: dict of dimension → (batch, 1) scores
        """
        batch_size, seq_len = token_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens + positions
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Convert attention mask to transformer format (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Extract [CLS] token (first token)
        cls_output = x[:, 0, :]
        
        # Project to feature extractor input dimension
        cls_projected = self.cls_projection(cls_output)
        
        # Extract scam features
        embedding, scores = self.feature_extractor(cls_projected)
        
        return embedding, scores


class ConversationSemanticsAnalyzer(BaseSignalAnalyzer):
    """Signal 1: Conversation Semantics Analyzer.
    
    Analyzes text content of communications to detect scam-indicative language.
    Supports Hindi, English, and code-mixed text.
    
    Usage:
        analyzer = ConversationSemanticsAnalyzer(config)
        result = analyzer.analyze("This is CBI calling. Your Aadhaar is linked to money laundering.")
        print(result.metadata)  # {'authority_impersonation': 0.92, 'urgency_pressure': 0.78, ...}
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        self.preprocessor = TextPreprocessor()
        self.model = ConversationSemanticsModel(
            embedding_dim=config.get("embedding_dim", 256),
            dropout=config.get("dropout", 0.3),
        ).to(self.device)
        self.model.eval()
        
        # Simple tokenizer (word-level) for prototype
        self._vocab = {}
        self._vocab_size = 30522
        self._max_seq_len = config.get("max_seq_length", 512)
        self._initialized = True
    
    def get_signal_name(self) -> str:
        return "conversation_semantics"
    
    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 256)
    
    def get_decay_rate(self) -> float:
        return self.config.get("confidence_decay_rate", 0.01)
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple hash-based tokenizer for prototype.
        
        In production, this would use a proper BERT tokenizer.
        """
        words = text.lower().split()
        if not words:
            words = ["call"]
        token_ids = [max(1, abs(hash(w)) % self._vocab_size) for w in words[:self._max_seq_len]]
        
        # Pad to fixed length
        if len(token_ids) < self._max_seq_len:
            token_ids += [0] * (self._max_seq_len - len(token_ids))
        
        return torch.tensor([token_ids], dtype=torch.long)
    
    def analyze(self, data: Any) -> SignalOutput:
        """Analyze text for scam indicators.
        
        Args:
            data: Input text string (or dict with 'text' or 'messages' key)
            
        Returns:
            SignalOutput with scam dimension scores in metadata
        """
        # Extract text from input
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            if "messages" in data and isinstance(data["messages"], list):
                text = " ".join(
                    m.get("text", "") if isinstance(m, dict) else str(m)
                    for m in data["messages"]
                )
            else:
                text = data.get("text", "")
        else:
            text = str(data)
        
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Extract keyword features (rule-based, always available)
        keyword_features = self.preprocessor.extract_keyword_features(cleaned_text)
        language_mix = self.preprocessor.get_language_mix(text)
        
        # Neural model analysis
        token_ids = self._tokenize(cleaned_text).to(self.device)
        attention_mask = (token_ids != 0).long()
        
        with torch.no_grad():
            embedding, scores = self.model(token_ids, attention_mask)
        
        neural_scores = {k: v.item() for k, v in scores.items()}
        
        # Combine neural scores with keyword features for robust calibrated confidence
        auth_hits = keyword_features.get("authority_hits", 0)
        urg_hits = keyword_features.get("urgency_hits", 0)
        iso_hits = keyword_features.get("isolation_hits", 0)
        fin_hits = keyword_features.get("financial_hits", 0)
        total_kw = keyword_features.get("total_scam_keywords", 0)

        # Keyword-grounded evidence per scam dimension
        kw_auth = min(1.0, auth_hits / 3.0)
        kw_urg = min(1.0, urg_hits / 3.0)
        kw_iso = min(1.0, iso_hits / 2.0)
        kw_fin = min(1.0, fin_hits / 2.0)
        kw_leg = min(1.0, (auth_hits + urg_hits) / 4.0)

        if total_kw == 0:
            calibrated_scores = {
                "authority_impersonation": 0.05,
                "urgency_pressure": 0.05,
                "isolation_instruction": 0.05,
                "financial_coercion": 0.05,
                "legal_terminology_abuse": 0.05,
            }
            combined_confidence = 0.08
        else:
            calibrated_scores = {
                "authority_impersonation": float(np.clip(0.15 * neural_scores.get("authority_impersonation", 0.5) + 0.85 * kw_auth, 0.05, 0.98)),
                "urgency_pressure": float(np.clip(0.15 * neural_scores.get("urgency_pressure", 0.5) + 0.85 * kw_urg, 0.05, 0.98)),
                "isolation_instruction": float(np.clip(0.15 * neural_scores.get("isolation_instruction", 0.5) + 0.85 * kw_iso, 0.05, 0.98)),
                "financial_coercion": float(np.clip(0.15 * neural_scores.get("financial_coercion", 0.5) + 0.85 * kw_fin, 0.05, 0.98)),
                "legal_terminology_abuse": float(np.clip(0.15 * neural_scores.get("legal_terminology_abuse", 0.5) + 0.85 * kw_leg, 0.05, 0.98)),
            }
            combined_confidence = float(min(1.0, 0.40 + 0.60 * min(1.0, total_kw / 3.0)))

        # Build rich metadata
        metadata = {
            **calibrated_scores,
            "keyword_features": keyword_features,
            "language": language_mix,
            "text_length": len(cleaned_text),
            "risk_level": self._classify_risk(calibrated_scores),
        }
        
        out_embedding = embedding.squeeze(0) * combined_confidence

        return self._create_output(
            embedding=out_embedding,
            confidence=combined_confidence,
            metadata=metadata,
        )
    
    @staticmethod
    def _classify_risk(scores: Dict[str, float]) -> str:
        """Classify overall risk level from sub-scores."""
        max_score = max(scores.values()) if scores else 0.0
        if max_score >= 0.8:
            return "CRITICAL"
        elif max_score >= 0.6:
            return "HIGH"
        elif max_score >= 0.4:
            return "MEDIUM"
        elif max_score >= 0.2:
            return "LOW"
        return "MINIMAL"
