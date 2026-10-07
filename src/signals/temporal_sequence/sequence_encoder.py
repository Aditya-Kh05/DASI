"""
Sequence Encoder — LSTM-based encoder for temporal scam sequences.

Takes a sequence of message embeddings (from conversation semantics)
and produces a temporal encoding that captures the progression pattern.

The key insight is that the ORDER of messages matters:
  "I am from CBI" → "Your Aadhaar is compromised" → "Transfer money now"
is very different from the same messages in random order.

Architecture:
  Sequence of message embeddings → Bi-LSTM → Attention pooling → Temporal embedding
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SequenceEncoder(nn.Module):
    """Bi-LSTM with attention for encoding message sequences.
    
    This encodes the temporal structure of a conversation, producing
    a single embedding that captures the progression pattern.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Attention mechanism for sequence pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        
        # Temporal position encoding (learned)
        self.position_encoding = nn.Embedding(512, input_dim)
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence of message embeddings.
        
        Args:
            sequence: (batch, seq_len, input_dim) — message embeddings in order
            lengths: (batch,) — actual sequence lengths (for masking padding)
            
        Returns:
            Tuple of:
              - temporal_embedding: (batch, output_dim) — sequence-level encoding
              - attention_weights: (batch, seq_len) — which messages were most important
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Add position encoding
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        sequence = sequence + self.position_encoding(positions)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(sequence)  # (batch, seq_len, hidden_dim * 2)
        
        # Attention pooling
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding positions
        if lengths is not None:
            mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim * 2)
        
        # Project to output
        temporal_embedding = self.output_projection(context)
        
        return temporal_embedding, attn_weights
