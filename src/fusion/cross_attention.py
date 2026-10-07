"""
Cross-Signal Correlation Attention — TCAFE Component 2.

Implements multi-head cross-attention where each signal attends to
all other signals, producing correlation-adjusted embeddings.

Core insight: Signals don't just add up — they INTERACT.

Examples of cross-signal interactions:
  - NLP "authority impersonation" + Voice "deepfake detected" → REINFORCEMENT
  - NLP "high urgency" + Temporal "Phase 1 only" → DAMPENING (too early)
  - Network "known cluster" + Geo "new region burst" → CAMPAIGN SPREADING

The attention mechanism learns these interaction patterns from data,
producing embeddings that capture the full cross-signal context.

Mathematical formulation:
  ẽᵢ = MultiHeadAttn(Q=eᵢ, K=[e₁,...,e₅], V=[e₁,...,e₅], weights=[c̃₁,...,c̃₅])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossSignalAttention(nn.Module):
    """Multi-head cross-attention for signal interaction modeling.
    
    Each signal embedding attends to all other signal embeddings,
    weighted by their decayed confidence scores.
    
    This produces correlation-adjusted embeddings where each signal
    is enriched with information from all other signals.
    """
    
    def __init__(
        self,
        signal_dims: dict,          # signal_name → embedding_dim
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.signal_names = list(signal_dims.keys())
        self.num_signals = len(self.signal_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Project each signal to a common dimension for attention
        self.signal_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in signal_dims.items()
        })
        
        # Multi-head attention components
        self.q_projections = nn.ModuleDict({
            name: nn.Linear(output_dim, output_dim)
            for name in self.signal_names
        })
        
        self.k_projection = nn.Linear(output_dim, output_dim)
        self.v_projection = nn.Linear(output_dim, output_dim)
        
        # Output projections (one per signal)
        self.output_projections = nn.ModuleDict({
            name: nn.Linear(output_dim, output_dim)
            for name in self.signal_names
        })
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(output_dim)
            for name in self.signal_names
        })
        
        # Correlation strength gate (learned per signal pair)
        # This captures which signal pairs should interact strongly
        self.correlation_gate = nn.Parameter(
            torch.ones(self.num_signals, self.num_signals) / self.num_signals
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        signal_embeddings: dict,     # signal_name → (batch, signal_dim)
        decayed_confidences: dict,   # signal_name → (batch,) or (batch, 1)
    ) -> Tuple[dict, torch.Tensor]:
        """Compute cross-signal attention.
        
        Args:
            signal_embeddings: Dict of signal_name → embedding tensor
            decayed_confidences: Dict of signal_name → confidence score
            
        Returns:
            Tuple of:
              - enriched_embeddings: Dict of signal_name → (batch, output_dim)
              - attention_matrix: (num_signals, num_signals) — interaction strengths
        """
        batch_size = next(iter(signal_embeddings.values())).shape[0]
        
        # Step 1: Project all signals to common dimension
        projected = {}
        for name in self.signal_names:
            if name in signal_embeddings:
                projected[name] = self.signal_projections[name](signal_embeddings[name])
            else:
                projected[name] = torch.zeros(batch_size, self.output_dim,
                                              device=next(iter(signal_embeddings.values())).device)
        
        # Step 2: Build key-value tensors from all signals
        # Shape: (batch, num_signals, output_dim)
        all_signals = torch.stack([projected[name] for name in self.signal_names], dim=1)
        
        # Keys and Values (shared across all queries)
        K = self.k_projection(all_signals)  # (batch, num_signals, output_dim)
        V = self.v_projection(all_signals)
        
        # Reshape for multi-head attention
        K = K.view(batch_size, self.num_signals, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.num_signals, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Build confidence-weighted attention mask
        conf_values = []
        for name in self.signal_names:
            if name in decayed_confidences:
                c = decayed_confidences[name]
                if isinstance(c, (int, float)):
                    c = torch.full((batch_size,), float(c), dtype=torch.float32, device=all_signals.device)
                elif isinstance(c, torch.Tensor):
                    if c.dim() == 0:
                        c = c.repeat(batch_size)
                    elif c.dim() > 1:
                        c = c.squeeze(-1)
                conf_values.append(c)
            else:
                conf_values.append(torch.zeros(batch_size, device=all_signals.device))
        
        # (batch, num_signals)
        confidence_mask = torch.stack(conf_values, dim=1)
        
        # Apply correlation gate (learned signal-pair interaction strengths)
        gate = torch.softmax(self.correlation_gate, dim=-1)
        
        # Step 4: Compute attention for each signal (as query)
        enriched = {}
        attention_scores_matrix = torch.zeros(self.num_signals, self.num_signals)
        
        for i, name in enumerate(self.signal_names):
            if name not in signal_embeddings:
                enriched[name] = torch.zeros(batch_size, self.output_dim, device=all_signals.device)
                continue
            
            # Query for this signal
            Q = self.q_projections[name](projected[name])  # (batch, output_dim)
            Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores: (batch, num_heads, 1, num_signals)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            # Weight by confidence and correlation gate
            conf_weight = confidence_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_signals)
            gate_weight = gate[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, num_signals)
            
            attn_scores = attn_scores + torch.log(conf_weight + 1e-10) + torch.log(gate_weight + 1e-10)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Store attention for visualization
            attention_scores_matrix[i] = attn_weights.mean(dim=(0, 1)).squeeze(0).detach().cpu()
            
            # Weighted sum of values
            context = torch.matmul(attn_weights, V)  # (batch, num_heads, 1, head_dim)
            context = context.transpose(1, 2).contiguous().view(batch_size, self.output_dim)
            
            # Output projection + residual + layer norm
            output = self.output_projections[name](context)
            output = self.layer_norms[name](output + projected[name])
            
            enriched[name] = output
        
        return enriched, attention_scores_matrix
    
    def get_correlation_matrix(self) -> dict:
        """Get the learned signal correlation strengths for visualization."""
        gate = torch.softmax(self.correlation_gate, dim=-1).detach().cpu().numpy()
        
        matrix = {}
        for i, src in enumerate(self.signal_names):
            matrix[src] = {}
            for j, tgt in enumerate(self.signal_names):
                matrix[src][tgt] = round(float(gate[i][j]), 4)
        
        return matrix
