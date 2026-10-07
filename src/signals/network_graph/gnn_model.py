"""
Scam Network GNN — Graph Neural Network for scam network detection.

Uses a GraphSAGE-like architecture to learn node embeddings that capture
the structural patterns of organized scam networks.

Key insight: Even if a phone number has never been seen before, its
position in the communication graph (connected to known scam clusters,
calling many victims in short time windows) reveals its true nature.

Architecture:
  Node features → GNN Layers (with edge-type-aware aggregation) → Node embeddings
  → Readout → Graph-level risk score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GraphConvLayer(nn.Module):
    """Edge-type-aware graph convolution layer.
    
    Unlike standard GCN, this layer uses different transformation
    matrices for different edge types (called, transferred_to, etc.),
    similar to R-GCN but more lightweight.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int = 5):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        
        # Type-specific transformations
        self.edge_transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # Self-loop transformation
        self.self_transform = nn.Linear(in_dim, out_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, 1),
        )
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: aggregate neighbor information.
        
        Args:
            node_features: (num_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
            
        Returns:
            Updated node features: (num_nodes, out_dim)
        """
        num_nodes = node_features.shape[0]
        
        # Self transformation
        out = self.self_transform(node_features)
        
        if edge_index.shape[1] == 0:
            return self.norm(F.relu(out))
        
        # Aggregate messages from neighbors by edge type
        src, dst = edge_index[0], edge_index[1]
        
        # Compute messages for each edge type
        messages = torch.zeros(num_nodes, self.out_dim, device=node_features.device)
        counts = torch.zeros(num_nodes, 1, device=node_features.device)
        
        for etype in range(self.num_edge_types):
            mask = edge_type == etype
            if not mask.any():
                continue
            
            src_masked = src[mask]
            dst_masked = dst[mask]
            
            # Transform source node features with edge-type-specific weight
            transformed = self.edge_transforms[etype](node_features[src_masked])
            
            # Compute attention weights
            dst_features = out[dst_masked]
            attn_input = torch.cat([transformed, dst_features], dim=-1)
            attn_weights = torch.sigmoid(self.attention(attn_input))
            weighted_messages = transformed * attn_weights
            
            # Scatter-add to destination nodes
            messages.scatter_add_(0, dst_masked.unsqueeze(1).expand_as(weighted_messages),
                                  weighted_messages)
            counts.scatter_add_(0, dst_masked.unsqueeze(1),
                                torch.ones(dst_masked.shape[0], 1, device=node_features.device))
        
        # Mean aggregation
        counts = counts.clamp(min=1)
        messages = messages / counts
        
        # Combine self and neighbor information
        out = out + messages
        out = self.norm(F.relu(out))
        out = self.dropout(out)
        
        return out


class ScamNetworkGNN(nn.Module):
    """Multi-layer GNN for scam network detection.
    
    Produces:
      1. Node embeddings — risk assessment for each entity
      2. Graph-level embedding — overall network risk
      3. Risk scores — per-node and graph-level scam probability
    """
    
    def __init__(
        self,
        input_dim: int = 6,      # node feature dimension
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        num_edge_types: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GraphConvLayer(hidden_dim, hidden_dim, num_edge_types)
            )
        
        # Node-level classifier
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Node embedding projection to output_dim
        self.node_projection = nn.Linear(hidden_dim, output_dim)
        
        # Graph-level readout
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        
        # Graph-level classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN.
        
        Args:
            node_features: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
            
        Returns:
            Dictionary with:
              - node_embeddings: (num_nodes, hidden_dim)
              - node_risk_scores: (num_nodes, 1)
              - graph_embedding: (output_dim,)
              - graph_risk_score: scalar
        """
        # Project input features
        x = self.input_projection(node_features)
        
        # Apply GNN layers with residual connections
        for layer in self.gnn_layers:
            x_new = layer(x, edge_index, edge_type)
            x = x + x_new  # Residual
        
        # Node-level risk
        node_risk = self.node_classifier(x)
        
        # Graph-level readout (mean pooling)
        graph_embedding = self.graph_readout(torch.mean(x, dim=0, keepdim=True))
        graph_risk = self.graph_classifier(graph_embedding)
        
        # Project node embeddings to output dimension
        projected_nodes = self.node_projection(x)
        
        return {
            "node_embeddings": projected_nodes,
            "node_risk_scores": node_risk,
            "graph_embedding": graph_embedding.squeeze(0),
            "graph_risk_score": graph_risk.squeeze(),
        }
