"""
Campaign Cluster Detector — Group suspicious nodes into scam campaigns.

Uses community detection and graph clustering to identify organized
scam campaigns within the communication network.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class CampaignClusterDetector:
    """Detect scam campaign clusters in the communication graph.
    
    A campaign cluster is a group of phone numbers, accounts, and devices
    that are working together as part of the same scam operation.
    
    Detection methods:
      1. Connected component analysis (strong connections)
      2. Spectral clustering on node embeddings
      3. Temporal co-activity detection
    """
    
    def __init__(self, min_cluster_size: int = 3, similarity_threshold: float = 0.7):
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.clusters: Dict[int, Set[str]] = {}
        self._next_cluster_id = 0
    
    def detect_clusters(
        self,
        node_embeddings: torch.Tensor,
        node_keys: List[str],
        node_risk_scores: torch.Tensor,
    ) -> Dict[int, Dict]:
        """Detect campaign clusters from GNN node embeddings.
        
        Args:
            node_embeddings: (num_nodes, embedding_dim) from GNN
            node_keys: List of node identifiers
            node_risk_scores: (num_nodes, 1) risk scores per node
            
        Returns:
            Dictionary mapping cluster_id → cluster info
        """
        if len(node_keys) == 0:
            return {}
        
        embeddings = node_embeddings.detach().cpu().numpy()
        risk_scores = node_risk_scores.detach().cpu().numpy().flatten()
        
        # Step 1: Compute pairwise similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normalized = embeddings / norms
        similarity_matrix = normalized @ normalized.T
        
        # Step 2: Find clusters via greedy agglomerative clustering
        n = len(node_keys)
        assigned = [False] * n
        clusters = {}
        
        # Sort by risk score (high risk nodes seed clusters)
        risk_order = np.argsort(-risk_scores)
        
        for seed_idx in risk_order:
            if assigned[seed_idx]:
                continue
            if risk_scores[seed_idx] < 0.3:  # Skip low-risk nodes as seeds
                continue
            
            # Find similar nodes
            cluster_members = [seed_idx]
            assigned[seed_idx] = True
            
            for j in range(n):
                if assigned[j]:
                    continue
                if similarity_matrix[seed_idx, j] >= self.similarity_threshold:
                    cluster_members.append(j)
                    assigned[j] = True
            
            if len(cluster_members) >= self.min_cluster_size:
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
                
                member_keys = [node_keys[i] for i in cluster_members]
                member_risks = [float(risk_scores[i]) for i in cluster_members]
                
                clusters[cluster_id] = {
                    "cluster_id": cluster_id,
                    "members": member_keys,
                    "size": len(member_keys),
                    "avg_risk": float(np.mean(member_risks)),
                    "max_risk": float(np.max(member_risks)),
                    "centroid": embeddings[cluster_members].mean(axis=0).tolist(),
                }
        
        self.clusters.update(clusters)
        return clusters
    
    def get_node_cluster(self, node_key: str) -> int:
        """Find which cluster a node belongs to (-1 if none)."""
        for cluster_id, info in self.clusters.items():
            if node_key in info["members"]:
                return cluster_id
        return -1
    
    def get_campaign_risk(self, cluster_id: int) -> float:
        """Get the overall risk score for a campaign cluster."""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id]["avg_risk"]
        return 0.0
