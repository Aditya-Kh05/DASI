"""
Network Graph Analyzer (Relationship Mapper) — Signal 4.

This signal answers: "Is this phone number / account part of an organized scam network?"

Combines:
  - CommunicationGraphBuilder: Constructs the relationship graph
  - ScamNetworkGNN: Produces node/graph embeddings and risk scores
  - CampaignClusterDetector: Groups suspicious entities into campaigns
"""

import time
from typing import Any, Dict, Optional

import torch
import numpy as np

from src.signals.base_signal import BaseSignalAnalyzer, SignalOutput
from src.signals.network_graph.graph_builder import CommunicationGraphBuilder
from src.signals.network_graph.gnn_model import ScamNetworkGNN
from src.signals.network_graph.cluster_detector import CampaignClusterDetector


class NetworkGraphAnalyzer(BaseSignalAnalyzer):
    """Signal 4: Communication Network Graph Intelligence.
    
    Maps relationships between entities (phones, accounts, IPs, devices)
    to identify organized scam networks.
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        
        embedding_dim = config.get("embedding_dim", 128)
        
        # Communication graph (shared, grows over time)
        self.graph = CommunicationGraphBuilder()
        
        # GNN model
        self.gnn = ScamNetworkGNN(
            input_dim=6,  # 5 node types + degree
            hidden_dim=config.get("gnn_hidden_dim", 256),
            output_dim=embedding_dim,
            num_layers=config.get("gnn_layers", 3),
            num_edge_types=5,
            dropout=config.get("dropout", 0.2),
        ).to(self.device)
        
        # Cluster detector
        self.cluster_detector = CampaignClusterDetector(
            min_cluster_size=config.get("cluster_min_size", 3),
            similarity_threshold=0.7,
        )
        
        self.gnn.eval()
        self._initialized = True
    
    def get_signal_name(self) -> str:
        return "network_graph"
    
    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 128)
    
    def get_decay_rate(self) -> float:
        return self.config.get("confidence_decay_rate", 0.005)
    
    def analyze(self, data: Any) -> SignalOutput:
        """Analyze network relationships around a communication event.
        
        Args:
            data: Dict with:
                - 'caller_phone': str
                - 'callee_phone': str
                - 'caller_ip': str (optional)
                - 'caller_device': str (optional)
                - 'timestamp': float
                - 'transaction': dict (optional) — {'from': ..., 'to': ..., 'amount': ...}
                
        Returns:
            SignalOutput with network risk and campaign cluster information
        """
        if not isinstance(data, dict):
            raise ValueError("NetworkGraphAnalyzer expects dict with communication event data")
        
        # Step 1: Add event to graph
        caller = data.get("caller_phone", "unknown_caller")
        callee = data.get("callee_phone", "unknown_callee")
        timestamp = data.get("timestamp", time.time())
        
        self.graph.add_call_event(
            caller_phone=caller,
            callee_phone=callee,
            caller_ip=data.get("caller_ip"),
            caller_device=data.get("caller_device"),
            timestamp=timestamp,
            duration=data.get("duration", 0.0),
        )
        
        # Add transaction if present
        if "transaction" in data:
            txn = data["transaction"]
            self.graph.add_transaction(
                from_account=txn.get("from", ""),
                to_account=txn.get("to", ""),
                amount=txn.get("amount", 0.0),
                timestamp=timestamp,
            )
        
        # Step 2: Convert graph to tensors
        graph_tensors = self.graph.to_tensors()
        
        # Step 3: Run GNN
        with torch.no_grad():
            node_features = graph_tensors["node_features"].to(self.device)
            edge_index = graph_tensors["edge_index"].to(self.device)
            edge_type = graph_tensors["edge_type"].to(self.device)
            
            gnn_output = self.gnn(node_features, edge_index, edge_type)
        
        # Step 4: Get the caller's node embedding and risk
        caller_key = f"phone:{caller}"
        caller_idx = self.graph._node_index.get(caller_key, 0)
        
        node_embedding = gnn_output["node_embeddings"][caller_idx]
        node_risk = gnn_output["node_risk_scores"][caller_idx].item()
        graph_risk = gnn_output["graph_risk_score"].item()
        
        # Step 5: Detect campaign clusters
        node_keys = list(self.graph._node_index.keys())
        clusters = self.cluster_detector.detect_clusters(
            gnn_output["node_embeddings"],
            node_keys,
            gnn_output["node_risk_scores"],
        )
        
        caller_cluster = self.cluster_detector.get_node_cluster(caller_key)
        
        # Step 6: Compute network-level risk signals
        caller_degree = self.graph.get_node_degree(caller_key)
        
        # Build metadata
        metadata = {
            "caller_node_risk": round(node_risk, 4),
            "graph_risk": round(graph_risk, 4),
            "caller_degree": caller_degree,
            "campaign_cluster_id": caller_cluster,
            "campaign_cluster_risk": round(
                self.cluster_detector.get_campaign_risk(caller_cluster), 4
            ) if caller_cluster >= 0 else 0.0,
            "num_active_clusters": len(clusters),
            "graph_stats": self.graph.get_statistics(),
        }
        
        # Confidence: higher with more graph data and clearer risk signal
        data_confidence = min(1.0, len(self.graph.nodes) / 20.0)  # More data = more confident
        risk_clarity = abs(node_risk - 0.5) * 2  # Distance from uncertain
        confidence = 0.5 * data_confidence + 0.5 * risk_clarity
        
        return self._create_output(
            embedding=node_embedding.cpu(),
            confidence=confidence,
            metadata=metadata,
        )
