"""
Communication Graph Builder — Constructs heterogeneous graphs from communication data.

Builds a graph where:
  Nodes = phones, bank accounts, IPs, devices, victims
  Edges = called, transferred_to, same_device, same_ip, same_time_window

This graph structure enables GNN-based detection of organized scam networks.
If Phone A called Victim 1 and Victim 2, and Phone B (same device as A) called
Victim 3, the graph captures this relationship that would be invisible to
per-call analysis.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GraphNode:
    """A node in the communication graph."""
    node_id: str
    node_type: str          # "phone", "bank_account", "ip_address", "device_id", "victim"
    features: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def key(self) -> str:
        return f"{self.node_type}:{self.node_id}"


@dataclass
class GraphEdge:
    """An edge in the communication graph."""
    source_id: str
    target_id: str
    edge_type: str          # "called", "transferred_to", "same_device", "same_ip", "same_time_window"
    weight: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationGraphBuilder:
    """Incrementally build a communication graph from events.
    
    Supports:
      - Adding nodes and edges
      - Querying neighbors and subgraphs
      - Converting to PyTorch Geometric format for GNN processing
      - Tracking temporal evolution
    """
    
    NODE_TYPES = {"phone", "bank_account", "ip_address", "device_id", "victim"}
    EDGE_TYPES = {"called", "transferred_to", "same_device", "same_ip", "same_time_window"}
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._node_index: Dict[str, int] = {}
        self._type_index: Dict[str, List[str]] = defaultdict(list)
    
    def add_node(self, node_id: str, node_type: str,
                 features: Optional[Dict[str, Any]] = None) -> GraphNode:
        """Add a node to the graph."""
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            features=features or {},
        )
        key = node.key
        if key not in self.nodes:
            self._node_index[key] = len(self._node_index)
            self._type_index[node_type].append(key)
        self.nodes[key] = node
        return node
    
    def add_edge(self, source_type: str, source_id: str,
                 target_type: str, target_id: str,
                 edge_type: str, weight: float = 1.0,
                 timestamp: float = 0.0, metadata: Optional[Dict] = None):
        """Add an edge between two nodes (creates nodes if needed)."""
        # Ensure nodes exist
        source_key = f"{source_type}:{source_id}"
        target_key = f"{target_type}:{target_id}"
        
        if source_key not in self.nodes:
            self.add_node(source_id, source_type)
        if target_key not in self.nodes:
            self.add_node(target_id, target_type)
        
        edge = GraphEdge(
            source_id=source_key,
            target_id=target_key,
            edge_type=edge_type,
            weight=weight,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self.edges.append(edge)
        self.adjacency[source_key].add(target_key)
        self.adjacency[target_key].add(source_key)  # Undirected
    
    def add_call_event(self, caller_phone: str, callee_phone: str,
                       caller_ip: Optional[str] = None,
                       caller_device: Optional[str] = None,
                       timestamp: float = 0.0, duration: float = 0.0):
        """Add a phone call event with all related entities."""
        self.add_edge("phone", caller_phone, "phone", callee_phone,
                      "called", timestamp=timestamp,
                      metadata={"duration": duration})
        
        if caller_ip:
            self.add_edge("phone", caller_phone, "ip_address", caller_ip,
                          "same_ip", timestamp=timestamp)
        
        if caller_device:
            self.add_edge("phone", caller_phone, "device_id", caller_device,
                          "same_device", timestamp=timestamp)
    
    def add_transaction(self, from_account: str, to_account: str,
                        amount: float, timestamp: float = 0.0):
        """Add a financial transaction edge."""
        self.add_edge("bank_account", from_account, "bank_account", to_account,
                      "transferred_to", weight=amount, timestamp=timestamp,
                      metadata={"amount": amount})
    
    def get_neighbors(self, node_key: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighboring nodes, optionally filtered by edge type."""
        if edge_type is None:
            return list(self.adjacency.get(node_key, set()))
        
        neighbors = []
        for edge in self.edges:
            if edge.edge_type != edge_type:
                continue
            if edge.source_id == node_key:
                neighbors.append(edge.target_id)
            elif edge.target_id == node_key:
                neighbors.append(edge.source_id)
        return neighbors
    
    def get_node_degree(self, node_key: str) -> int:
        """Get the degree (number of connections) of a node."""
        return len(self.adjacency.get(node_key, set()))
    
    def get_subgraph(self, center_node: str, depth: int = 2) -> 'CommunicationGraphBuilder':
        """Extract a subgraph around a center node up to given depth."""
        visited = set()
        queue = [(center_node, 0)]
        visited.add(center_node)
        
        while queue:
            node, d = queue.pop(0)
            if d < depth:
                for neighbor in self.adjacency.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))
        
        # Build subgraph
        subgraph = CommunicationGraphBuilder()
        for key in visited:
            node = self.nodes.get(key)
            if node:
                subgraph.add_node(node.node_id, node.node_type, node.features)
        
        for edge in self.edges:
            if edge.source_id in visited and edge.target_id in visited:
                source_node = self.nodes[edge.source_id]
                target_node = self.nodes[edge.target_id]
                subgraph.add_edge(
                    source_node.node_type, source_node.node_id,
                    target_node.node_type, target_node.node_id,
                    edge.edge_type, edge.weight, edge.timestamp, edge.metadata,
                )
        
        return subgraph
    
    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert graph to tensor format for GNN processing.
        
        Returns:
            Dictionary with:
              - node_features: (num_nodes, feature_dim) — node feature matrix
              - edge_index: (2, num_edges) — edge connectivity
              - edge_type: (num_edges,) — edge type indices
              - node_type: (num_nodes,) — node type indices
        """
        num_nodes = len(self._node_index)
        type_to_idx = {t: i for i, t in enumerate(sorted(self.NODE_TYPES))}
        edge_type_to_idx = {t: i for i, t in enumerate(sorted(self.EDGE_TYPES))}
        
        # Node features: type one-hot + degree
        feature_dim = len(self.NODE_TYPES) + 1  # type one-hot + degree
        node_features = torch.zeros(num_nodes, feature_dim)
        node_types = torch.zeros(num_nodes, dtype=torch.long)
        
        for key, idx in self._node_index.items():
            node = self.nodes[key]
            type_idx = type_to_idx.get(node.node_type, 0)
            node_features[idx, type_idx] = 1.0
            node_features[idx, -1] = float(self.get_node_degree(key))
            node_types[idx] = type_idx
        
        # Edge index
        src_indices = []
        tgt_indices = []
        edge_types = []
        
        for edge in self.edges:
            src_idx = self._node_index.get(edge.source_id)
            tgt_idx = self._node_index.get(edge.target_id)
            if src_idx is not None and tgt_idx is not None:
                src_indices.append(src_idx)
                tgt_indices.append(tgt_idx)
                edge_types.append(edge_type_to_idx.get(edge.edge_type, 0))
                # Undirected: add reverse
                src_indices.append(tgt_idx)
                tgt_indices.append(src_idx)
                edge_types.append(edge_type_to_idx.get(edge.edge_type, 0))
        
        if src_indices:
            edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
            edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type_tensor = torch.zeros(0, dtype=torch.long)
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_type": edge_type_tensor,
            "node_type": node_types,
            "num_nodes": num_nodes,
            "num_edges": len(self.edges),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics for monitoring."""
        type_counts = {t: len(nodes) for t, nodes in self._type_index.items()}
        edge_type_counts = defaultdict(int)
        for edge in self.edges:
            edge_type_counts[edge.edge_type] += 1
        
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "node_type_counts": dict(type_counts),
            "edge_type_counts": dict(edge_type_counts),
            "avg_degree": np.mean([len(n) for n in self.adjacency.values()]) if self.adjacency else 0,
        }
