"""
Network Graph Generator — Synthetic Communication Graph Topologies.

Generates realistic network graphs representing communication patterns
between phones, bank accounts, IPs, and devices for:
  - Organized scam syndicates (star/hub topologies, shared infrastructure)
  - Legitimate communication networks (small-world, sparse graphs)

The graphs are directly consumable by the NetworkGraphAnalyzer's GNN.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GeneratedGraph:
    """A synthetic communication graph."""
    graph_id: str
    node_features: torch.Tensor         # (num_nodes, feature_dim)
    edge_index: torch.Tensor            # (2, num_edges)
    edge_type: torch.Tensor             # (num_edges,) — edge type indices
    node_labels: torch.Tensor           # (num_nodes,) — 0=benign, 1=scam
    node_types: List[str]               # per-node type names
    node_keys: List[str]                # per-node unique identifiers
    is_syndicate: bool                  # whether this graph contains a scam syndicate
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "graph_id": self.graph_id,
            "num_nodes": len(self.node_keys),
            "num_edges": self.edge_index.shape[1],
            "is_syndicate": self.is_syndicate,
            "node_types": dict(zip(*np.unique(self.node_types, return_counts=True))),
            "num_scam_nodes": int(self.node_labels.sum().item()),
        }


class NetworkGraphGenerator:
    """Generates synthetic communication network graphs.
    
    Syndicate topology patterns:
      - Star topology: One hub phone calling many victims
      - Chain topology: Sequential forwarding through multiple handlers
      - Cluster topology: Small group of phones sharing infrastructure
      - Hybrid: Combination with shared bank accounts and IPs
    
    Legitimate topology patterns:
      - Small-world: High clustering, short path lengths
      - Random sparse: Low density, no clear hubs
    """
    
    NODE_TYPES = ["phone", "bank_account", "ip_address", "device_id", "victim"]
    EDGE_TYPES = ["called", "transferred_to", "same_device", "same_ip", "same_time_window"]
    NODE_FEATURE_DIM = 16
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._graph_counter = 0
    
    def _generate_node_features(
        self,
        node_type: str,
        is_scam_node: bool,
    ) -> np.ndarray:
        """Generate feature vector for a node.
        
        Scam nodes have:
          - Higher call frequency features
          - More connections to bank accounts
          - Shorter call durations (scripted)
          - Activity concentrated in specific time windows
        """
        features = np.zeros(self.NODE_FEATURE_DIM, dtype=np.float32)
        
        type_index = self.NODE_TYPES.index(node_type)
        features[type_index] = 1.0  # One-hot for node type
        
        if node_type == "phone":
            # Call frequency (normalized)
            features[5] = self.np_rng.uniform(0.6, 1.0) if is_scam_node else self.np_rng.uniform(0.05, 0.4)
            # Avg call duration (scammers: short, scripted)
            features[6] = self.np_rng.uniform(0.1, 0.35) if is_scam_node else self.np_rng.uniform(0.3, 0.9)
            # Number of unique contacts
            features[7] = self.np_rng.uniform(0.5, 1.0) if is_scam_node else self.np_rng.uniform(0.05, 0.3)
            # Activity time concentration (scammers work specific shifts)
            features[8] = self.np_rng.uniform(0.7, 1.0) if is_scam_node else self.np_rng.uniform(0.2, 0.6)
            # International call ratio
            features[9] = self.np_rng.uniform(0.0, 0.2) if is_scam_node else self.np_rng.uniform(0.0, 0.1)
        
        elif node_type == "bank_account":
            # Transaction frequency
            features[5] = self.np_rng.uniform(0.7, 1.0) if is_scam_node else self.np_rng.uniform(0.1, 0.5)
            # Avg transaction amount (normalized)
            features[6] = self.np_rng.uniform(0.5, 1.0) if is_scam_node else self.np_rng.uniform(0.05, 0.4)
            # Account age (scam accounts are new)
            features[7] = self.np_rng.uniform(0.0, 0.2) if is_scam_node else self.np_rng.uniform(0.3, 1.0)
            # Number of incoming sources
            features[8] = self.np_rng.uniform(0.6, 1.0) if is_scam_node else self.np_rng.uniform(0.05, 0.3)
        
        elif node_type == "ip_address":
            # VPN/proxy likelihood
            features[5] = self.np_rng.uniform(0.6, 1.0) if is_scam_node else self.np_rng.uniform(0.0, 0.2)
            # Geographic anomaly
            features[6] = self.np_rng.uniform(0.5, 1.0) if is_scam_node else self.np_rng.uniform(0.0, 0.15)
            # Connection count
            features[7] = self.np_rng.uniform(0.4, 0.9) if is_scam_node else self.np_rng.uniform(0.05, 0.3)
        
        elif node_type == "device_id":
            # Multiple SIM usage
            features[5] = self.np_rng.uniform(0.7, 1.0) if is_scam_node else self.np_rng.uniform(0.0, 0.1)
            # Rooted/jailbroken flag
            features[6] = self.np_rng.uniform(0.5, 1.0) if is_scam_node else self.np_rng.uniform(0.0, 0.05)
        
        elif node_type == "victim":
            # These are always benign nodes
            features[5] = self.np_rng.uniform(0.05, 0.2)
            features[6] = self.np_rng.uniform(0.3, 0.8)
            features[7] = self.np_rng.uniform(0.5, 1.0)
        
        # Add small noise
        features += self.np_rng.normal(0, 0.02, self.NODE_FEATURE_DIM).astype(np.float32)
        
        return features
    
    def _generate_syndicate_graph(
        self,
        topology: str = "star",
        num_scam_phones: int = 3,
        num_victims: int = 5,
    ) -> GeneratedGraph:
        """Generate a scam syndicate communication graph.
        
        Topology types:
          - "star": One hub phone connected to many victims
          - "chain": Sequential handler forwarding
          - "cluster": Multiple connected scam phones
        """
        self._graph_counter += 1
        graph_id = f"syndicate_{self._graph_counter:05d}"
        
        nodes = []       # (type, key, is_scam)
        edges = []       # (src_idx, dst_idx, edge_type_idx)
        
        # Create scam phone nodes
        scam_phone_indices = []
        for i in range(num_scam_phones):
            idx = len(nodes)
            nodes.append(("phone", f"phone:scam_{i}", True))
            scam_phone_indices.append(idx)
        
        # Create shared infrastructure
        # Shared bank account (mule account)
        num_mule_accounts = self.rng.randint(1, 3)
        mule_indices = []
        for i in range(num_mule_accounts):
            idx = len(nodes)
            nodes.append(("bank_account", f"bank:mule_{i}", True))
            mule_indices.append(idx)
        
        # Shared IP addresses
        num_shared_ips = self.rng.randint(1, 2)
        ip_indices = []
        for i in range(num_shared_ips):
            idx = len(nodes)
            nodes.append(("ip_address", f"ip:scam_{i}", True))
            ip_indices.append(idx)
        
        # Shared devices
        num_devices = self.rng.randint(1, 2)
        device_indices = []
        for i in range(num_devices):
            idx = len(nodes)
            nodes.append(("device_id", f"device:scam_{i}", True))
            device_indices.append(idx)
        
        # Create victim nodes
        victim_indices = []
        for i in range(num_victims):
            idx = len(nodes)
            nodes.append(("victim", f"victim:{i}", False))
            victim_indices.append(idx)
            
            # Victim also has a phone and bank account
            phone_idx = len(nodes)
            nodes.append(("phone", f"phone:victim_{i}", False))
            bank_idx = len(nodes)
            nodes.append(("bank_account", f"bank:victim_{i}", False))
            
            # Victim phone ↔ victim
            edges.append((phone_idx, idx, 0))  # called
            edges.append((idx, phone_idx, 0))
            
            # Victim bank → mule account (money transfer)
            mule = self.rng.choice(mule_indices)
            edges.append((bank_idx, mule, 1))  # transferred_to
        
        # Build topology-specific connections
        if topology == "star":
            # One hub phone calls all victims
            hub = scam_phone_indices[0]
            for v_idx in victim_indices:
                v_phone_idx = v_idx + 1  # victim's phone is right after victim node
                edges.append((hub, v_phone_idx, 0))  # called
            
            # Other scam phones connect to hub
            for sp in scam_phone_indices[1:]:
                edges.append((sp, hub, 0))
                edges.append((hub, sp, 0))
        
        elif topology == "chain":
            # Sequential forwarding: phone0 → phone1 → phone2 → victims
            for i in range(len(scam_phone_indices) - 1):
                edges.append((scam_phone_indices[i], scam_phone_indices[i + 1], 0))
            
            # Last phone calls victims
            last_phone = scam_phone_indices[-1]
            for v_idx in victim_indices:
                v_phone_idx = v_idx + 1
                edges.append((last_phone, v_phone_idx, 0))
        
        elif topology == "cluster":
            # All scam phones interconnected
            for i, sp1 in enumerate(scam_phone_indices):
                for sp2 in scam_phone_indices[i + 1:]:
                    edges.append((sp1, sp2, 0))
                    edges.append((sp2, sp1, 0))
            
            # Each scam phone calls some victims
            for v_idx in victim_indices:
                caller = self.rng.choice(scam_phone_indices)
                v_phone_idx = v_idx + 1
                edges.append((caller, v_phone_idx, 0))
        
        # Connect scam phones to shared infrastructure
        for sp in scam_phone_indices:
            # same_ip
            for ip_idx in ip_indices:
                edges.append((sp, ip_idx, 3))
            # same_device (some phones share device)
            if self.rng.random() < 0.6:
                dev = self.rng.choice(device_indices)
                edges.append((sp, dev, 2))
        
        # Scam phones to mule accounts
        for sp in scam_phone_indices:
            mule = self.rng.choice(mule_indices)
            edges.append((sp, mule, 1))
        
        # Some same_time_window edges (calls happening together)
        for i, sp1 in enumerate(scam_phone_indices):
            for sp2 in scam_phone_indices[i + 1:]:
                if self.rng.random() < 0.4:
                    edges.append((sp1, sp2, 4))
                    edges.append((sp2, sp1, 4))
        
        return self._build_graph(graph_id, nodes, edges, is_syndicate=True, metadata={
            "topology": topology,
            "num_scam_phones": num_scam_phones,
            "num_victims": num_victims,
        })
    
    def _generate_legitimate_graph(
        self,
        num_nodes: int = 10,
    ) -> GeneratedGraph:
        """Generate a legitimate communication graph (no syndicate structure)."""
        self._graph_counter += 1
        graph_id = f"legit_{self._graph_counter:05d}"
        
        nodes = []
        edges = []
        
        # Mix of node types
        for i in range(num_nodes):
            node_type = self.rng.choices(
                ["phone", "bank_account", "ip_address", "device_id"],
                weights=[0.4, 0.25, 0.2, 0.15],
                k=1,
            )[0]
            nodes.append((node_type, f"{node_type}:legit_{i}", False))
        
        # Random sparse connections (small-world-ish)
        num_edges = self.rng.randint(num_nodes, num_nodes * 2)
        for _ in range(num_edges):
            src = self.rng.randint(0, num_nodes - 1)
            dst = self.rng.randint(0, num_nodes - 1)
            if src != dst:
                edge_type = self.rng.randint(0, len(self.EDGE_TYPES) - 1)
                edges.append((src, dst, edge_type))
        
        return self._build_graph(graph_id, nodes, edges, is_syndicate=False, metadata={
            "num_nodes": num_nodes,
        })
    
    def _build_graph(
        self,
        graph_id: str,
        nodes: List[Tuple[str, str, bool]],
        edges: List[Tuple[int, int, int]],
        is_syndicate: bool,
        metadata: Dict[str, Any],
    ) -> GeneratedGraph:
        """Construct the final GeneratedGraph from raw nodes and edges."""
        num_nodes = len(nodes)
        
        # Generate node features
        feature_list = []
        node_types = []
        node_keys = []
        node_labels = []
        
        for node_type, node_key, is_scam in nodes:
            feature_list.append(self._generate_node_features(node_type, is_scam))
            node_types.append(node_type)
            node_keys.append(node_key)
            node_labels.append(1 if is_scam else 0)
        
        node_features = torch.tensor(np.stack(feature_list), dtype=torch.float32)
        labels = torch.tensor(node_labels, dtype=torch.float32)
        
        # Build edge tensors
        if edges:
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]
            type_list = [e[2] for e in edges]
            
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_type = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        
        return GeneratedGraph(
            graph_id=graph_id,
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            node_labels=labels,
            node_types=node_types,
            node_keys=node_keys,
            is_syndicate=is_syndicate,
            metadata=metadata,
        )
    
    def generate_sample(
        self,
        is_syndicate: Optional[bool] = None,
        topology: Optional[str] = None,
    ) -> GeneratedGraph:
        """Generate one graph sample."""
        if is_syndicate is None:
            is_syndicate = self.rng.random() < 0.4
        
        if is_syndicate:
            if topology is None:
                topology = self.rng.choice(["star", "chain", "cluster"])
            return self._generate_syndicate_graph(
                topology=topology,
                num_scam_phones=self.rng.randint(2, 5),
                num_victims=self.rng.randint(3, 8),
            )
        else:
            return self._generate_legitimate_graph(
                num_nodes=self.rng.randint(5, 15),
            )
    
    def generate_batch(
        self,
        num_syndicate: int = 300,
        num_legitimate: int = 300,
    ) -> List[GeneratedGraph]:
        """Generate a balanced batch of graphs."""
        graphs = []
        
        for _ in range(num_syndicate):
            topology = self.rng.choice(["star", "chain", "cluster"])
            graphs.append(self.generate_sample(is_syndicate=True, topology=topology))
        
        for _ in range(num_legitimate):
            graphs.append(self.generate_sample(is_syndicate=False))
        
        self.rng.shuffle(graphs)
        return graphs
