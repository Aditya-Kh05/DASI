"""
Spatial-Temporal Clustering — Detect geographic hotspots of scam activity.

Uses a custom ST-DBSCAN (Spatial-Temporal DBSCAN) implementation to cluster
scam events that are close in both space AND time.

Key insight: Scam campaigns often start in specific regions and spread outward.
Detecting geographic clusters early allows preemptive warnings to adjacent regions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GeoEvent:
    """A geographically-located scam event."""
    event_id: str
    latitude: float
    longitude: float
    timestamp: float
    risk_score: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SpatialTemporalClusterer:
    """ST-DBSCAN implementation for scam campaign geographic analysis.
    
    Parameters:
        spatial_eps: Maximum distance (km) between events in a cluster
        temporal_eps: Maximum time difference (seconds) between events
        min_samples: Minimum events to form a cluster
    """
    
    def __init__(self, spatial_eps: float = 50.0, temporal_eps: float = 3600.0,
                 min_samples: int = 5):
        self.spatial_eps = spatial_eps
        self.temporal_eps = temporal_eps
        self.min_samples = min_samples
        
        self.events: List[GeoEvent] = []
        self.clusters: Dict[int, List[int]] = {}
        self.labels: np.ndarray = np.array([])
    
    def add_event(self, event: GeoEvent):
        """Add a new geo event."""
        self.events.append(event)
    
    def fit(self) -> Dict[int, List[int]]:
        """Run ST-DBSCAN clustering on all events.
        
        Returns:
            Dictionary mapping cluster_id → list of event indices
        """
        n = len(self.events)
        if n == 0:
            return {}
        
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            
            neighbors = self._get_neighbors(i)
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Noise
                continue
            
            # Expand cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0
            
            while j < len(seed_set):
                q = seed_set[j]
                
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = self._get_neighbors(q)
                    
                    if len(q_neighbors) >= self.min_samples:
                        seed_set.extend([x for x in q_neighbors if x not in seed_set])
                
                if labels[q] == -1:
                    labels[q] = cluster_id
                
                j += 1
            
            cluster_id += 1
        
        self.labels = labels
        
        # Build cluster dictionary
        self.clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                self.clusters[label].append(idx)
        
        return dict(self.clusters)
    
    def _get_neighbors(self, idx: int) -> List[int]:
        """Find spatio-temporal neighbors of event idx."""
        event = self.events[idx]
        neighbors = []
        
        for j, other in enumerate(self.events):
            if j == idx:
                continue
            
            # Temporal check (fast)
            time_diff = abs(event.timestamp - other.timestamp)
            if time_diff > self.temporal_eps:
                continue
            
            # Spatial check
            dist = self._haversine_distance(
                event.latitude, event.longitude,
                other.latitude, other.longitude,
            )
            if dist <= self.spatial_eps:
                neighbors.append(j)
        
        return neighbors
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute distance between two points in km."""
        R = 6371.0  # Earth radius in km
        
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def get_cluster_info(self) -> List[Dict]:
        """Get summary information about detected clusters."""
        cluster_info = []
        
        for cluster_id, event_indices in self.clusters.items():
            events = [self.events[i] for i in event_indices]
            lats = [e.latitude for e in events]
            lons = [e.longitude for e in events]
            timestamps = [e.timestamp for e in events]
            risks = [e.risk_score for e in events]
            
            cluster_info.append({
                "cluster_id": cluster_id,
                "size": len(events),
                "center_lat": float(np.mean(lats)),
                "center_lon": float(np.mean(lons)),
                "radius_km": float(max(
                    self._haversine_distance(np.mean(lats), np.mean(lons), lat, lon)
                    for lat, lon in zip(lats, lons)
                )) if len(events) > 1 else 0.0,
                "time_span_hours": float((max(timestamps) - min(timestamps)) / 3600),
                "avg_risk": float(np.mean(risks)),
                "event_rate_per_hour": float(
                    len(events) / max(1, (max(timestamps) - min(timestamps)) / 3600)
                ),
            })
        
        return cluster_info
