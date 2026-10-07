"""
Geo-Temporal Pattern Analyzer — Signal 5.

This signal answers: "Where are scams happening? Is there a geographic burst?
How fast is the campaign spreading?"

Combines:
  - SpatialTemporalClusterer: Detect geographic hotspots
  - CampaignSpreadModel: Predict campaign trajectory
  - Time-series anomaly detection on call density
"""

import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from src.signals.base_signal import BaseSignalAnalyzer, SignalOutput
from src.signals.geo_temporal.spatial_clustering import SpatialTemporalClusterer, GeoEvent
from src.signals.geo_temporal.spread_model import CampaignSpreadModel


# Major Indian cities for the spread model
INDIAN_CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Bhopal": (23.2599, 77.4126),
}


class GeoTemporalAnalyzer(BaseSignalAnalyzer):
    """Signal 5: Geographical & Temporal Pattern Analyzer.
    
    Detects geographic clustering, temporal bursts, and campaign
    spread patterns from location and timing data.
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        
        # Spatial-temporal clustering
        self.clusterer = SpatialTemporalClusterer(
            spatial_eps=config.get("spatial_eps", 50.0),
            temporal_eps=config.get("temporal_eps", 3600.0),
            min_samples=config.get("min_cluster_size", 5),
        )
        
        # Campaign spread model (pre-loaded with Indian cities)
        self.spread_model = CampaignSpreadModel(
            beta=0.3, gamma=0.1, distance_kernel_scale=100.0,
        )
        for city, (lat, lon) in INDIAN_CITIES.items():
            self.spread_model.add_region(city, lat, lon)
        
        # Time-series buffer for anomaly detection
        self._event_times: List[float] = []
        self._event_locations: List[Tuple] = []
        
        # Embedding network
        self.embedding_net = torch.nn.Sequential(
            torch.nn.Linear(20, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, config.get("embedding_dim", 64)),
        ).to(self.device)
        
        self.embedding_net.eval()
        self._initialized = True
    
    def get_signal_name(self) -> str:
        return "geo_temporal"
    
    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 64)
    
    def get_decay_rate(self) -> float:
        return self.config.get("confidence_decay_rate", 0.1)
    
    def analyze(self, data: Any) -> SignalOutput:
        """Analyze geographic and temporal patterns.
        
        Args:
            data: Dict with:
                - 'latitude': float
                - 'longitude': float
                - 'timestamp': float
                - 'risk_score': float (from other signals)
                - 'region': str (optional — nearest city name)
                
        Returns:
            SignalOutput with geographic cluster and spread metrics
        """
        if not isinstance(data, dict):
            raise ValueError("GeoTemporalAnalyzer expects dict with location data")
        
        lat = data.get("latitude") if data.get("latitude") is not None else 28.6139
        lon = data.get("longitude") if data.get("longitude") is not None else 77.2090
        timestamp = data.get("timestamp") if data.get("timestamp") is not None else time.time()
        risk_score = data.get("risk_score") if data.get("risk_score") is not None else 0.5
        region = data.get("region") or self._nearest_city(lat, lon)
        
        # Step 1: Add event to clusterer
        event = GeoEvent(
            event_id=f"evt_{len(self.clusterer.events)}",
            latitude=lat, longitude=lon,
            timestamp=timestamp, risk_score=risk_score,
        )
        self.clusterer.add_event(event)
        self._event_times.append(timestamp)
        self._event_locations.append((lat, lon))
        
        # Step 2: Report to spread model
        self.spread_model.report_incident(region, timestamp)
        
        # Step 3: Run clustering (if enough data)
        clusters = {}
        cluster_info = []
        if len(self.clusterer.events) >= self.clusterer.min_samples:
            clusters = self.clusterer.fit()
            cluster_info = self.clusterer.get_cluster_info()
        
        # Step 4: Compute temporal anomaly (call density)
        temporal_anomaly = self._compute_temporal_anomaly(timestamp)
        
        # Step 5: Get spread velocity
        spread_velocity = self.spread_model.get_spread_velocity()
        
        # Step 6: Build feature vector for embedding
        feature_vector = self._build_feature_vector(
            lat, lon, timestamp, risk_score,
            len(clusters), temporal_anomaly, spread_velocity,
            cluster_info,
        )
        
        with torch.no_grad():
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            embedding = self.embedding_net(feature_tensor).squeeze(0)
        
        # Find which cluster this event belongs to
        event_cluster = -1
        if hasattr(self.clusterer, 'labels') and len(self.clusterer.labels) > 0:
            event_idx = len(self.clusterer.events) - 1
            if event_idx < len(self.clusterer.labels):
                event_cluster = int(self.clusterer.labels[event_idx])
        
        # Build metadata
        metadata = {
            "latitude": lat,
            "longitude": lon,
            "region": region,
            "event_cluster_id": event_cluster,
            "num_geo_clusters": len(clusters),
            "cluster_info": cluster_info[:5],  # Top 5 clusters
            "temporal_anomaly_score": round(temporal_anomaly, 4),
            "campaign_spread_velocity": round(spread_velocity, 4),
            "total_events_tracked": len(self.clusterer.events),
            "active_regions": sum(
                1 for r in self.spread_model.regions.values() if r.infected > 0
            ),
        }
        
        # Confidence: higher with more data and clearer patterns
        data_volume = min(1.0, len(self.clusterer.events) / 20.0)
        pattern_strength = max(temporal_anomaly, min(1.0, spread_velocity))
        confidence = 0.4 * data_volume + 0.6 * pattern_strength
        
        return self._create_output(
            embedding=embedding.cpu(),
            confidence=confidence,
            metadata=metadata,
        )
    
    def _compute_temporal_anomaly(self, current_time: float) -> float:
        """Detect anomalous burst in event frequency.
        
        Compares recent event rate to historical average.
        High anomaly = sudden spike in scam activity.
        """
        if len(self._event_times) < 5:
            return 0.0
        
        times = np.array(self._event_times)
        
        # Recent window (last hour) vs historical
        one_hour_ago = current_time - 3600
        recent_count = np.sum(times > one_hour_ago)
        
        if len(times) < 10:
            return min(1.0, recent_count / 5.0)
        
        # Historical rate (events per hour)
        total_span = max(1, (times[-1] - times[0]) / 3600)
        historical_rate = len(times) / total_span
        
        # Anomaly = how much recent rate exceeds historical
        recent_rate = recent_count  # events in last hour
        
        if historical_rate <= 0:
            return 0.0
        
        anomaly_ratio = recent_rate / max(1, historical_rate)
        anomaly_score = min(1.0, max(0.0, (anomaly_ratio - 1.0) / 2.0))
        
        return anomaly_score
    
    def _build_feature_vector(
        self, lat: float, lon: float, timestamp: float,
        risk_score: float, num_clusters: int, temporal_anomaly: float,
        spread_velocity: float, cluster_info: List[Dict],
    ) -> List[float]:
        """Build a fixed-size feature vector for the embedding network."""
        features = [
            lat / 90.0,           # Normalized latitude
            lon / 180.0,          # Normalized longitude
            risk_score,
            float(num_clusters),
            temporal_anomaly,
            spread_velocity,
            float(len(self._event_times)),
        ]
        
        # Cluster statistics
        if cluster_info:
            sizes = [c["size"] for c in cluster_info]
            risks = [c["avg_risk"] for c in cluster_info]
            features.extend([
                float(np.mean(sizes)),
                float(np.max(sizes)),
                float(np.mean(risks)),
                float(np.max(risks)),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Spread model state
        infected_regions = sum(
            1 for r in self.spread_model.regions.values() if r.infected > 0
        )
        total_infection = sum(
            r.infected for r in self.spread_model.regions.values()
        )
        features.extend([
            float(infected_regions),
            float(total_infection),
        ])
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _nearest_city(self, lat: float, lon: float) -> str:
        """Find the nearest major Indian city."""
        min_dist = float('inf')
        nearest = "Unknown"
        
        for city, (clat, clon) in INDIAN_CITIES.items():
            dist = np.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = city
        
        return nearest


# Type alias for import
Tuple = tuple
