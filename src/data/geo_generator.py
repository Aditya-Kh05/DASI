"""
Geo-Temporal Event Generator — Synthetic Geographic & Temporal Pattern Data.

Generates realistic spatial-temporal event patterns for:
  - Coordinated scam campaigns (geographic clustering, temporal bursts)
  - Legitimate activity (random/uniform distribution)

These patterns feed the GeoTemporalAnalyzer's ST-DBSCAN and SIR spread model.
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GeoEvent:
    """A single geographic-temporal event."""
    event_id: str
    latitude: float
    longitude: float
    timestamp: float             # Unix timestamp
    region: str
    is_scam: bool
    campaign_id: Optional[int]   # None for legitimate events
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedGeoDataset:
    """A collection of geo-temporal events."""
    dataset_id: str
    events: List[GeoEvent]
    num_campaigns: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "dataset_id": self.dataset_id,
            "total_events": len(self.events),
            "num_scam": sum(1 for e in self.events if e.is_scam),
            "num_legitimate": sum(1 for e in self.events if not e.is_scam),
            "num_campaigns": self.num_campaigns,
            "regions": list(set(e.region for e in self.events)),
        }
    
    def get_feature_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert to tensors for the GeoTemporalAnalyzer."""
        coords = torch.tensor(
            [[e.latitude, e.longitude] for e in self.events],
            dtype=torch.float32,
        )
        timestamps = torch.tensor(
            [e.timestamp for e in self.events],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [1 if e.is_scam else 0 for e in self.events],
            dtype=torch.float32,
        )
        campaign_ids = torch.tensor(
            [e.campaign_id if e.campaign_id is not None else -1 for e in self.events],
            dtype=torch.long,
        )
        return {
            "coordinates": coords,
            "timestamps": timestamps,
            "labels": labels,
            "campaign_ids": campaign_ids,
        }


class GeoTemporalGenerator:
    """Generates synthetic geo-temporal event patterns.
    
    Scam campaign patterns:
      - Geographic clustering: Events concentrated around specific cities
      - Temporal bursts: Sudden spike in activity within a short window
      - Sequential spread: Campaign starts in one city and spreads to others
      - Recurring patterns: Same time-of-day activity (call center shifts)
    
    Legitimate patterns:
      - Uniform geographic distribution
      - Steady temporal distribution
      - No clustering
    """
    
    # Major Indian city coordinates
    CITY_COORDS = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.7041, 77.1025),
        "Bangalore": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707),
        "Hyderabad": (17.3850, 78.4867),
        "Kolkata": (22.5726, 88.3639),
        "Pune": (18.5204, 73.8567),
        "Ahmedabad": (23.0225, 72.5714),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462),
        "Chandigarh": (30.7333, 76.7794),
        "Kochi": (9.9312, 76.2673),
        "Indore": (22.7196, 75.8577),
        "Bhopal": (23.2599, 77.4126),
        "Nagpur": (21.1458, 79.0882),
    }
    
    # Scam call center operating windows (IST hours)
    SCAM_TIME_WINDOWS = [
        (9, 13),    # Morning shift
        (14, 18),   # Afternoon shift
        (19, 22),   # Evening shift (targeting people at home)
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._dataset_counter = 0
    
    def _generate_campaign_events(
        self,
        campaign_id: int,
        num_events: int,
        spread_pattern: str = "clustered",
        base_time: float = 0.0,
    ) -> List[GeoEvent]:
        """Generate events for a single scam campaign.
        
        Spread patterns:
          - "clustered": All events near one city
          - "spreading": Starts in one city, spreads to neighbors
          - "multi_city": Simultaneous activity in multiple cities
        """
        events = []
        
        cities = list(self.CITY_COORDS.keys())
        
        if spread_pattern == "clustered":
            # All events near one city
            center_city = self.rng.choice(cities)
            center_lat, center_lon = self.CITY_COORDS[center_city]
            
            for i in range(num_events):
                # Scatter within ~50km of center
                lat = center_lat + self.np_rng.normal(0, 0.15)
                lon = center_lon + self.np_rng.normal(0, 0.15)
                
                # Temporal: Concentrated within a few days, following shift patterns
                day_offset = self.np_rng.exponential(2.0)  # Days from campaign start
                shift = self.rng.choice(self.SCAM_TIME_WINDOWS)
                hour = self.np_rng.uniform(shift[0], shift[1])
                timestamp = base_time + day_offset * 86400 + hour * 3600
                
                events.append(GeoEvent(
                    event_id=f"scam_c{campaign_id}_e{i}",
                    latitude=lat, longitude=lon,
                    timestamp=timestamp,
                    region=center_city,
                    is_scam=True,
                    campaign_id=campaign_id,
                ))
        
        elif spread_pattern == "spreading":
            # Start in one city, spread to 2-3 others over time
            start_city = self.rng.choice(cities)
            target_cities = self.rng.sample(
                [c for c in cities if c != start_city],
                k=min(3, len(cities) - 1),
            )
            all_campaign_cities = [start_city] + target_cities
            
            events_per_city = max(1, num_events // len(all_campaign_cities))
            
            for city_idx, city in enumerate(all_campaign_cities):
                center_lat, center_lon = self.CITY_COORDS[city]
                # Later cities start later (spread delay)
                city_delay = city_idx * self.np_rng.uniform(1.0, 4.0)  # Days
                
                for i in range(events_per_city):
                    lat = center_lat + self.np_rng.normal(0, 0.12)
                    lon = center_lon + self.np_rng.normal(0, 0.12)
                    
                    day_offset = city_delay + self.np_rng.exponential(1.5)
                    shift = self.rng.choice(self.SCAM_TIME_WINDOWS)
                    hour = self.np_rng.uniform(shift[0], shift[1])
                    timestamp = base_time + day_offset * 86400 + hour * 3600
                    
                    events.append(GeoEvent(
                        event_id=f"scam_c{campaign_id}_e{len(events)}",
                        latitude=lat, longitude=lon,
                        timestamp=timestamp,
                        region=city,
                        is_scam=True,
                        campaign_id=campaign_id,
                    ))
        
        elif spread_pattern == "multi_city":
            # Simultaneous activity in multiple cities
            num_cities = self.rng.randint(2, 4)
            selected_cities = self.rng.sample(cities, k=num_cities)
            events_per_city = max(1, num_events // num_cities)
            
            for city in selected_cities:
                center_lat, center_lon = self.CITY_COORDS[city]
                
                for i in range(events_per_city):
                    lat = center_lat + self.np_rng.normal(0, 0.1)
                    lon = center_lon + self.np_rng.normal(0, 0.1)
                    
                    day_offset = self.np_rng.exponential(1.0)
                    shift = self.rng.choice(self.SCAM_TIME_WINDOWS)
                    hour = self.np_rng.uniform(shift[0], shift[1])
                    timestamp = base_time + day_offset * 86400 + hour * 3600
                    
                    events.append(GeoEvent(
                        event_id=f"scam_c{campaign_id}_e{len(events)}",
                        latitude=lat, longitude=lon,
                        timestamp=timestamp,
                        region=city,
                        is_scam=True,
                        campaign_id=campaign_id,
                    ))
        
        return events
    
    def _generate_legitimate_events(
        self,
        num_events: int,
        base_time: float = 0.0,
    ) -> List[GeoEvent]:
        """Generate uniformly distributed legitimate events."""
        events = []
        cities = list(self.CITY_COORDS.keys())
        
        for i in range(num_events):
            city = self.rng.choice(cities)
            center_lat, center_lon = self.CITY_COORDS[city]
            
            # Wider scatter than scam events
            lat = center_lat + self.np_rng.normal(0, 0.5)
            lon = center_lon + self.np_rng.normal(0, 0.5)
            
            # Uniform time distribution across all hours
            day_offset = self.np_rng.uniform(0, 30)  # Across 30 days
            hour = self.np_rng.uniform(6, 23)         # Waking hours
            timestamp = base_time + day_offset * 86400 + hour * 3600
            
            events.append(GeoEvent(
                event_id=f"legit_e{i}",
                latitude=lat, longitude=lon,
                timestamp=timestamp,
                region=city,
                is_scam=False,
                campaign_id=None,
            ))
        
        return events
    
    def generate_dataset(
        self,
        num_campaigns: int = 5,
        events_per_campaign: int = 20,
        num_legitimate: int = 200,
        base_time: float = 1696550400.0,  # Oct 6, 2023
    ) -> GeneratedGeoDataset:
        """Generate a complete geo-temporal dataset.
        
        Args:
            num_campaigns: Number of scam campaigns to simulate.
            events_per_campaign: Average events per campaign.
            num_legitimate: Number of legitimate background events.
            base_time: Base Unix timestamp for the dataset.
        """
        self._dataset_counter += 1
        dataset_id = f"geo_dataset_{self._dataset_counter:03d}"
        
        all_events = []
        
        # Generate scam campaigns
        spread_patterns = ["clustered", "spreading", "multi_city"]
        for campaign_id in range(num_campaigns):
            pattern = self.rng.choice(spread_patterns)
            n_events = max(5, int(events_per_campaign + self.np_rng.normal(0, 5)))
            campaign_start = base_time + self.np_rng.uniform(0, 15) * 86400
            
            campaign_events = self._generate_campaign_events(
                campaign_id=campaign_id,
                num_events=n_events,
                spread_pattern=pattern,
                base_time=campaign_start,
            )
            all_events.extend(campaign_events)
        
        # Generate legitimate events
        legit_events = self._generate_legitimate_events(num_legitimate, base_time)
        all_events.extend(legit_events)
        
        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        
        return GeneratedGeoDataset(
            dataset_id=dataset_id,
            events=all_events,
            num_campaigns=num_campaigns,
            metadata={
                "num_scam_events": len(all_events) - num_legitimate,
                "num_legitimate_events": num_legitimate,
                "time_span_days": 30,
            },
        )
