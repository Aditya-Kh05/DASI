"""
Campaign Spread Model — Predict how a scam campaign will spread geographically.

Uses an SIR (Susceptible-Infected-Recovered) epidemic model adapted for
scam campaign propagation. The analogy is:
  - Susceptible: Regions not yet targeted
  - Infected: Regions currently under active scam campaign
  - Recovered: Regions where campaign has been detected and blocked

This allows forecasting which regions will be targeted next.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RegionState:
    """State of a geographic region in the spread model."""
    region_id: str
    latitude: float
    longitude: float
    population: float = 1000000.0
    
    susceptible: float = 1.0      # Fraction susceptible
    infected: float = 0.0         # Fraction under attack
    recovered: float = 0.0        # Fraction with awareness/protection
    
    reported_incidents: int = 0
    first_incident_time: float = 0.0
    last_incident_time: float = 0.0


class CampaignSpreadModel:
    """SIR-based model for predicting scam campaign geographic spread.
    
    Parameters:
        beta: Transmission rate (how fast scam spreads to new regions)
        gamma: Recovery rate (how fast awareness/blocking takes effect)
        distance_kernel_scale: How distance affects transmission
    """
    
    def __init__(self, beta: float = 0.3, gamma: float = 0.1,
                 distance_kernel_scale: float = 100.0):
        self.beta = beta
        self.gamma = gamma
        self.distance_kernel_scale = distance_kernel_scale
        
        self.regions: Dict[str, RegionState] = {}
        self.history: List[Dict[str, Dict[str, float]]] = []
    
    def add_region(self, region_id: str, lat: float, lon: float,
                   population: float = 1000000.0):
        """Register a geographic region."""
        self.regions[region_id] = RegionState(
            region_id=region_id,
            latitude=lat, longitude=lon,
            population=population,
        )
    
    def report_incident(self, region_id: str, timestamp: float):
        """Report a scam incident in a region."""
        if region_id not in self.regions:
            return
        
        region = self.regions[region_id]
        region.reported_incidents += 1
        
        if region.infected == 0.0:
            region.first_incident_time = timestamp
            region.infected = 0.01  # Initial infection seed
            region.susceptible = 0.99
        else:
            region.infected = min(1.0, region.infected + 0.01)
            region.susceptible = max(0.0, 1.0 - region.infected - region.recovered)
        
        region.last_incident_time = timestamp
    
    def simulate_step(self, dt: float = 1.0) -> Dict[str, Dict[str, float]]:
        """Simulate one time step of campaign spread.
        
        Args:
            dt: Time step size (in hours)
            
        Returns:
            Updated states for all regions
        """
        updates = {}
        region_list = list(self.regions.values())
        
        for region in region_list:
            # Compute infection pressure from neighboring infected regions
            infection_pressure = 0.0
            
            for other in region_list:
                if other.region_id == region.region_id:
                    continue
                if other.infected <= 0:
                    continue
                
                # Distance-based decay
                dist = self._haversine(
                    region.latitude, region.longitude,
                    other.latitude, other.longitude,
                )
                distance_factor = np.exp(-dist / self.distance_kernel_scale)
                infection_pressure += other.infected * distance_factor
            
            # SIR update
            new_infections = self.beta * region.susceptible * (region.infected + infection_pressure * 0.1) * dt
            new_recoveries = self.gamma * region.infected * dt
            
            region.susceptible = max(0.0, region.susceptible - new_infections)
            region.infected = max(0.0, region.infected + new_infections - new_recoveries)
            region.recovered = min(1.0, region.recovered + new_recoveries)
            
            updates[region.region_id] = {
                "susceptible": round(region.susceptible, 4),
                "infected": round(region.infected, 4),
                "recovered": round(region.recovered, 4),
                "risk_level": round(region.infected + infection_pressure * 0.05, 4),
            }
        
        self.history.append(updates)
        return updates
    
    def forecast(self, steps: int = 24) -> List[Dict]:
        """Forecast campaign spread for the next N hours.
        
        Returns list of regional risk predictions over time.
        """
        # Save current state
        saved_states = {}
        for rid, region in self.regions.items():
            saved_states[rid] = (region.susceptible, region.infected, region.recovered)
        
        forecasts = []
        for step in range(steps):
            result = self.simulate_step(dt=1.0)
            forecasts.append(result)
        
        # Restore state (forecast shouldn't mutate)
        for rid, (s, i, r) in saved_states.items():
            self.regions[rid].susceptible = s
            self.regions[rid].infected = i
            self.regions[rid].recovered = r
        
        return forecasts
    
    def get_spread_velocity(self) -> float:
        """Compute how fast the campaign is spreading to new regions."""
        infected_regions = [r for r in self.regions.values() if r.infected > 0]
        
        if len(infected_regions) < 2:
            return 0.0
        
        # Sort by first incident time
        sorted_regions = sorted(infected_regions, key=lambda r: r.first_incident_time or 0.0)
        
        # Compute average time between new region infections
        time_diffs = []
        for i in range(1, len(sorted_regions)):
            t_curr = sorted_regions[i].first_incident_time or 0.0
            t_prev = sorted_regions[i - 1].first_incident_time or 0.0
            diff = t_curr - t_prev
            if diff > 0:
                time_diffs.append(diff)
        
        if not time_diffs:
            return 0.0
        
        # Velocity = regions per hour
        avg_time_between = np.mean(time_diffs)
        velocity = 3600.0 / avg_time_between if avg_time_between > 0 else 0.0
        
        return float(velocity)
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
