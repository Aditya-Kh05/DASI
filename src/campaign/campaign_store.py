"""
Campaign Store — Persistence layer for campaign state.

Provides in-memory storage with optional JSON file persistence,
enabling campaign state to survive process restarts.
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CampaignStore:
    """In-memory + JSON file persistence for campaign state.
    
    On save, writes all active campaigns and alerts to a JSON file.
    On load, restores state from the file.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or "data/campaign_state.json"
        self._campaigns: Dict[int, Dict] = {}
        self._alerts: List[Dict] = []
        self._metadata: Dict[str, Any] = {
            "created_at": time.time(),
            "last_saved_at": None,
            "save_count": 0,
        }
    
    def store_campaign(self, campaign_id: int, campaign_data: Dict):
        """Store or update a campaign."""
        self._campaigns[campaign_id] = campaign_data
    
    def get_campaign(self, campaign_id: int) -> Optional[Dict]:
        """Retrieve a campaign by ID."""
        return self._campaigns.get(campaign_id)
    
    def get_all_campaigns(self) -> Dict[int, Dict]:
        """Get all stored campaigns."""
        return dict(self._campaigns)
    
    def store_alert(self, alert_data: Dict):
        """Store an alert."""
        self._alerts.append(alert_data)
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent alerts."""
        return sorted(
            self._alerts,
            key=lambda a: a.get("timestamp", 0),
            reverse=True,
        )[:limit]
    
    def save(self):
        """Persist state to JSON file."""
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        
        state = {
            "campaigns": {
                str(k): v for k, v in self._campaigns.items()
            },
            "alerts": self._alerts[-1000:],  # Keep last 1000 alerts
            "metadata": {
                **self._metadata,
                "last_saved_at": time.time(),
                "save_count": self._metadata["save_count"] + 1,
            },
        }
        
        with open(self.persist_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        self._metadata = state["metadata"]
        logger.info(
            f"Campaign state saved: {len(self._campaigns)} campaigns, "
            f"{len(self._alerts)} alerts."
        )
    
    def load(self) -> bool:
        """Load state from JSON file.
        
        Returns:
            True if state was loaded, False if file not found.
        """
        if not os.path.exists(self.persist_path):
            return False
        
        try:
            with open(self.persist_path, "r") as f:
                state = json.load(f)
            
            self._campaigns = {
                int(k): v for k, v in state.get("campaigns", {}).items()
            }
            self._alerts = state.get("alerts", [])
            self._metadata = state.get("metadata", self._metadata)
            
            logger.info(
                f"Campaign state loaded: {len(self._campaigns)} campaigns, "
                f"{len(self._alerts)} alerts."
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load campaign state: {e}")
            return False
    
    def clear(self):
        """Clear all stored state."""
        self._campaigns.clear()
        self._alerts.clear()
        logger.info("Campaign store cleared.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_campaigns": len(self._campaigns),
            "total_alerts": len(self._alerts),
            "persist_path": self.persist_path,
            "metadata": self._metadata,
        }
