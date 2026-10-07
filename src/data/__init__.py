"""
Data Module — Synthetic Dataset Generation for DASI.

Since real digital arrest scam data is not publicly available,
this module generates realistic, labeled synthetic datasets
covering all 5 signal modalities.
"""

from src.data.dataset_generator import DatasetGenerator

__all__ = ["DatasetGenerator"]
