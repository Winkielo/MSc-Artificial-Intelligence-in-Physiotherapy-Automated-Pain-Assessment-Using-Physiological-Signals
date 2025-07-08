"""
Data Preprocessing Module for AI Physiotherapy Pain Assessment System

This module implements the DataPreprocessor deliverable:
- Loads and aligns x.npy data with continuous pain labels from CSV files
- Uses timestamp-based mapping algorithm for synchronization
- Leverages subjects.npy for subject-specific analysis
- Handles NRS (PMCD) and CoVAS (PMED) pain scales

Author: Wing Kiu Lo
Date: July 2025
"""

from .data_loader import PainMonitDataLoader
from .data_synchronizer import DataSynchronizer
from .pmd_parser import PMDParser

__all__ = [
    'PainMonitDataLoader',
    'DataSynchronizer', 
    'PMDParser'
]

__version__ = "1.0.0"
__author__ = "Wing Kiu Lo" 