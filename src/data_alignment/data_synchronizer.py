"""
Data Synchronizer Module for Automated Pain Assessment System

This module implements the critical path timestamp-based data synchronization as specified in FR3:
- FR3.1: Temporal boundary establishment for each window
- FR3.2: Pain rating query within specific time intervals
- FR3.3: Synchronized dataset creation
- FR3.4: Data window filtering for missing labels

Author: Wing Kiu Lo
Date: June 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataSynchronizer:
    """
    Data synchronizer for timestamp-based alignment of physiological data with pain ratings.
    
    Implements FR3: Timestamp-Based Data Synchronization (Critical Path)
    - FR3.1: Establish temporal boundaries for each physiological data window
    - FR3.2: Query pain ratings within specific time intervals
    - FR3.3: Create synchronized dataset by pairing windows with labels
    - FR3.4: Filter out windows without corresponding pain labels
    """
    
    def __init__(self, window_duration: int = 10):
        """
        Initialize the data synchronizer.
        
        Args:
            window_duration: Duration of each window in seconds (10s for PMED, 4s for PMCD)
        """
        self.window_duration = window_duration
        logger.info(f"DataSynchronizer initialized with {window_duration}s window duration")
    
    def establish_temporal_boundaries(self, 
                                   x_data: np.ndarray, 
                                   start_time: datetime,
                                   dataset_type: str) -> List[Dict]:
        """
        Establish temporal boundaries for each individual window of physiological data.
        
        Implements FR3.1: Temporal Boundary Establishment
        - Creates start and end time for each window in x.npy array
        
        Args:
            x_data: Physiological data from x.npy
            start_time: Starting time of the experiment
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            List of dictionaries containing window information with temporal boundaries
        """
        # Adjust window duration based on dataset type
        if dataset_type == 'PMED':
            window_duration = 10  # 10s windows for PMED
        else:
            window_duration = 4   # 4s windows for PMCD
        
        windows = []
        num_windows = x_data.shape[0]
        
        for i in range(num_windows):
            window_start = start_time + timedelta(seconds=i * window_duration)
            window_end = window_start + timedelta(seconds=window_duration)
            
            window_info = {
                'window_index': i,
                'start_time': window_start,
                'end_time': window_end,
                'duration': window_duration,
                'physiological_data': x_data[i, :]
            }
            windows.append(window_info)
        
        logger.info(f"Established temporal boundaries for {len(windows)} windows")
        return windows
    
    def query_pain_ratings(self, 
                          windows: List[Dict], 
                          pain_data: pd.DataFrame,
                          dataset_type: str) -> List[Dict]:
        """
        Query pain ratings within each window's specific time interval.
        
        Implements FR3.2: Pain Rating Query
        - For each physiological data window, query corresponding raw .csv file
        - Determine if continuous pain rating was recorded within window's time interval
        
        Args:
            windows: List of window dictionaries with temporal boundaries
            pain_data: DataFrame containing pain ratings with timestamps
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            List of windows with corresponding pain ratings
        """
        # Ensure pain_data has timestamp column
        if 'timestamp' not in pain_data.columns:
            # Try to find timestamp column
            timestamp_cols = [col for col in pain_data.columns if 'time' in col.lower()]
            if timestamp_cols:
                pain_data['timestamp'] = pd.to_datetime(pain_data[timestamp_cols[0]])
            else:
                # Create synthetic timestamps if none exist
                logger.warning("No timestamp column found, creating synthetic timestamps")
                pain_data['timestamp'] = pd.date_range(
                    start=datetime.now(), 
                    periods=len(pain_data), 
                    freq='1S'
                )
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(pain_data['timestamp']):
            pain_data['timestamp'] = pd.to_datetime(pain_data['timestamp'])
        
        synchronized_windows = []
        
        for window in windows:
            window_start = window['start_time']
            window_end = window['end_time']
            
            # Find pain ratings within this window's time interval
            mask = (pain_data['timestamp'] >= window_start) & (pain_data['timestamp'] < window_end)
            window_pain_ratings = pain_data[mask]
            
            if len(window_pain_ratings) > 0:
                # Calculate average pain rating for the window
                avg_pain_rating = window_pain_ratings['Pain rates'].mean()
                
                # Add pain rating to window info
                window['pain_rating'] = avg_pain_rating
                window['pain_rating_count'] = len(window_pain_ratings)
                window['has_pain_data'] = True
                
                synchronized_windows.append(window)
            else:
                # Mark window as having no pain data
                window['pain_rating'] = None
                window['pain_rating_count'] = 0
                window['has_pain_data'] = False
                
                # Still add to list for filtering later
                synchronized_windows.append(window)
        
        logger.info(f"Queried pain ratings for {len(synchronized_windows)} windows")
        return synchronized_windows
    
    def create_synchronized_dataset(self, 
                                  synchronized_windows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synchronized dataset by pairing physiological data windows with pain labels.
        
        Implements FR3.3: Synchronized Dataset Creation
        - Pairs each physiological data window with its corresponding continuous pain label
        
        Args:
            synchronized_windows: List of windows with pain ratings
            
        Returns:
            Tuple of (physiological_features, pain_labels)
        """
        # Filter windows that have pain data
        valid_windows = [w for w in synchronized_windows if w['has_pain_data']]
        
        if not valid_windows:
            raise ValueError("No windows with valid pain data found")
        
        # Extract physiological features and pain labels
        physiological_features = []
        pain_labels = []
        
        for window in valid_windows:
            physiological_features.append(window['physiological_data'])
            pain_labels.append(window['pain_rating'])
        
        # Convert to numpy arrays
        X = np.array(physiological_features)
        y = np.array(pain_labels)
        
        logger.info(f"Created synchronized dataset: X={X.shape}, y={y.shape}")
        return X, y
    
    def filter_data_windows(self, 
                           synchronized_windows: List[Dict]) -> List[Dict]:
        """
        Discard physiological data windows without corresponding pain labels.
        
        Implements FR3.4: Data Window Filtering
        - Discards any physiological data window for which no corresponding 
          continuous pain label can be found within its time interval
        
        Args:
            synchronized_windows: List of windows with pain ratings
            
        Returns:
            List of windows that have valid pain labels
        """
        # Filter out windows without pain data
        valid_windows = [w for w in synchronized_windows if w['has_pain_data']]
        
        discarded_count = len(synchronized_windows) - len(valid_windows)
        logger.info(f"Filtered data windows: {len(valid_windows)} valid, {discarded_count} discarded")
        
        return valid_windows
    
    def normalize_pain_scale(self, 
                           pain_labels: np.ndarray, 
                           dataset_type: str) -> np.ndarray:
        """
        Normalize pain scale to 0-10 range.
        
        Implements FR3.6: Pain Scale Normalization
        - Normalizes CoVAS (0-100) to 0-10 for consistency with NRS
        
        Args:
            pain_labels: Array of pain labels
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            Normalized pain labels in 0-10 range
        """
        if dataset_type == 'PMCD':
            # CoVAS is 0-100, normalize to 0-10
            normalized_labels = pain_labels / 10.0
            logger.info("Normalized CoVAS (0-100) to NRS (0-10) scale")
        else:
            # PMED already uses NRS (0-10)
            normalized_labels = pain_labels
            logger.info("PMED already uses NRS (0-10) scale")
        
        return normalized_labels
    
    def synchronize_dataset(self, 
                          x_data: np.ndarray,
                          pain_data: pd.DataFrame,
                          subjects_data: np.ndarray,
                          dataset_type: str,
                          start_time: datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data synchronization process.
        
        Implements the complete FR3: Timestamp-Based Data Synchronization (Critical Path)
        
        Args:
            x_data: Physiological data from x.npy
            pain_data: Ground truth pain ratings from CSV
            subjects_data: Subject identifiers from subjects.npy
            dataset_type: Either 'PMED' or 'PMCD'
            start_time: Starting time of the experiment
            
        Returns:
            Tuple of (synchronized_features, synchronized_labels, synchronized_subjects)
        """
        logger.info(f"Starting data synchronization for {dataset_type}")
        
        # FR3.1: Establish temporal boundaries
        windows = self.establish_temporal_boundaries(x_data, start_time, dataset_type)
        
        # FR3.2: Query pain ratings within time intervals
        synchronized_windows = self.query_pain_ratings(windows, pain_data, dataset_type)
        
        # FR3.4: Filter windows without pain labels
        valid_windows = self.filter_data_windows(synchronized_windows)
        
        # FR3.3: Create synchronized dataset
        X, y = self.create_synchronized_dataset(valid_windows)
        
        # Get corresponding subject identifiers
        valid_indices = [w['window_index'] for w in valid_windows]
        synchronized_subjects = subjects_data[valid_indices]
        
        # Normalize pain scale
        y_normalized = self.normalize_pain_scale(y, dataset_type)
        
        logger.info(f"Data synchronization completed for {dataset_type}:")
        logger.info(f"  - Features: {X.shape}")
        logger.info(f"  - Labels: {y_normalized.shape}")
        logger.info(f"  - Subjects: {synchronized_subjects.shape}")
        
        return X, y_normalized, synchronized_subjects


def main():
    """
    Example usage of the DataSynchronizer.
    """
    # Example data (replace with actual data loading)
    x_data = np.random.rand(100, 9)  # 100 windows, 9 features
    pain_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='5S'),
        'Pain rates': np.random.uniform(0, 10, 100)
    })
    subjects_data = np.random.randint(0, 10, 100)
    
    # Initialize synchronizer
    synchronizer = DataSynchronizer(window_duration=10)
    
    # Synchronize data
    start_time = datetime(2025, 1, 1)
    X_sync, y_sync, subjects_sync = synchronizer.synchronize_dataset(
        x_data, pain_data, subjects_data, 'PMED', start_time
    )
    
    print("Data synchronization completed successfully!")
    print(f"Synchronized dataset shape: {X_sync.shape}")


if __name__ == "__main__":
    main() 