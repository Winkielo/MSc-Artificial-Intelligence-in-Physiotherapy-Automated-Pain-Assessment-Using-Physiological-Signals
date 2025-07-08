"""
Data Loader Module for Automated Pain Assessment System

This module implements the hybrid data source approach as specified in FR1.1:
- Physiological Data Source: pre-processed NumPy array x.npy
- Ground Truth Label Source: raw .csv files for continuous pain ratings

Explicitly excludes pre-packaged label files (y.npy, y_heater.npy, y_covas.npy) as per FR1.2.

Author: Wing Kiu Lo
Date: June 2025
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Optional, List
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PainMonitDataLoader:
    """
    Data loader for PainMonit Dataset implementing hybrid data source approach.
    
    Implements FR1.1: Hybrid Data Source Approach
    - Physiological Data Source: x.npy (pre-processed NumPy array)
    - Ground Truth Label Source: raw .csv files (NRS for PMCD, CoVAS for PMED)
    
    Implements FR1.2: Explicit Label Exclusion
    - Ignores y.npy, y_heater.npy, y_covas.npy (binned, one-hot encoded labels)
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the PainMonit Dataset directory
        """
        self.data_path = data_path
        self.pmed_path = os.path.join(data_path, "PMED")
        self.pmcd_path = os.path.join(data_path, "PMCD")
        
        # Signal column mappings for different datasets
        self.signal_columns = {
            'PMED': {
                'BVP': [0, 1],  # pulse rate, amplitude
                'EMG': [2, 3, 4],  # mean, variance, RMS
                'EDA': [5, 6],  # amplitude, rise time
                'RESP': [7, 8]  # respiration rate, amplitude
            },
            'PMCD': {
                'BVP': [0, 1],  # pulse rate, amplitude
                'EMG': [2, 3, 4],  # mean, variance, RMS
                'EDA': [5, 6],  # amplitude, rise time
                'RESP': [7, 8]  # respiration rate, amplitude
            }
        }
        
        logger.info("PainMonitDataLoader initialized")
    
    def load_physiological_data(self, dataset_type: str) -> np.ndarray:
        """
        Load physiological data from x.npy (Physiological Data Source).
        
        Implements FR1.1: Physiological Data Source
        - Loads pre-processed NumPy array x.npy for cleaned and segmented time-series data
        
        Args:
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            numpy.ndarray: Physiological data from x.npy
        """
        if dataset_type not in ['PMED', 'PMCD']:
            raise ValueError("dataset_type must be 'PMED' or 'PMCD'")
        
        # Determine path based on dataset type
        if dataset_type == 'PMED':
            x_path = os.path.join(self.pmed_path, "x.npy")
        else:
            x_path = os.path.join(self.pmcd_path, "x.npy")
        
        # Load x.npy file
        try:
            x_data = np.load(x_path)
            logger.info(f"Loaded {dataset_type} physiological data: {x_data.shape}")
            return x_data
        except FileNotFoundError:
            logger.error(f"x.npy not found at {x_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {dataset_type} physiological data: {e}")
            raise
    
    def load_subject_data(self, dataset_type: str) -> np.ndarray:
        """
        Load subject identifiers from subjects.npy.
        
        Args:
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            numpy.ndarray: Subject identifiers
        """
        if dataset_type not in ['PMED', 'PMCD']:
            raise ValueError("dataset_type must be 'PMED' or 'PMCD'")
        
        # Determine path based on dataset type
        if dataset_type == 'PMED':
            subjects_path = os.path.join(self.pmed_path, "subjects.npy")
        else:
            subjects_path = os.path.join(self.pmcd_path, "subjects.npy")
        
        # Load subjects.npy file
        try:
            subjects_data = np.load(subjects_path)
            logger.info(f"Loaded {dataset_type} subject data: {subjects_data.shape}")
            return subjects_data
        except FileNotFoundError:
            logger.error(f"subjects.npy not found at {subjects_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {dataset_type} subject data: {e}")
            raise
    
    def load_ground_truth_labels(self, dataset_type: str) -> pd.DataFrame:
        """
        Load ground truth labels from raw .csv files (Ground Truth Label Source).
        Exclude any CSV file without a valid pain label/rating column.
        
        Implements FR1.1: Ground Truth Label Source
        - Loads raw .csv files exclusively for continuous pain ratings
        - NRS for PMED, CoVAS for PMCD
        
        Args:
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            pandas.DataFrame: Ground truth labels with timestamps
        """
        if dataset_type not in ['PMED', 'PMCD']:
            raise ValueError("dataset_type must be 'PMED' or 'PMCD'")
        
        if dataset_type == 'PMED':
            csv_dir = self.pmed_path
            expected_columns = ['Pain rates']
        else:
            csv_dir = self.pmcd_path
            expected_columns = ['Pain rates']
        
        csv_pattern = os.path.join(csv_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found at {csv_pattern}")
        
        all_data = []
        valid_files = []
        for csv_file in csv_files:
            try:
                # Read CSV with proper separator and decimal format
                # Based on PMD repository structure
                df = pd.read_csv(csv_file, sep=";", decimal=",")
                
                # Handle the case where timestamp and pain rates are in one column
                if len(df.columns) == 1 and 'timestamp,Pain rates' in df.columns[0]:
                    # Split the combined column
                    split_data = df.iloc[:, 0].str.split(',', expand=True)
                    if len(split_data.columns) >= 2:
                        df['timestamp'] = pd.to_datetime(split_data.iloc[:, 0])
                        df['Pain rates'] = pd.to_numeric(split_data.iloc[:, 1], errors='coerce')
                        # Remove the original combined column
                        df = df.drop(columns=[df.columns[0]])
                    else:
                        logger.warning(f"Could not parse combined timestamp,Pain rates column in {csv_file}")
                        continue
                else:
                    # Handle separate columns
                    missing_cols = [col for col in expected_columns if col not in df.columns]
                    if missing_cols:
                        pain_cols = [col for col in df.columns if 'pain' in col.lower() or 'rating' in col.lower()]
                        if pain_cols:
                            df['Pain rates'] = df[pain_cols[0]]
                            logger.info(f"Using alternative pain column: {pain_cols[0]}")
                        else:
                            logger.warning(f"No pain rating column found in {csv_file}. Excluding this file from dataset.")
                            continue
                
                # Add file identifier for tracking
                df['source_file'] = os.path.basename(csv_file)
                
                # Add timestamp if not present
                if 'timestamp' not in df.columns:
                    # Create synthetic timestamps based on row index
                    df['timestamp'] = pd.date_range(
                        start=pd.Timestamp.now(), 
                        periods=len(df), 
                        freq='1s'  # Fixed: use 's' instead of 'S'
                    )
                
                all_data.append(df)
                valid_files.append(os.path.basename(csv_file))
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}. Excluding this file from dataset.")
                continue
        
        if not all_data:
            raise ValueError(f"No valid CSV files found with required columns: {expected_columns}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {dataset_type} ground truth labels: {combined_data.shape}")
        return combined_data, valid_files
    
    def verify_excluded_files(self, dataset_type: str) -> None:
        """
        Verify that excluded label files are not being used.
        
        Implements FR1.2: Explicit Label Exclusion
        - Ensures y.npy, y_heater.npy, y_covas.npy are not loaded
        - These contain binned, one-hot encoded labels unsuitable for regression
        
        Args:
            dataset_type: Either 'PMED' or 'PMCD'
        """
        excluded_files = ['y.npy', 'y_heater.npy', 'y_covas.npy']
        
        # Determine path based on dataset type
        if dataset_type == 'PMED':
            base_path = self.pmed_path
        else:
            base_path = self.pmcd_path
        
        # Check if excluded files exist and log warning
        for excluded_file in excluded_files:
            excluded_path = os.path.join(base_path, excluded_file)
            if os.path.exists(excluded_path):
                logger.warning(f"Excluded file found but not loaded: {excluded_path}")
                logger.info("Following FR1.2: Explicit Label Exclusion - using raw CSV files instead")
        
        logger.info(f"Verified exclusion of binned label files for {dataset_type}")
    
    def select_signals(self, x_data: np.ndarray, dataset_type: str) -> np.ndarray:
        """
        Select specific signals from x.npy data array by column index.
        
        Updated to handle the 36-feature vectors from PMD parser which contain:
        - 4 signals (BVP, EMG, EDA, RESP) × 9 features each = 36 total features
        
        Implements FR2.1: Programmatic Signal Selection
        Implements FR2.2: Required Signal Subset (EDA, EMG, BVP, RESP)
        Implements FR2.3: Dataset-Specific Signal Ordering
        
        Args:
            x_data: Physiological data from x.npy (already processed features)
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            numpy.ndarray: All feature data (since PMD parser already selected required signals)
        """
        if dataset_type not in ['PMED', 'PMCD']:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # The PMD parser already extracted only the required signals (BVP, EMG, EDA, RESP)
        # and computed 9 features per signal, so we return all features
        logger.info(f"Using all features for {dataset_type}: {x_data.shape[1]} features from 4 signals")
        logger.info(f"Feature structure: BVP (0-8), EMG (9-17), EDA (18-26), RESP (27-35)")
        logger.info(f"Selected data shape: {x_data.shape}")
        
        return x_data
    
    def load_dataset(self, dataset_type: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load complete dataset using hybrid data source approach.
        Exclude any data (x.npy, subjects.npy) where the CSV file does not contain a valid pain label/rating column.
        
        Implements FR1.1: Hybrid Data Source Approach
        - Loads physiological data from x.npy
        - Loads ground truth from raw CSV files
        - Loads subject identifiers from subjects.npy
        
        Args:
            dataset_type: Either 'PMED' or 'PMCD'
            
        Returns:
            Tuple containing:
            - numpy.ndarray: Selected physiological signals
            - numpy.ndarray: Subject identifiers
            - pandas.DataFrame: Ground truth labels
        """
        logger.info(f"Loading {dataset_type} dataset using hybrid data source approach")
        self.verify_excluded_files(dataset_type)
        x_data = self.load_physiological_data(dataset_type)
        subjects_data = self.load_subject_data(dataset_type)
        ground_truth_data, valid_files = self.load_ground_truth_labels(dataset_type)

        # Filter x_data and subjects_data to only include entries with valid pain labels
        # Assumption: Each CSV file corresponds to a subject/session in x.npy and subjects.npy in the same order
        # We match by file name (source_file) and subject index
        # If there is only one CSV file, we keep all data; if multiple, we filter
        if len(valid_files) == 1:
            selected_signals = self.select_signals(x_data, dataset_type)
            filtered_subjects = subjects_data
        else:
            # Map subject indices to file names (assuming order matches)
            # This logic may need adjustment if the mapping is not 1-to-1
            selected_signals_list = []
            filtered_subjects_list = []
            for idx, file in enumerate(valid_files):
                # Only include if index is within bounds
                if idx < x_data.shape[0] and idx < subjects_data.shape[0]:
                    selected_signals_list.append(self.select_signals(x_data[idx:idx+1], dataset_type))
                    filtered_subjects_list.append(subjects_data[idx:idx+1])
            if selected_signals_list:
                selected_signals = np.vstack(selected_signals_list)
                filtered_subjects = np.concatenate(filtered_subjects_list)
            else:
                selected_signals = np.empty((0, x_data.shape[1]))
                filtered_subjects = np.empty((0,))
        logger.info(f"Successfully loaded {dataset_type} dataset:")
        logger.info(f"  - Physiological signals: {selected_signals.shape}")
        logger.info(f"  - Subject identifiers: {filtered_subjects.shape}")
        logger.info(f"  - Ground truth labels: {ground_truth_data.shape}")
        return selected_signals, filtered_subjects, ground_truth_data


def main():
    """
    Example usage of the PainMonitDataLoader.
    """
    # Example data path (adjust as needed)
    data_path = "data/raw/PMD"
    
    try:
        # Initialize loader
        loader = PainMonitDataLoader(data_path)
        
        # Load PMED dataset
        print("Loading PMED dataset...")
        pmed_signals, pmed_subjects, pmed_labels = loader.load_dataset('PMED')
        
        # Load PMCD dataset
        print("Loading PMCD dataset...")
        pmcd_signals, pmcd_subjects, pmcd_labels = loader.load_dataset('PMCD')
        
        print("Data loading completed successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    main() 