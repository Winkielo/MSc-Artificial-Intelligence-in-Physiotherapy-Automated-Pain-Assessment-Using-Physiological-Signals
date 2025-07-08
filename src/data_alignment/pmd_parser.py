"""
PainMonit Dataset (PMD) Parser Module

This module implements a comprehensive parser for the PainMonit Dataset that:
1. Parses raw CSV files from both PMED (experimental) and PMCD (clinical) datasets
2. Extracts physiological signals: BVP, EMG, EDA, RESP (excluding ECG per requirements)
3. Extracts pain ratings: NRS (0-10) for PMED, CoVAS (0-100) normalized to 0-10 for PMCD
4. Generates x.npy and subjects.npy files following the GitHub repository structure
5. Implements FR1.1: Hybrid Data Source Approach
6. Implements FR2.2: Required Signal Subset

Author: Wing Kiu Lo
Date: June 2025
"""

import numpy as np
import pandas as pd
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMDParser:
    """
    Comprehensive parser for PainMonit Dataset (PMD) implementing hybrid data source approach.
    
    Supports both PMED (experimental) and PMCD (clinical) datasets according to FR1.1 and FR2.2.
    """
    
    def __init__(self, pmd_root_path: str):
        """
        Initialize PMD parser.
        
        Args:
            pmd_root_path: Root path to PMD dataset directory
        """
        self.pmd_root_path = Path(pmd_root_path)
        self.pmed_raw_path = self.pmd_root_path / "PMHDB" / "raw-data"  # PMED experimental
        self.pmcd_raw_path = self.pmd_root_path / "PMPDB" / "raw-data"  # PMCD clinical
        
        # Signal column mappings according to FR2.2: Required Signal Subset
        # BVP, EMG, EDA, RESP (excluding ECG as per requirements)
        self.pmed_signal_columns = {
            'Bvp': 'BVP',          # Blood Volume Pulse
            'Emg': 'EMG',          # Electromyography  
            'Eda_E4': 'EDA',       # Electrodermal Activity (E4 sensor)
            'Resp': 'RESP'         # Respiration
        }
        
        self.pmcd_signal_columns = {
            'Bvp': 'BVP',          # Blood Volume Pulse
            'Emg': 'EMG',          # Electromyography
            'Eda_E4': 'EDA',       # Electrodermal Activity
            'Resp': 'RESP'         # Respiration
        }
        
        # Pain rating columns according to FR1.1
        self.pain_rating_columns = {
            'PMED': 'COVAS',       # Continuous Visual Analog Scale (0-100)
            'PMCD': 'Pain rates'   # Numerical Rating Scale (0-10)
        }
        
        logger.info("PMDParser initialized")
    
    def parse_pmed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse PMED (experimental) dataset from raw CSV files.
        
        Implements FR1.1: Ground Truth Label Source (raw CSV)
        Implements FR2.2: Required Signal Subset (BVP, EMG, EDA, RESP)
        
        Returns:
            Tuple of (x_data, y_data, subjects_data)
        """
        logger.info("Parsing PMED dataset...")
        
        # Find all PMED CSV files
        csv_files = list(self.pmed_raw_path.glob("S_*-synchronised-data.csv"))
        logger.info(f"Found {len(csv_files)} PMED CSV files")
        
        all_features = []
        all_labels = []
        all_subjects = []
        
        for csv_file in sorted(csv_files):
            try:
                # Extract subject ID from filename (S_01, S_02, etc.)
                subject_id = self._extract_subject_id(csv_file.name, 'PMED')
                
                # Load CSV with proper formatting
                df = pd.read_csv(csv_file, sep=';', decimal=',')
                
                # Extract physiological signals
                signal_data = self._extract_signals(df, 'PMED')
                
                # Extract pain ratings (COVAS: 0-100, normalize to 0-10)
                pain_data = self._extract_pain_ratings(df, 'PMED')
                
                if signal_data is not None and pain_data is not None:
                    # Ensure same length
                    min_len = min(len(signal_data), len(pain_data))
                    signal_data = signal_data[:min_len]
                    pain_data = pain_data[:min_len]
                    
                    # Create segments (10-second windows for PMED)
                    segments = self._create_segments(signal_data, pain_data, subject_id, 
                                                   window_size=2500, overlap=0.5)  # 10s @ 250Hz
                    
                    if segments:
                        for segment in segments:
                            all_features.append(segment['features'])
                            all_labels.append(segment['label'])
                            all_subjects.append(segment['subject'])
                
                logger.info(f"Processed PMED file: {csv_file.name} (Subject {subject_id})")
                
            except Exception as e:
                logger.warning(f"Error processing PMED file {csv_file}: {e}")
                continue
        
        # Convert to numpy arrays
        x_data = np.array(all_features) if all_features else np.empty((0, 0))
        y_data = np.array(all_labels) if all_labels else np.empty((0,))
        subjects_data = np.array(all_subjects) if all_subjects else np.empty((0,))
        
        logger.info(f"PMED parsing complete: {x_data.shape[0]} segments")
        return x_data, y_data, subjects_data
    
    def parse_pmcd_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse PMCD (clinical) dataset from raw CSV files.
        
        Implements FR1.1: Ground Truth Label Source (raw CSV)
        Implements FR2.2: Required Signal Subset (BVP, EMG, EDA, RESP)
        
        Returns:
            Tuple of (x_data, y_data, subjects_data)
        """
        logger.info("Parsing PMCD dataset...")
        
        # Find all PMCD patient directories
        patient_dirs = [d for d in self.pmcd_raw_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(patient_dirs)} PMCD patient directories")
        
        all_features = []
        all_labels = []
        all_subjects = []
        
        for patient_dir in sorted(patient_dirs):
            try:
                # Extract patient ID from directory name (P01_1, P01_2, etc.)
                patient_id = self._extract_subject_id(patient_dir.name, 'PMCD')
                
                # Find main CSV file in patient directory
                csv_files = list(patient_dir.glob("*.csv"))
                main_csv = [f for f in csv_files if not f.name.endswith('_runUp.csv')]
                
                if not main_csv:
                    logger.warning(f"No main CSV file found in {patient_dir}")
                    continue
                
                csv_file = main_csv[0]
                
                # Load CSV with proper formatting
                df = pd.read_csv(csv_file, sep=';', decimal=',')
                
                # Extract physiological signals
                signal_data = self._extract_signals(df, 'PMCD')
                
                # Extract pain ratings (Pain rates: 0-10, already normalized)
                pain_data = self._extract_pain_ratings(df, 'PMCD')
                
                if signal_data is not None and pain_data is not None:
                    # Ensure same length
                    min_len = min(len(signal_data), len(pain_data))
                    signal_data = signal_data[:min_len]
                    pain_data = pain_data[:min_len]
                    
                    # Create segments (4-second windows for PMCD)
                    segments = self._create_segments(signal_data, pain_data, patient_id, 
                                                   window_size=1000, overlap=0.5)  # 4s @ 250Hz
                    
                    if segments:
                        for segment in segments:
                            all_features.append(segment['features'])
                            all_labels.append(segment['label'])
                            all_subjects.append(segment['subject'])
                
                logger.info(f"Processed PMCD patient: {patient_dir.name} (Subject {patient_id})")
                
            except Exception as e:
                logger.warning(f"Error processing PMCD patient {patient_dir}: {e}")
                continue
        
        # Convert to numpy arrays
        x_data = np.array(all_features) if all_features else np.empty((0, 0))
        y_data = np.array(all_labels) if all_labels else np.empty((0,))
        subjects_data = np.array(all_subjects) if all_subjects else np.empty((0,))
        
        logger.info(f"PMCD parsing complete: {x_data.shape[0]} segments")
        return x_data, y_data, subjects_data
    
    def _extract_subject_id(self, filename: str, dataset_type: str) -> int:
        """Extract subject ID from filename."""
        if dataset_type == 'PMED':
            # S_01-synchronised-data.csv -> 1
            match = re.search(r'S_(\d+)', filename)
            return int(match.group(1)) if match else 0
        else:  # PMCD
            # P01_1 -> 1 (combine patient and session)
            match = re.search(r'P(\d+)_(\d+)', filename)
            if match:
                patient_num = int(match.group(1))
                session_num = int(match.group(2))
                return patient_num * 10 + session_num  # Unique ID
            return 0
    
    def _extract_signals(self, df: pd.DataFrame, dataset_type: str) -> Optional[np.ndarray]:
        """
        Extract physiological signals according to FR2.2: Required Signal Subset.
        
        Args:
            df: DataFrame with physiological data
            dataset_type: 'PMED' or 'PMCD'
            
        Returns:
            numpy array with signals [BVP, EMG, EDA, RESP] or None if extraction fails
        """
        try:
            if dataset_type == 'PMED':
                signal_columns = self.pmed_signal_columns
            else:
                signal_columns = self.pmcd_signal_columns
            
            extracted_signals = []
            
            for col_name, signal_type in signal_columns.items():
                if col_name in df.columns:
                    signal_values = df[col_name].values
                    # Handle NaN values
                    signal_values = np.nan_to_num(signal_values, nan=0.0)
                    extracted_signals.append(signal_values)
                else:
                    logger.warning(f"Signal column {col_name} not found in {dataset_type} data")
                    return None
            
            # Stack signals as columns
            if extracted_signals:
                return np.column_stack(extracted_signals)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting signals from {dataset_type}: {e}")
            return None
    
    def _extract_pain_ratings(self, df: pd.DataFrame, dataset_type: str) -> Optional[np.ndarray]:
        """
        Extract pain ratings according to FR1.1: Ground Truth Label Source.
        
        Args:
            df: DataFrame with pain rating data
            dataset_type: 'PMED' or 'PMCD'
            
        Returns:
            numpy array with pain ratings (0-10 scale) or None if extraction fails
        """
        try:
            pain_col = self.pain_rating_columns[dataset_type]
            
            if pain_col not in df.columns:
                logger.warning(f"Pain rating column {pain_col} not found in {dataset_type} data")
                return None
            
            pain_values = df[pain_col].values
            
            # Handle NaN values by interpolation or exclusion
            pain_values = pd.Series(pain_values).interpolate().fillna(0).values
            
            # Normalize COVAS (0-100) to NRS (0-10) for PMED
            if dataset_type == 'PMED':
                pain_values = pain_values / 10.0  # Convert 0-100 to 0-10
            
            # Ensure values are within 0-10 range
            pain_values = np.clip(pain_values, 0, 10)
            
            return pain_values
            
        except Exception as e:
            logger.error(f"Error extracting pain ratings from {dataset_type}: {e}")
            return None
    
    def _create_segments(self, signal_data: np.ndarray, pain_data: np.ndarray, 
                        subject_id: int, window_size: int, overlap: float) -> List[Dict]:
        """
        Create segments for machine learning according to project requirements.
        
        Args:
            signal_data: Physiological signal data
            pain_data: Pain rating data
            subject_id: Subject identifier
            window_size: Window size in samples
            overlap: Overlap ratio (0.0 to 1.0)
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        step_size = int(window_size * (1 - overlap))
        
        for start_idx in range(0, len(signal_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Extract signal window
            signal_window = signal_data[start_idx:end_idx]
            
            # Extract corresponding pain ratings
            pain_window = pain_data[start_idx:end_idx]
            
            # Use mean pain rating for the segment (following common practice)
            mean_pain_rating = np.mean(pain_window)
            
            # Skip segments with invalid pain ratings
            if np.isnan(mean_pain_rating) or mean_pain_rating < 0:
                continue
            
            # Calculate features from the signal window
            features = self._extract_features(signal_window)
            
            if features is not None:
                segments.append({
                    'features': features,
                    'label': mean_pain_rating,
                    'subject': subject_id
                })
        
        return segments
    
    def _extract_features(self, signal_window: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a signal window according to FR4.1 and FR4.4.
        
        Args:
            signal_window: Signal data window [samples, signals]
            
        Returns:
            Feature vector or None if extraction fails
        """
        try:
            features = []
            
            # For each signal (BVP, EMG, EDA, RESP)
            for signal_idx in range(signal_window.shape[1]):
                signal = signal_window[:, signal_idx]
                
                # Statistical features
                features.extend([
                    np.mean(signal),           # Mean
                    np.std(signal),            # Standard deviation
                    np.min(signal),            # Minimum
                    np.max(signal),            # Maximum
                    np.median(signal),         # Median
                    np.percentile(signal, 25), # 25th percentile
                    np.percentile(signal, 75), # 75th percentile
                ])
                
                # Temporal features
                if len(signal) > 1:
                    features.extend([
                        np.mean(np.diff(signal)),  # Mean derivative
                        np.std(np.diff(signal)),   # Std of derivative
                    ])
                else:
                    features.extend([0.0, 0.0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def save_npy_files(self, x_data: np.ndarray, y_data: np.ndarray, 
                      subjects_data: np.ndarray, dataset_type: str):
        """
        Save data in .npy format following GitHub repository structure.
        
        Args:
            x_data: Feature data
            y_data: Pain rating labels
            subjects_data: Subject identifiers
            dataset_type: 'PMED' or 'PMCD'
        """
        if dataset_type == 'PMED':
            output_dir = self.pmd_root_path / "PMED"
        else:
            output_dir = self.pmd_root_path / "PMCD"
        
        output_dir.mkdir(exist_ok=True)
        
        # Save files
        np.save(output_dir / "x.npy", x_data)
        np.save(output_dir / "y.npy", y_data)  # For compatibility
        np.save(output_dir / "subjects.npy", subjects_data)
        
        # Also save raw pain ratings as CSV for FR1.1 compliance
        pain_df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=len(y_data), freq='1s'),
            'Pain rates': y_data
        })
        pain_df.to_csv(output_dir / "pain_data.csv", index=False)
        
        logger.info(f"Saved {dataset_type} data to {output_dir}")
        logger.info(f"  - x.npy: {x_data.shape}")
        logger.info(f"  - y.npy: {y_data.shape}")
        logger.info(f"  - subjects.npy: {subjects_data.shape}")
        logger.info(f"  - pain_data.csv: {len(pain_df)} rows")


def main():
    """
    Main function to parse PMD dataset and generate .npy files.
    """
    # Initialize parser
    pmd_root = "data/raw/PMD"
    parser = PMDParser(pmd_root)
    
    try:
        # Parse PMED dataset
        print("=" * 60)
        print("Parsing PMED (Experimental) Dataset")
        print("=" * 60)
        pmed_x, pmed_y, pmed_subjects = parser.parse_pmed_data()
        parser.save_npy_files(pmed_x, pmed_y, pmed_subjects, 'PMED')
        
        # Parse PMCD dataset
        print("\n" + "=" * 60)
        print("Parsing PMCD (Clinical) Dataset")
        print("=" * 60)
        pmcd_x, pmcd_y, pmcd_subjects = parser.parse_pmcd_data()
        parser.save_npy_files(pmcd_x, pmcd_y, pmcd_subjects, 'PMCD')
        
        print("\n" + "=" * 60)
        print("PMD Dataset Parsing Complete!")
        print("=" * 60)
        print(f"PMED: {pmed_x.shape[0]} segments")
        print(f"PMCD: {pmcd_x.shape[0]} segments")
        print("\nFiles generated:")
        print("- data/raw/PMD/PMED/x.npy, y.npy, subjects.npy, pain_data.csv")
        print("- data/raw/PMD/PMCD/x.npy, y.npy, subjects.npy, pain_data.csv")
        
    except Exception as e:
        logger.error(f"Error during PMD parsing: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 