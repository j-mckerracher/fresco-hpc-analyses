"""
Feature engineering pipeline for FRESCO failure detection.

Orchestrates feature computation from telemetry data with proper
temporal alignment and no-leakage guarantees.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from .config import TELEMETRY_METRICS, PREDICTION_HORIZONS
from .feature_windows import WindowFeatureComputer
from .labeling import LabelExample
from .utils import (
    setup_logging,
    log_memory_usage,
    save_artifact,
    load_artifact,
    create_stable_hash,
    ProgressTracker,
    batch_process_with_progress
)


logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Main feature engineering pipeline.
    
    Combines telemetry data with labels to create training/test features
    while maintaining strict temporal boundaries.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True,
        n_jobs: int = -1
    ):
        self.window_computer = WindowFeatureComputer()
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        self.n_jobs = n_jobs if n_jobs > 0 else 1
        
        # Feature scalers (fit during training)
        self.feature_scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.feature_columns: Optional[List[str]] = None
    
    def create_features(
        self,
        telemetry_data: pd.DataFrame,
        labels: List[LabelExample],
        job_metadata: Optional[pd.DataFrame] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Create feature matrix from telemetry data and labels.
        
        Args:
            telemetry_data: Raw telemetry data
            labels: List of label examples with timestamps
            job_metadata: Optional job context data
            parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Creating features for {len(labels)} examples")
        
        # Group telemetry by job for efficient access
        telemetry_by_job = self._group_telemetry_by_job(telemetry_data)
        
        # Prepare job metadata lookup
        metadata_by_job = {}
        if job_metadata is not None:
            jm = job_metadata.copy()
            # Ensure we have a jid column to index on
            if 'jid' not in jm.columns:
                logger.warning("job_metadata missing 'jid' column; skipping metadata lookup")
            else:
                # Normalize jid to string for consistency
                jm['jid'] = jm['jid'].astype(str)
                # Drop duplicate rows per jid keeping the last occurrence deterministically
                # This guarantees a unique index before converting with orient='index'
                if jm['jid'].duplicated().any():
                    dup_count = int(jm['jid'].duplicated().sum())
                    logger.warning(f"job_metadata has {dup_count} duplicate jids; keeping last occurrence per jid")
                jm = jm.drop_duplicates(subset=['jid'], keep='last').set_index('jid')
                # As a final guard, coalesce any residual duplicates if present
                if not jm.index.is_unique:
                    logger.warning("job_metadata index still not unique after de-dup; coalescing by keeping last")
                    jm = jm[~jm.index.duplicated(keep='last')]
                # Convert to a mapping of jid -> metadata dict
                metadata_by_job = jm.to_dict(orient='index')
        
        # Process examples
        if parallel and self.n_jobs > 1 and len(labels) > 100:
            feature_records = self._create_features_parallel(
                labels, telemetry_by_job, metadata_by_job
            )
        else:
            feature_records = self._create_features_sequential(
                labels, telemetry_by_job, metadata_by_job
            )
        
        # Convert to DataFrame
        if not feature_records:
            logger.warning("No features were created successfully")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(feature_records)
        
        # Add basic validation
        self._validate_features(features_df)
        
        logger.info(f"Created {len(features_df)} feature records with {len(features_df.columns)} columns")
        return features_df
    
    def _group_telemetry_by_job(self, telemetry_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group telemetry data by job ID for efficient lookup"""
        
        if 'jid' not in telemetry_data.columns:
            logger.warning("No job ID column found in telemetry data")
            return {}
        
        logger.debug("Grouping telemetry data by job ID")
        telemetry_by_job = {}
        
        for jid, job_data in telemetry_data.groupby('jid'):
            # Sort by time and ensure time column is datetime
            job_data = job_data.copy()
            if 'time' in job_data.columns:
                job_data['time'] = pd.to_datetime(job_data['time'], utc=True)
                job_data = job_data.sort_values('time')
            telemetry_by_job[str(jid)] = job_data
        
        logger.info(f"Grouped telemetry data for {len(telemetry_by_job)} jobs")
        return telemetry_by_job
    
    def _create_features_sequential(
        self,
        labels: List[LabelExample],
        telemetry_by_job: Dict[str, pd.DataFrame],
        metadata_by_job: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Create features sequentially"""
        
        feature_records = []
        progress = ProgressTracker(len(labels), "Creating features")
        
        for label_example in labels:
            try:
                features = self._create_single_example_features(
                    label_example, telemetry_by_job, metadata_by_job
                )
                if features:
                    feature_records.append(features)
                
                progress.update()
                
            except Exception as e:
                logger.warning(f"Failed to create features for example {label_example.jid}: {e}")
                progress.update()
                continue
        
        return feature_records
    
    def _create_features_parallel(
        self,
        labels: List[LabelExample],
        telemetry_by_job: Dict[str, pd.DataFrame],
        metadata_by_job: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Create features in parallel"""
        
        # Worker function
        worker_func = partial(
            _create_features_worker,
            telemetry_by_job=telemetry_by_job,
            metadata_by_job=metadata_by_job
        )
        
        feature_records = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit batches for processing
            batch_size = max(1, len(labels) // (self.n_jobs * 2))
            batches = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
            
            future_to_batch = {
                executor.submit(worker_func, batch): batch 
                for batch in batches
            }
            
            progress = ProgressTracker(len(batches), "Creating features (parallel)")
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    feature_records.extend(batch_results)
                    progress.update()
                except Exception as e:
                    logger.error(f"Parallel feature creation failed: {e}")
                    progress.update()
        
        return feature_records
    
    def _create_single_example_features(
        self,
        label_example: LabelExample,
        telemetry_by_job: Dict[str, pd.DataFrame],
        metadata_by_job: Dict[str, Dict]
    ) -> Optional[Dict[str, Any]]:
        """Create features for a single label example"""
        
        jid = label_example.jid
        target_time = label_example.timestamp
        
        # Get telemetry data for this job
        if jid not in telemetry_by_job:
            logger.debug(f"No telemetry data found for job {jid}")
            return None
        
        job_telemetry = telemetry_by_job[jid]
        
        # Get job metadata
        job_meta = metadata_by_job.get(jid, {})
        
        # Add example info to metadata
        job_meta = job_meta.copy()
        job_meta.update({
            'jid': jid,
            'target_time': target_time,
            'horizon_minutes': label_example.horizon_minutes
        })
        
        # Compute window features
        features = self.window_computer.compute_features(
            job_telemetry, target_time, job_meta
        )
        
        # Add label information
        features.update({
            'jid': jid,
            'timestamp': target_time,
            'label': label_example.label,
            'horizon_minutes': label_example.horizon_minutes,
            'failure_time': label_example.failure_time,
            'lead_time_minutes': label_example.lead_time_minutes
        })
        
        return features
    
    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate feature matrix"""
        
        if features_df.empty:
            raise ValueError("Feature matrix is empty")
        
        # Check for required columns
        required_columns = ['jid', 'timestamp', 'label', 'horizon_minutes']
        missing_columns = set(required_columns) - set(features_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for infinite values
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        
        for col in numeric_columns:
            inf_count = np.isinf(features_df[col]).sum()
            if inf_count > 0:
                infinite_counts[col] = inf_count
        
        if infinite_counts:
            logger.warning(f"Found infinite values in columns: {infinite_counts}")
            # Replace infinite values with NaN
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log basic statistics
        logger.info(f"Feature matrix shape: {features_df.shape}")
        logger.info(f"Label distribution: {features_df['label'].value_counts().to_dict()}")
        
        # Check missing value rates
        missing_rates = features_df.isnull().sum() / len(features_df)
        high_missing = missing_rates[missing_rates > 0.5]
        
        if len(high_missing) > 0:
            logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")
    
    def fit_preprocessors(
        self,
        train_features: pd.DataFrame,
        scaler_type: str = "robust",
        feature_selection: bool = True,
        k_features: int = 200
    ) -> None:
        """
        Fit feature preprocessors on training data.
        
        Args:
            train_features: Training feature matrix
            scaler_type: Type of scaler ("standard", "robust", or None)
            feature_selection: Whether to perform feature selection
            k_features: Number of features to select
        """
        logger.info("Fitting feature preprocessors")
        
        # Separate features from metadata
        feature_columns = self._get_feature_columns(train_features)
        X_train = train_features[feature_columns].copy()
        y_train = train_features['label'].copy()
        
        # Handle missing values (fill with median)
        X_train = X_train.fillna(X_train.median())
        
        # Fit scaler
        if scaler_type == "standard":
            self.feature_scaler = StandardScaler()
        elif scaler_type == "robust":
            self.feature_scaler = RobustScaler()
        else:
            self.feature_scaler = None
        
        if self.feature_scaler:
            self.feature_scaler.fit(X_train)
            logger.info(f"Fitted {scaler_type} scaler")
        
        # Fit feature selector
        if feature_selection and len(feature_columns) > k_features:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,  # Good for mixed data types
                k=min(k_features, len(feature_columns))
            )
            
            X_scaled = self.feature_scaler.transform(X_train) if self.feature_scaler else X_train
            self.feature_selector.fit(X_scaled, y_train)
            
            selected_features = [
                feature_columns[i] for i in self.feature_selector.get_support(indices=True)
            ]
            
            logger.info(f"Selected {len(selected_features)} features out of {len(feature_columns)}")
            self.feature_columns = selected_features
        else:
            self.feature_columns = feature_columns
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessors.
        
        Args:
            features_df: Raw feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.feature_columns is None:
            logger.warning("Preprocessors not fitted. Call fit_preprocessors first.")
            return features_df
        
        result_df = features_df.copy()
        
        # Select and extract features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features during transform: {missing}")
        
        X = result_df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if not X.empty else 0)
        
        # Apply scaling
        if self.feature_scaler:
            X_scaled = self.feature_scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=available_features, index=X.index)
        
        # Apply feature selection
        if self.feature_selector:
            feature_mask = self.feature_selector.get_support()
            if len(feature_mask) == len(available_features):
                selected_columns = [col for col, selected in zip(available_features, feature_mask) if selected]
                X = X[selected_columns]
        
        # Update result DataFrame with transformed features
        for col in X.columns:
            result_df[col] = X[col]
        
        # Remove unused feature columns
        unused_feature_cols = set(self._get_feature_columns(result_df)) - set(X.columns)
        result_df = result_df.drop(columns=list(unused_feature_cols), errors='ignore')
        
        return result_df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata)"""
        
        metadata_columns = {
            'jid', 'timestamp', 'label', 'horizon_minutes', 
            'failure_time', 'lead_time_minutes'
        }
        
        feature_columns = [col for col in df.columns if col not in metadata_columns]
        return feature_columns
    
    def get_feature_importance(
        self,
        train_features: pd.DataFrame,
        method: str = "mutual_info"
    ) -> pd.DataFrame:
        """
        Calculate feature importance scores.
        
        Args:
            train_features: Training feature matrix
            method: Importance method ("mutual_info" or "f_score")
            
        Returns:
            DataFrame with feature importance scores
        """
        feature_columns = self._get_feature_columns(train_features)
        X = train_features[feature_columns].fillna(0)
        y = train_features['label']
        
        if method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=42)
        elif method == "f_score":
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_preprocessors(self, filepath: str) -> None:
        """Save fitted preprocessors to disk"""
        preprocessors = {
            'feature_scaler': self.feature_scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns
        }
        save_artifact(preprocessors, filepath)
        logger.info(f"Saved preprocessors to {filepath}")
    
    def load_preprocessors(self, filepath: str) -> None:
        """Load preprocessors from disk"""
        preprocessors = load_artifact(filepath)
        
        self.feature_scaler = preprocessors.get('feature_scaler')
        self.feature_selector = preprocessors.get('feature_selector')
        self.feature_columns = preprocessors.get('feature_columns')
        
        logger.info(f"Loaded preprocessors from {filepath}")


def _create_features_worker(
    label_batch: List[LabelExample],
    telemetry_by_job: Dict[str, pd.DataFrame],
    metadata_by_job: Dict[str, Dict]
) -> List[Dict[str, Any]]:
    """Worker function for parallel feature creation"""
    
    # Set up logging for worker
    logger = setup_logging()
    
    # Create window computer for this worker
    window_computer = WindowFeatureComputer()
    
    batch_results = []
    
    for label_example in label_batch:
        try:
            jid = label_example.jid
            target_time = label_example.timestamp
            
            # Get telemetry data
            if jid not in telemetry_by_job:
                continue
            
            job_telemetry = telemetry_by_job[jid]
            job_meta = metadata_by_job.get(jid, {})
            
            # Add example info
            job_meta = job_meta.copy()
            job_meta.update({
                'jid': jid,
                'target_time': target_time,
                'horizon_minutes': label_example.horizon_minutes
            })
            
            # Compute features
            features = window_computer.compute_features(
                job_telemetry, target_time, job_meta
            )
            
            # Add label info
            features.update({
                'jid': jid,
                'timestamp': target_time,
                'label': label_example.label,
                'horizon_minutes': label_example.horizon_minutes,
                'failure_time': label_example.failure_time,
                'lead_time_minutes': label_example.lead_time_minutes
            })
            
            batch_results.append(features)
            
        except Exception as e:
            logger.warning(f"Worker failed for example {label_example.jid}: {e}")
            continue
    
    return batch_results