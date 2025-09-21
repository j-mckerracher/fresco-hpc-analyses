"""
Training/test dataset sampling with strict no-leakage guarantees.

Implements temporal splits and balanced sampling strategies while
ensuring no data leakage across time boundaries.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .config import SAMPLING_CONFIG, PREDICTION_HORIZONS
from .labeling import LabelExample, LabelGenerator
from .features import FeatureEngineering
from .utils import ProgressTracker, save_artifact, load_artifact


logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Container for dataset split information"""
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    test_features: pd.DataFrame
    split_info: Dict
    
    @property
    def train_size(self) -> int:
        return len(self.train_features)
    
    @property
    def val_size(self) -> int:
        return len(self.val_features)
    
    @property
    def test_size(self) -> int:
        return len(self.test_features)


class DatasetSampler:
    """
    Creates training/validation/test datasets with proper temporal splits
    and no data leakage guarantees.
    """
    
    def __init__(
        self,
        validation_split: float = SAMPLING_CONFIG["validation_split"],
        random_state: int = 42
    ):
        self.validation_split = validation_split
        self.random_state = random_state
        # self.label_generator = LabelGenerator()  # Will instantiate with correct horizons
        self.feature_engineer = FeatureEngineering()
    
    def create_temporal_splits(
        self,
        job_data: pd.DataFrame,
        telemetry_data: pd.DataFrame,
        horizons: List[int] = PREDICTION_HORIZONS,
        train_months: int = 12,
        val_months: int = 3,
        test_months: int = 3,
        min_examples_per_split: int = 100
    ) -> Dict[int, DatasetSplit]:
        """
        Create temporal train/val/test splits for each horizon.
        
        Args:
            job_data: Job metadata with timestamps and exit codes
            telemetry_data: Raw telemetry time series
            horizons: Prediction horizons in minutes
            train_months: Number of months for training data
            val_months: Number of months for validation data  
            test_months: Number of months for test data
            min_examples_per_split: Minimum examples required per split
            
        Returns:
            Dictionary mapping horizon -> DatasetSplit
        """
        logger.info("Creating temporal dataset splits")
        # Instantiate label generator with correct horizons
        self.label_generator = LabelGenerator(horizons=horizons)
        
        # Determine date ranges for splits
        split_dates = self._determine_split_dates(
            job_data, train_months, val_months, test_months
        )
        
        logger.info(f"Split dates: train={split_dates['train']}, "
                   f"val={split_dates['val']}, test={split_dates['test']}")
        
        dataset_splits = {}
        
        for horizon in horizons:
            logger.info(f"Creating splits for {horizon}-minute horizon")
            
            split = self._create_horizon_split(
                job_data, telemetry_data, horizon, split_dates, min_examples_per_split
            )
            
            if split:
                dataset_splits[horizon] = split
                logger.info(f"Horizon {horizon}: train={split.train_size}, "
                           f"val={split.val_size}, test={split.test_size}")
            else:
                logger.warning(f"Failed to create valid split for horizon {horizon}")
        
        return dataset_splits
    
    def _determine_split_dates(
        self,
        job_data: pd.DataFrame,
        train_months: int,
        val_months: int,
        test_months: int
    ) -> Dict[str, Tuple[date, date]]:
        """Determine date ranges for train/val/test splits"""
        
        # Get date range from job data
        job_data['end_date'] = pd.to_datetime(job_data['end_time'], utc=True).dt.date
        min_date = job_data['end_date'].min()
        max_date = job_data['end_date'].max()
        
        logger.info(f"Data date range: {min_date} to {max_date}")
        
        # Work backwards from max date to ensure we use most recent data for testing
        test_end = max_date
        test_start = test_end - timedelta(days=test_months * 30)
        
        val_end = test_start - timedelta(days=1)
        val_start = val_end - timedelta(days=val_months * 30)
        
        train_end = val_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_months * 30)
        
        # Ensure train_start is not before data availability
        if train_start < min_date:
            train_start = min_date
            logger.warning(f"Adjusted train start to data availability: {train_start}")
        
        return {
            'train': (train_start, train_end),
            'val': (val_start, val_end),
            'test': (test_start, test_end)
        }
    
    def _create_horizon_split(
        self,
        job_data: pd.DataFrame,
        telemetry_data: pd.DataFrame,
        horizon: int,
        split_dates: Dict,
        min_examples_per_split: int
    ) -> Optional[DatasetSplit]:
        """Create dataset split for a single horizon"""
        
        # Filter job data by split dates
        job_data = job_data.copy()
        job_data['end_date'] = pd.to_datetime(job_data['end_time'], utc=True).dt.date
        
        train_jobs = job_data[
            (job_data['end_date'] >= split_dates['train'][0]) &
            (job_data['end_date'] <= split_dates['train'][1])
        ]
        
        val_jobs = job_data[
            (job_data['end_date'] >= split_dates['val'][0]) &
            (job_data['end_date'] <= split_dates['val'][1])
        ]
        
        test_jobs = job_data[
            (job_data['end_date'] >= split_dates['test'][0]) &
            (job_data['end_date'] <= split_dates['test'][1])
        ]
        
        logger.debug(f"Horizon {horizon}: {len(train_jobs)} train jobs, "
                    f"{len(val_jobs)} val jobs, {len(test_jobs)} test jobs")
        
        # Generate labels for each split
        train_labels = self.label_generator.generate_labels(train_jobs, deduplicate=True)[horizon]
        val_labels = self.label_generator.generate_labels(val_jobs, deduplicate=True)[horizon]
        test_labels = self.label_generator.generate_labels(test_jobs, deduplicate=True)[horizon]
        
        # Check minimum example requirements
        if (len(train_labels) < min_examples_per_split or 
            len(val_labels) < min_examples_per_split or 
            len(test_labels) < min_examples_per_split):
            logger.warning(f"Insufficient examples for horizon {horizon}: "
                          f"train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")
            return None
        
        # Create features for each split
        train_features = self.feature_engineer.create_features(
            telemetry_data, train_labels, train_jobs
        )
        
        val_features = self.feature_engineer.create_features(
            telemetry_data, val_labels, val_jobs
        )
        
        test_features = self.feature_engineer.create_features(
            telemetry_data, test_labels, test_jobs
        )
        
        # Check for empty feature sets
        if train_features.empty or val_features.empty or test_features.empty:
            logger.warning(f"Empty features for horizon {horizon}")
            return None
        
        # Create split info
        split_info = {
            'horizon': horizon,
            'split_dates': split_dates,
            'train_label_dist': pd.Series([ex.label for ex in train_labels]).value_counts().to_dict(),
            'val_label_dist': pd.Series([ex.label for ex in val_labels]).value_counts().to_dict(),
            'test_label_dist': pd.Series([ex.label for ex in test_labels]).value_counts().to_dict(),
            'feature_count': len(self.feature_engineer._get_feature_columns(train_features))
        }
        
        return DatasetSplit(
            train_features=train_features,
            val_features=val_features, 
            test_features=test_features,
            split_info=split_info
        )
    
    def create_cross_cluster_splits(
        self,
        job_data: pd.DataFrame,
        telemetry_data: pd.DataFrame,
        train_clusters: List[str],
        test_cluster: str,
        horizons: List[int] = PREDICTION_HORIZONS
    ) -> Dict[int, DatasetSplit]:
        """
        Create cross-cluster generalization splits.
        
        Args:
            job_data: Job metadata
            telemetry_data: Raw telemetry data
            train_clusters: Clusters to use for training
            test_cluster: Cluster to use for testing
            horizons: Prediction horizons
            
        Returns:
            Dictionary mapping horizon -> DatasetSplit
        """
        logger.info(f"Creating cross-cluster splits: train={train_clusters}, test={test_cluster}")
        
        # Filter data by clusters
        if 'cluster' not in job_data.columns and 'cluster' not in telemetry_data.columns:
            logger.error("Cluster information not available in data")
            return {}
        
        train_job_data = job_data[job_data['cluster'].isin(train_clusters)]
        test_job_data = job_data[job_data['cluster'] == test_cluster]
        
        train_telemetry = telemetry_data[telemetry_data['cluster'].isin(train_clusters)]
        test_telemetry = telemetry_data[telemetry_data['cluster'] == test_cluster]
        
        # Split training data into train/val (temporal)
        train_split_date = train_job_data['end_time'].quantile(0.8)  # Use 80% for training
        
        actual_train_jobs = train_job_data[train_job_data['end_time'] <= train_split_date]
        val_jobs = train_job_data[train_job_data['end_time'] > train_split_date]
        
        dataset_splits = {}
        
        for horizon in horizons:
            # Generate labels
            train_labels = self.label_generator.generate_labels(actual_train_jobs)[horizon]
            val_labels = self.label_generator.generate_labels(val_jobs)[horizon]
            test_labels = self.label_generator.generate_labels(test_job_data)[horizon]
            
            if not all([train_labels, val_labels, test_labels]):
                logger.warning(f"Insufficient labels for cross-cluster split, horizon {horizon}")
                continue
            
            # Create features
            train_features = self.feature_engineer.create_features(
                train_telemetry, train_labels, actual_train_jobs
            )
            val_features = self.feature_engineer.create_features(
                train_telemetry, val_labels, val_jobs
            )
            test_features = self.feature_engineer.create_features(
                test_telemetry, test_labels, test_job_data
            )
            
            if any(df.empty for df in [train_features, val_features, test_features]):
                logger.warning(f"Empty features for cross-cluster split, horizon {horizon}")
                continue
            
            split_info = {
                'horizon': horizon,
                'train_clusters': train_clusters,
                'test_cluster': test_cluster,
                'split_type': 'cross_cluster'
            }
            
            dataset_splits[horizon] = DatasetSplit(
                train_features=train_features,
                val_features=val_features,
                test_features=test_features,
                split_info=split_info
            )
        
        return dataset_splits
    
    def balance_dataset(
        self,
        features_df: pd.DataFrame,
        strategy: str = "undersample",
        target_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Balance dataset for class imbalance.
        
        Args:
            features_df: Feature matrix with labels
            strategy: Balancing strategy ("undersample", "oversample", "class_weight")
            target_ratio: Target positive/negative ratio (None = 1:1)
            
        Returns:
            Balanced feature matrix
        """
        if 'label' not in features_df.columns:
            return features_df
        
        positive_count = (features_df['label'] == 1).sum()
        negative_count = (features_df['label'] == 0).sum()
        
        logger.info(f"Original class distribution: {negative_count} negative, {positive_count} positive")
        
        if positive_count == 0 or negative_count == 0:
            logger.warning("Cannot balance dataset with only one class")
            return features_df
        
        if strategy == "undersample":
            # Undersample majority class
            target_pos = positive_count
            target_neg = int(target_pos * (1 / (target_ratio or 1.0)))
            
            positive_examples = features_df[features_df['label'] == 1]
            negative_examples = features_df[features_df['label'] == 0]
            
            if len(negative_examples) > target_neg:
                negative_examples = negative_examples.sample(
                    n=target_neg, random_state=self.random_state
                )
            
            balanced_df = pd.concat([positive_examples, negative_examples], ignore_index=True)
            
        elif strategy == "oversample":
            # Simple duplication-based oversampling
            target_neg = negative_count
            target_pos = int(target_neg * (target_ratio or 1.0))
            
            positive_examples = features_df[features_df['label'] == 1]
            negative_examples = features_df[features_df['label'] == 0]
            
            if len(positive_examples) < target_pos:
                # Repeat positive examples to reach target
                repeat_factor = target_pos // len(positive_examples)
                remainder = target_pos % len(positive_examples)
                
                repeated_pos = pd.concat(
                    [positive_examples] * repeat_factor + 
                    [positive_examples.sample(n=remainder, random_state=self.random_state)]
                    if remainder > 0 else [positive_examples] * repeat_factor,
                    ignore_index=True
                )
                positive_examples = repeated_pos
            
            balanced_df = pd.concat([positive_examples, negative_examples], ignore_index=True)
            
        else:  # class_weight - return original with weight info
            balanced_df = features_df.copy()
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=features_df['label']
            )
            
            balanced_df['class_weight'] = balanced_df['label'].map({
                0: class_weights[0],
                1: class_weights[1]
            })
        
        new_positive = (balanced_df['label'] == 1).sum()
        new_negative = (balanced_df['label'] == 0).sum()
        
        logger.info(f"Balanced class distribution: {new_negative} negative, {new_positive} positive")
        
        return balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def create_stratified_splits(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified random splits (when temporal splits not appropriate).
        
        Args:
            features_df: Feature matrix
            test_size: Fraction for test set
            val_size: Fraction of remaining data for validation
            stratify_cols: Columns to stratify on
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Creating stratified random splits")
        
        if stratify_cols is None:
            stratify_cols = ['label']
        
        # Create stratification key
        if len(stratify_cols) == 1:
            stratify_key = features_df[stratify_cols[0]]
        else:
            stratify_key = features_df[stratify_cols].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
        
        # First split: separate test set
        train_val_idx, test_idx = next(StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_state
        ).split(features_df, stratify_key))
        
        train_val_df = features_df.iloc[train_val_idx]
        test_df = features_df.iloc[test_idx]
        
        # Second split: separate train and validation
        if val_size > 0:
            train_val_stratify = stratify_key.iloc[train_val_idx]
            
            train_idx, val_idx = next(StratifiedShuffleSplit(
                n_splits=1, test_size=val_size, random_state=self.random_state
            ).split(train_val_df, train_val_stratify))
            
            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_dataset_splits(
        self,
        splits: Dict[int, DatasetSplit],
        output_dir: str,
        prefix: str = "dataset"
    ) -> None:
        """Save dataset splits to disk"""
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for horizon, split in splits.items():
            horizon_dir = output_path / f"{prefix}_h{horizon}"
            horizon_dir.mkdir(exist_ok=True)
            
            # Save feature matrices
            save_artifact(split.train_features, horizon_dir / "train_features.parquet")
            save_artifact(split.val_features, horizon_dir / "val_features.parquet")
            save_artifact(split.test_features, horizon_dir / "test_features.parquet")
            
            # Save split info
            save_artifact(split.split_info, horizon_dir / "split_info.json")
        
        logger.info(f"Saved dataset splits to {output_dir}")
    
    def load_dataset_splits(
        self,
        input_dir: str,
        horizons: List[int],
        prefix: str = "dataset"
    ) -> Dict[int, DatasetSplit]:
        """Load dataset splits from disk"""
        
        from pathlib import Path
        input_path = Path(input_dir)
        
        splits = {}
        
        for horizon in horizons:
            horizon_dir = input_path / f"{prefix}_h{horizon}"
            
            if not horizon_dir.exists():
                logger.warning(f"Dataset split not found for horizon {horizon}")
                continue
            
            try:
                train_features = load_artifact(horizon_dir / "train_features.parquet")
                val_features = load_artifact(horizon_dir / "val_features.parquet")
                test_features = load_artifact(horizon_dir / "test_features.parquet")
                split_info = load_artifact(horizon_dir / "split_info.json")
                
                splits[horizon] = DatasetSplit(
                    train_features=train_features,
                    val_features=val_features,
                    test_features=test_features,
                    split_info=split_info
                )
                
                logger.info(f"Loaded dataset split for horizon {horizon}")
                
            except Exception as e:
                logger.error(f"Failed to load dataset split for horizon {horizon}: {e}")
        
        return splits