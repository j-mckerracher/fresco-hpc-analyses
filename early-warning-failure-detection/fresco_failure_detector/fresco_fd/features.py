"""
Optimized Feature Engineering Module for FRESCO Failure Detection
Performance improvements:
1. Vectorized operations instead of loops
2. Parallel processing support 
3. Caching of intermediate results
4. Efficient memory management
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import hashlib

from .labeling import LabelExample

# Performance monitoring decorator
def timed(label: str):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get('FFD_FEATURE_TRACE') == '1':
                t0 = time.perf_counter()
                logger = logging.getLogger("fresco_fd")
                logger.info(f"START | {label}")
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    dt = time.perf_counter() - t0
                    logger.info(f"END | {label} | took_s={dt:.3f}")
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

class OptimizedFeatureEngineering:
    """Optimized feature engineering with performance improvements"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger("fresco_fd")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Feature computation cache
        self._feature_cache = {}
        self._enable_cache = os.environ.get('FFD_ENABLE_CACHE', '1') == '1'
        
    @timed("group_telemetry_by_job")
    def _group_telemetry_by_job(self, telemetry_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Efficiently group telemetry by job ID"""
        # Use dictionary comprehension with pre-filtering
        if 'jid' not in telemetry_data.columns:
            return {}
            
        # Sort once for better cache locality
        telemetry_sorted = telemetry_data.sort_values(['jid', 'time'])
        
        # Use groupby object more efficiently
        grouped = telemetry_sorted.groupby('jid', sort=False)
        return {str(jid): group for jid, group in grouped}
    
    @timed("compute_statistical_features") 
    def _compute_statistical_features_vectorized(
        self, 
        window_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute statistical features using vectorized operations"""
        features = {}
        
        if window_data.empty:
            return self._get_empty_features()
        
        # Vectorized computation for numeric columns
        numeric_cols = window_data.select_dtypes(include=[np.number]).columns
        
        # Compute all statistics at once using agg
        if len(numeric_cols) > 0:
            stats = window_data[numeric_cols].agg([
                'mean', 'std', 'min', 'max', 'median'
            ])
            
            for col in numeric_cols:
                features[f'{col}_mean'] = stats.loc['mean', col]
                features[f'{col}_std'] = stats.loc['std', col] 
                features[f'{col}_min'] = stats.loc['min', col]
                features[f'{col}_max'] = stats.loc['max', col]
                features[f'{col}_median'] = stats.loc['median', col]
                
                # Compute percentiles efficiently
                percentiles = window_data[col].quantile([0.25, 0.75])
                features[f'{col}_q25'] = percentiles.iloc[0]
                features[f'{col}_q75'] = percentiles.iloc[1]
        
        # Add temporal features
        if 'timestamp' in window_data.columns:
            features['window_duration_sec'] = (
                window_data['timestamp'].max() - window_data['timestamp'].min()
            ).total_seconds()
            features['n_samples'] = len(window_data)
            
        return features
    
    @timed("compute_rolling_features")
    def _compute_rolling_features_optimized(
        self,
        window_data: pd.DataFrame,
        window_sizes: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Compute rolling window features efficiently"""
        features = {}
        
        if window_data.empty or len(window_data) < 2:
            return features
            
        numeric_cols = window_data.select_dtypes(include=[np.number]).columns
        
        for window in window_sizes:
            if len(window_data) < window:
                continue
                
            # Use rolling with method chaining for efficiency
            for col in numeric_cols:
                rolling = window_data[col].rolling(window=window, min_periods=1)
                
                # Compute multiple statistics in one pass
                features[f'{col}_roll{window}_mean'] = rolling.mean().iloc[-1]
                features[f'{col}_roll{window}_std'] = rolling.std().iloc[-1]
                features[f'{col}_roll{window}_max'] = rolling.max().iloc[-1]
                features[f'{col}_roll{window}_min'] = rolling.min().iloc[-1]
                
        return features
    
    @timed("process_single_example")
    def _process_single_example(
        self,
        example: Tuple[Any, ...],
        telemetry_by_job: Dict[str, pd.DataFrame],
        metadata_by_job: Dict[str, Dict]
    ) -> Optional[Dict]:
        """Process a single example with all optimizations"""
        
        jid, timestamp, label, horizon = example
        jid_str = str(jid)
        
        # Check cache first
        cache_key = f"{jid_str}_{timestamp}_{horizon}"
        if self._enable_cache and cache_key in self._feature_cache:
            cached_features = self._feature_cache[cache_key].copy()
            cached_features['label'] = label
            return cached_features
        
        # Get telemetry for this job
        if jid_str not in telemetry_by_job:
            return None
            
        job_telemetry = telemetry_by_job[jid_str]
        
        # Filter to window before prediction time
        window_data = job_telemetry[job_telemetry['time'] < timestamp]
        
        if window_data.empty:
            return None
        
        # Compute features
        features = {}
        features['jid'] = jid_str
        features['timestamp'] = timestamp
        features['horizon'] = horizon
        features['label'] = label
        
        # Statistical features (vectorized)
        stat_features = self._compute_statistical_features_vectorized(window_data)
        features.update(stat_features)
        
        # Rolling features (optimized)
        roll_features = self._compute_rolling_features_optimized(window_data)
        features.update(roll_features)
        
        # Add metadata features if available
        if jid_str in metadata_by_job:
            features.update(metadata_by_job[jid_str])
        
        # Cache the result (excluding label)
        if self._enable_cache:
            cache_value = {k: v for k, v in features.items() if k != 'label'}
            self._feature_cache[cache_key] = cache_value
            
        return features
    
    @timed("create_features_parallel")
    def create_features(
        self,
        telemetry_data: pd.DataFrame,
        labels: Union[List[Tuple], List[LabelExample]],
        job_metadata: Optional[pd.DataFrame] = None,
        n_workers: int = None
    ) -> pd.DataFrame:
        """Create features with parallel processing"""

        # Convert LabelExample objects to tuples if needed
        if labels and isinstance(labels[0], LabelExample):
            labels = [(ex.jid, ex.timestamp, ex.label, ex.horizon_minutes) for ex in labels]

        self.logger.info(f"Creating features for {len(labels)} examples")
        
        # Prepare data structures
        telemetry_by_job = self._group_telemetry_by_job(telemetry_data)
        
        metadata_by_job = {}
        if job_metadata is not None and 'jid' in job_metadata.columns:
            metadata_by_job = job_metadata.set_index('jid').to_dict('index')
        
        # Determine number of workers
        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, 8)
        
        # Process in parallel if beneficial
        results = []
        
        if len(labels) > 1000 and n_workers > 1:
            # Parallel processing for large datasets
            self.logger.info(f"Using {n_workers} workers for parallel processing")
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit jobs in batches to avoid memory issues
                batch_size = max(100, len(labels) // (n_workers * 10))
                futures = []
                
                for i in range(0, len(labels), batch_size):
                    batch = labels[i:i+batch_size]
                    future = executor.submit(
                        self._process_batch,
                        batch,
                        telemetry_by_job,
                        metadata_by_job
                    )
                    futures.append(future)
                
                # Collect results with progress tracking
                for future in as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    if len(results) % 10000 == 0:
                        self.logger.info(f"Processed {len(results)}/{len(labels)} examples")
        else:
            # Sequential processing for small datasets
            for i, example in enumerate(labels):
                result = self._process_single_example(
                    example, 
                    telemetry_by_job,
                    metadata_by_job
                )
                if result:
                    results.append(result)
                
                # Progress logging
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"Processed {i+1}/{len(labels)} examples")
        
        # Convert to DataFrame
        if not results:
            self.logger.warning("No features were successfully created")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(results)
        
        # Optimize memory usage
        features_df = self._optimize_dtypes(features_df)
        
        self.logger.info(f"Created features shape: {features_df.shape}")
        return features_df
    
    def _process_batch(
        self, 
        batch: List[Tuple],
        telemetry_by_job: Dict[str, pd.DataFrame],
        metadata_by_job: Dict[str, Dict]
    ) -> List[Dict]:
        """Process a batch of examples"""
        results = []
        for example in batch:
            result = self._process_single_example(
                example,
                telemetry_by_job, 
                metadata_by_job
            )
            if result:
                results.append(result)
        return results
    
    @timed("optimize_dtypes")
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        
        return df
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary with default values"""
        return {
            'cpu_mean': 0.0, 'cpu_std': 0.0, 'cpu_min': 0.0, 'cpu_max': 0.0,
            'memory_mean': 0.0, 'memory_std': 0.0, 'memory_min': 0.0, 'memory_max': 0.0,
            'n_samples': 0, 'window_duration_sec': 0.0
        }
    
    def clear_cache(self):
        """Clear the feature cache"""
        self._feature_cache.clear()
        self.logger.info("Feature cache cleared")
