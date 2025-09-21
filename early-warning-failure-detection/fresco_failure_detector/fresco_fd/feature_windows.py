"""
Rolling window feature computation for telemetry data.

Computes time-windowed statistics with strict no-leakage guarantees.
All features are computed using only data up to time t.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .config import (
    TELEMETRY_METRICS,
    FEATURE_WINDOWS,
    CLUSTER_CONFIGS,
    ClusterType
)
from .utils import safe_divide, safe_percentile


logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')


class WindowFeatureComputer:
    """
    Computes rolling window features from telemetry time series.
    
    Features computed:
    - Level: mean, median, last, percentiles
    - Dynamics: slope, acceleration, percent change
    - Variability: std, IQR, coefficient of variation
    - Extremes: min, max, time over threshold, spike count
    - Temporal: autocorrelation, burstiness, entropy
    """
    
    def __init__(
        self,
        short_windows: List[int] = FEATURE_WINDOWS["short"],
        trend_windows: List[int] = FEATURE_WINDOWS["trend"],
        min_points: int = 3
    ):
        self.short_windows = short_windows  # minutes
        self.trend_windows = trend_windows  # minutes  
        self.min_points = min_points  # Minimum points required for computation
    
    def compute_features(
        self,
        telemetry_data: pd.DataFrame,
        target_time: datetime,
        job_metadata: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute all window features at target time.
        
        Args:
            telemetry_data: DataFrame with time-indexed telemetry
            target_time: Time at which to compute features (no leakage beyond this)
            job_metadata: Optional job context (ncores, cluster, etc.)
            
        Returns:
            Dictionary of computed features
        """
        # Filter data up to target time (strict no-leakage)
        valid_data = telemetry_data[telemetry_data['time'] <= target_time]
        
        if len(valid_data) == 0:
            logger.warning(f"No telemetry data available up to {target_time}")
            return self._get_default_features()
        
        # Sort by time for proper windowing
        valid_data = valid_data.sort_values('time')
        
        features = {}
        
        # Compute features for each metric
        for metric_name, metric_config in TELEMETRY_METRICS.items():
            if metric_name not in valid_data.columns:
                continue
                
            metric_features = self._compute_metric_features(
                valid_data,
                metric_name,
                target_time,
                metric_config,
                job_metadata
            )
            
            features.update(metric_features)
        
        # Cross-metric coupling features
        coupling_features = self._compute_coupling_features(valid_data, target_time)
        features.update(coupling_features)
        
        # Job context features
        if job_metadata:
            context_features = self._compute_context_features(job_metadata, target_time)
            features.update(context_features)
        
        return features
    
    def _compute_metric_features(
        self,
        data: pd.DataFrame,
        metric_name: str,
        target_time: datetime,
        metric_config: Dict,
        job_metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute features for a single telemetry metric"""
        
        features = {}
        metric_series = data[metric_name].dropna()
        
        if len(metric_series) == 0:
            return self._get_default_metric_features(metric_name)
        
        # Compute features for each window type
        for window_type, window_sizes in [
            ("short", self.short_windows),
            ("trend", self.trend_windows)
        ]:
            for window_minutes in window_sizes:
                window_start = target_time - timedelta(minutes=window_minutes)
                window_data = data[
                    (data['time'] >= window_start) & (data['time'] <= target_time)
                ][metric_name].dropna()
                
                if len(window_data) < self.min_points:
                    # Fill with defaults
                    window_features = self._get_default_window_features(
                        metric_name, window_type, window_minutes
                    )
                else:
                    window_features = self._compute_window_features(
                        window_data,
                        data[data[metric_name].notna()]['time'],
                        metric_name,
                        window_type, 
                        window_minutes,
                        metric_config,
                        job_metadata
                    )
                
                features.update(window_features)
        
        return features
    
    def _compute_window_features(
        self,
        values: pd.Series,
        timestamps: pd.Series,
        metric_name: str,
        window_type: str,
        window_minutes: int,
        metric_config: Dict,
        job_metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute statistical features for a time window"""
        
        prefix = f"{metric_name}_{window_type}_{window_minutes}min"
        features = {}
        
        if len(values) == 0:
            return self._get_default_window_features(metric_name, window_type, window_minutes)
        
        # Level statistics
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_median"] = float(np.median(values))
        features[f"{prefix}_last"] = float(values.iloc[-1])
        features[f"{prefix}_first"] = float(values.iloc[0])
        
        # Variability statistics
        if len(values) > 1:
            features[f"{prefix}_std"] = float(np.std(values))
            features[f"{prefix}_var"] = float(np.var(values))
            features[f"{prefix}_cv"] = safe_divide(np.std(values), np.mean(values), 0.0)
            features[f"{prefix}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        else:
            features[f"{prefix}_std"] = 0.0
            features[f"{prefix}_var"] = 0.0
            features[f"{prefix}_cv"] = 0.0
            features[f"{prefix}_iqr"] = 0.0
        
        # Extreme statistics
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        features[f"{prefix}_range"] = float(np.max(values) - np.min(values))
        features[f"{prefix}_p10"] = float(np.percentile(values, 10))
        features[f"{prefix}_p90"] = float(np.percentile(values, 90))
        
        # Dynamics (require at least 3 points)
        if len(values) >= 3 and len(timestamps) >= 3:
            dynamics = self._compute_dynamics_features(values, timestamps, prefix)
            features.update(dynamics)
        
        # Threshold-based features (context-aware)
        threshold_features = self._compute_threshold_features(
            values, prefix, metric_config, job_metadata
        )
        features.update(threshold_features)
        
        # Temporal pattern features
        if len(values) >= 5:
            temporal_features = self._compute_temporal_features(values, timestamps, prefix)
            features.update(temporal_features)
        
        return features
    
    def _compute_dynamics_features(
        self,
        values: pd.Series,
        timestamps: pd.Series,
        prefix: str
    ) -> Dict[str, float]:
        """Compute dynamics features (slope, acceleration, change)"""
        
        features = {}
        
        if len(values) < 3:
            features[f"{prefix}_slope"] = 0.0
            features[f"{prefix}_pct_change"] = 0.0
            return features
        
        # Convert timestamps to numeric (seconds since start)
        time_numeric = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        values_numeric = values.values
        
        # Linear slope (robust to outliers)
        try:
            if len(set(time_numeric)) > 1:  # Avoid division by zero
                slope, _, _, _, _ = stats.linregress(time_numeric, values_numeric)
                features[f"{prefix}_slope"] = float(slope)
            else:
                features[f"{prefix}_slope"] = 0.0
        except Exception:
            features[f"{prefix}_slope"] = 0.0
        
        # Percent change (last vs first)
        first_val = values_numeric[0]
        last_val = values_numeric[-1]
        
        if abs(first_val) > 1e-10:
            pct_change = (last_val - first_val) / first_val
            features[f"{prefix}_pct_change"] = float(pct_change)
        else:
            features[f"{prefix}_pct_change"] = 0.0
        
        # Acceleration (change in slope, requires 5+ points)
        if len(values) >= 5:
            mid_point = len(values) // 2
            
            # First half slope
            try:
                slope1, _, _, _, _ = stats.linregress(
                    time_numeric[:mid_point+1], 
                    values_numeric[:mid_point+1]
                )
            except Exception:
                slope1 = 0.0
            
            # Second half slope  
            try:
                slope2, _, _, _, _ = stats.linregress(
                    time_numeric[mid_point:], 
                    values_numeric[mid_point:]
                )
            except Exception:
                slope2 = 0.0
            
            features[f"{prefix}_acceleration"] = float(slope2 - slope1)
        else:
            features[f"{prefix}_acceleration"] = 0.0
        
        return features
    
    def _compute_threshold_features(
        self,
        values: pd.Series,
        prefix: str,
        metric_config: Dict,
        job_metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """Compute threshold-based features (context-aware)"""
        
        features = {}
        
        # Generic thresholds based on data distribution
        if len(values) > 0:
            p75 = np.percentile(values, 75)
            p95 = np.percentile(values, 95)
            
            # Time over high thresholds
            features[f"{prefix}_time_over_p75"] = float(np.mean(values > p75))
            features[f"{prefix}_time_over_p95"] = float(np.mean(values > p95))
            
            # Spike detection (values > mean + 2*std)
            if len(values) > 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                spike_threshold = mean_val + 2 * std_val
                spike_count = np.sum(values > spike_threshold)
                features[f"{prefix}_spike_count"] = float(spike_count)
                features[f"{prefix}_spike_rate"] = float(spike_count / len(values))
            else:
                features[f"{prefix}_spike_count"] = 0.0
                features[f"{prefix}_spike_rate"] = 0.0
        
        # Metric-specific thresholds
        metric_name = prefix.split('_')[0]
        
        if metric_name == 'value_memused' and job_metadata:
            # Memory pressure features
            cluster = job_metadata.get('cluster')
            ncores = job_metadata.get('ncores', 1)
            
            if cluster and cluster in CLUSTER_CONFIGS:
                gb_per_core = CLUSTER_CONFIGS[cluster].memory_gb_per_core
                estimated_allocation = ncores * gb_per_core
                
                if estimated_allocation > 0:
                    memory_pressure = values / estimated_allocation
                    features[f"{prefix}_memory_pressure"] = float(np.mean(memory_pressure))
                    features[f"{prefix}_time_over_80pct"] = float(np.mean(memory_pressure > 0.8))
                    features[f"{prefix}_time_over_95pct"] = float(np.mean(memory_pressure > 0.95))
        
        elif metric_name == 'value_cpuuser':
            # CPU utilization features
            features[f"{prefix}_time_idle"] = float(np.mean(values < 5.0))  # Very low CPU
            features[f"{prefix}_time_busy"] = float(np.mean(values > 80.0))  # High CPU
            
        elif metric_name in ['value_nfs', 'value_block']:
            # I/O activity features
            features[f"{prefix}_time_no_io"] = float(np.mean(values < 0.01))  # No I/O
            features[f"{prefix}_time_high_io"] = float(np.mean(values > np.percentile(values, 90)))
        
        return features
    
    def _compute_temporal_features(
        self,
        values: pd.Series,
        timestamps: pd.Series,
        prefix: str
    ) -> Dict[str, float]:
        """Compute temporal pattern features"""
        
        features = {}
        
        if len(values) < 5:
            features[f"{prefix}_autocorr_lag1"] = 0.0
            features[f"{prefix}_burstiness"] = 0.0
            return features
        
        values_array = values.values
        
        # Lag-1 autocorrelation (safe computation)
        try:
            if len(values_array) > 1 and np.std(values_array) > 1e-10:
                autocorr = np.corrcoef(values_array[:-1], values_array[1:])[0, 1]
                if not np.isnan(autocorr):
                    features[f"{prefix}_autocorr_lag1"] = float(autocorr)
                else:
                    features[f"{prefix}_autocorr_lag1"] = 0.0
            else:
                features[f"{prefix}_autocorr_lag1"] = 0.0
        except Exception:
            features[f"{prefix}_autocorr_lag1"] = 0.0
        
        # Burstiness (variance in inter-crossing intervals)
        try:
            mean_val = np.mean(values_array)
            crossings = np.where(np.diff(np.signbit(values_array - mean_val)))[0]
            
            if len(crossings) > 2:
                intervals = np.diff(crossings)
                if len(intervals) > 1 and np.mean(intervals) > 0:
                    burstiness = np.std(intervals) / np.mean(intervals)
                    features[f"{prefix}_burstiness"] = float(burstiness)
                else:
                    features[f"{prefix}_burstiness"] = 0.0
            else:
                features[f"{prefix}_burstiness"] = 0.0
        except Exception:
            features[f"{prefix}_burstiness"] = 0.0
        
        return features
    
    def _compute_coupling_features(
        self,
        data: pd.DataFrame,
        target_time: datetime
    ) -> Dict[str, float]:
        """Compute cross-metric coupling features"""
        
        features = {}
        
        # Get recent window for coupling analysis
        coupling_window = 15  # minutes
        window_start = target_time - timedelta(minutes=coupling_window)
        window_data = data[
            (data['time'] >= window_start) & (data['time'] <= target_time)
        ]
        
        if len(window_data) < 3:
            return {}
        
        # Memory vs CPU coupling
        if 'value_memused' in window_data.columns and 'value_cpuuser' in window_data.columns:
            mem_data = window_data['value_memused'].dropna()
            cpu_data = window_data['value_cpuuser'].dropna()
            
            if len(mem_data) > 0 and len(cpu_data) > 0:
                # High memory, low CPU (potential memory leak/pressure)
                mem_high = np.mean(mem_data) > np.percentile(mem_data, 75)
                cpu_low = np.mean(cpu_data) < 20.0
                features['mem_high_cpu_low'] = float(mem_high and cpu_low)
                
                # Memory slope vs CPU slope
                if len(mem_data) >= 3 and len(cpu_data) >= 3:
                    try:
                        mem_times = window_data[window_data['value_memused'].notna()]['time']
                        cpu_times = window_data[window_data['value_cpuuser'].notna()]['time']
                        
                        mem_time_numeric = (mem_times - mem_times.iloc[0]).dt.total_seconds()
                        cpu_time_numeric = (cpu_times - cpu_times.iloc[0]).dt.total_seconds()
                        
                        mem_slope, _, _, _, _ = stats.linregress(mem_time_numeric, mem_data)
                        cpu_slope, _, _, _, _ = stats.linregress(cpu_time_numeric, cpu_data)
                        
                        features['mem_cpu_slope_diff'] = float(mem_slope - cpu_slope)
                    except Exception:
                        features['mem_cpu_slope_diff'] = 0.0
        
        # I/O vs CPU coupling
        io_metrics = ['value_nfs', 'value_block']
        available_io = [m for m in io_metrics if m in window_data.columns]
        
        if available_io and 'value_cpuuser' in window_data.columns:
            # Aggregate I/O activity
            io_activity = 0
            for io_metric in available_io:
                io_data = window_data[io_metric].dropna()
                if len(io_data) > 0:
                    io_activity += np.mean(io_data)
            
            cpu_data = window_data['value_cpuuser'].dropna()
            if len(cpu_data) > 0:
                cpu_low = np.mean(cpu_data) < 20.0
                io_high = io_activity > 0  # Any I/O activity
                features['io_high_cpu_low'] = float(io_high and cpu_low)
        
        # GPU vs CPU coupling (if GPU available)
        if 'value_gpu' in window_data.columns and 'value_cpuuser' in window_data.columns:
            gpu_data = window_data['value_gpu'].dropna()
            cpu_data = window_data['value_cpuuser'].dropna()
            
            if len(gpu_data) > 0 and len(cpu_data) > 0:
                gpu_high = np.mean(gpu_data) > 50.0
                cpu_low = np.mean(cpu_data) < 20.0
                features['gpu_high_cpu_low'] = float(gpu_high and cpu_low)
        
        return features
    
    def _compute_context_features(
        self,
        job_metadata: Dict,
        target_time: datetime
    ) -> Dict[str, float]:
        """Compute job context features"""
        
        features = {}
        
        # Job size features
        ncores = job_metadata.get('ncores', 1)
        nhosts = job_metadata.get('nhosts', 1)
        
        features['job_ncores'] = float(ncores)
        features['job_nhosts'] = float(nhosts)
        features['job_cores_per_host'] = safe_divide(ncores, nhosts, 1.0)
        
        # Job age (if start time available)
        start_time = job_metadata.get('start_time')
        if start_time and isinstance(start_time, datetime):
            job_age_minutes = (target_time - start_time).total_seconds() / 60.0
            features['job_age_minutes'] = float(job_age_minutes)
            
            # Job age ratio (estimate based on typical job durations)
            # This is a rough estimate - in practice could use historical data
            estimated_duration = max(60, job_age_minutes * 2)  # Rough heuristic
            features['job_age_ratio'] = float(job_age_minutes / estimated_duration)
        
        # Cluster context
        cluster = job_metadata.get('cluster')
        if cluster:
            for cluster_type in ClusterType:
                features[f'cluster_{cluster_type.value}'] = float(cluster == cluster_type.value)
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when no data available"""
        features = {}
        
        for metric_name in TELEMETRY_METRICS.keys():
            metric_features = self._get_default_metric_features(metric_name)
            features.update(metric_features)
        
        return features
    
    def _get_default_metric_features(self, metric_name: str) -> Dict[str, float]:
        """Get default features for a single metric"""
        features = {}
        
        for window_type, window_sizes in [
            ("short", self.short_windows),
            ("trend", self.trend_windows)
        ]:
            for window_minutes in window_sizes:
                window_features = self._get_default_window_features(
                    metric_name, window_type, window_minutes
                )
                features.update(window_features)
        
        return features
    
    def _get_default_window_features(
        self, 
        metric_name: str, 
        window_type: str, 
        window_minutes: int
    ) -> Dict[str, float]:
        """Get default values for window features"""
        prefix = f"{metric_name}_{window_type}_{window_minutes}min"
        
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_last": 0.0,
            f"{prefix}_first": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_var": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_slope": 0.0,
            f"{prefix}_pct_change": 0.0,
            f"{prefix}_acceleration": 0.0,
            f"{prefix}_time_over_p75": 0.0,
            f"{prefix}_time_over_p95": 0.0,
            f"{prefix}_spike_count": 0.0,
            f"{prefix}_spike_rate": 0.0,
            f"{prefix}_autocorr_lag1": 0.0,
            f"{prefix}_burstiness": 0.0,
        }