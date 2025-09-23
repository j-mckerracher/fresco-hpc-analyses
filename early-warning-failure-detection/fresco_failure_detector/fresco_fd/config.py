"""
Configuration management for FRESCO failure detection pipeline.

Contains all configurable parameters, thresholds, and system settings.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass
import logging


class ExitCodeClass(Enum):
    """Canonical job exit code classifications"""
    COMPLETED = "completed"
    FAILED = "failed" 
    TIMEOUT = "timeout"
    OOM = "oom"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ClusterType(Enum):
    """Supported HPC clusters"""
    CONTE = "conte"
    STAMPEDE = "stampede" 
    ANVIL = "anvil"


@dataclass
class ClusterConfig:
    """Per-cluster configuration"""
    name: str
    memory_gb_per_core: float
    has_gpu: bool = False
    max_job_duration_hours: int = 72


# Exit code mappings (extend as needed based on actual data)
EXITCODE_MAPPINGS: Dict[Union[int, str], ExitCodeClass] = {
    # Standard SLURM exit codes
    0: ExitCodeClass.COMPLETED,
    "0": ExitCodeClass.COMPLETED,
    "COMPLETED": ExitCodeClass.COMPLETED,
    "SUCCESS": ExitCodeClass.COMPLETED,
    
    # Failure categories
    1: ExitCodeClass.FAILED,
    "1": ExitCodeClass.FAILED,
    "FAILED": ExitCodeClass.FAILED,
    "ERROR": ExitCodeClass.FAILED,
    "NODE_FAIL": ExitCodeClass.FAILED,
    
    # Timeout
    "TIMEOUT": ExitCodeClass.TIMEOUT,
    "DL": ExitCodeClass.TIMEOUT,  # DeadLine exceeded
    
    # Out of memory
    "OOM": ExitCodeClass.OOM,
    "OUT_OF_MEMORY": ExitCodeClass.OOM,
    
    # System/admin terminated
    "CANCELLED": ExitCodeClass.CANCELLED,
    "PREEMPTED": ExitCodeClass.CANCELLED,
    
    # Aborted
    "ABORTED": ExitCodeClass.ABORTED,
    "BOOT_FAIL": ExitCodeClass.ABORTED,
    "REVOKED": ExitCodeClass.ABORTED,
}

# Classification into positive (failure) and negative (success) classes
POSITIVE_CLASSES: Set[ExitCodeClass] = {
    ExitCodeClass.FAILED,
    ExitCodeClass.TIMEOUT, 
    ExitCodeClass.OOM,
    ExitCodeClass.ABORTED
}

NEGATIVE_CLASSES: Set[ExitCodeClass] = {
    ExitCodeClass.COMPLETED,
    ExitCodeClass.CANCELLED  # Treat user cancellation as non-failure by default
}

# Prediction horizons (minutes)
PREDICTION_HORIZONS: List[int] = [5, 15, 60]

# Cluster configurations
CLUSTER_CONFIGS: Dict[ClusterType, ClusterConfig] = {
    ClusterType.CONTE: ClusterConfig(
        name="conte", 
        memory_gb_per_core=4.0,
        has_gpu=False
    ),
    ClusterType.STAMPEDE: ClusterConfig(
        name="stampede",
        memory_gb_per_core=2.0, 
        has_gpu=False
    ),
    ClusterType.ANVIL: ClusterConfig(
        name="anvil",
        memory_gb_per_core=2.0,
        has_gpu=True  # GPU nodes available
    )
}

# Telemetry metrics configuration
TELEMETRY_METRICS: Dict[str, Dict] = {
    "value_cpuuser": {
        "display_name": "CPU User %",
        "unit": "percent", 
        "missing_value": 0.0,
        "valid_range": (0.0, 100.0)
    },
    "value_memused": {
        "display_name": "Memory Used",
        "unit": "GB",
        "missing_value": 0.0,
        "valid_range": (0.0, 2048.0)  # Max reasonable memory per node
    },
    "value_memused_minus_diskcache": {
        "display_name": "Memory Used (No Cache)",
        "unit": "GB", 
        "missing_value": None,  # Optional metric
        "valid_range": (0.0, 2048.0)
    },
    "value_nfs": {
        "display_name": "NFS I/O",
        "unit": "MB/s",
        "missing_value": 0.0,
        "valid_range": (0.0, 10000.0)
    },
    "value_block": {
        "display_name": "Block I/O", 
        "unit": "GB/s",
        "missing_value": 0.0,
        "valid_range": (0.0, 100.0)
    },
    "value_gpu": {
        "display_name": "GPU Utilization",
        "unit": "percent",
        "missing_value": None,  # Only available on GPU nodes
        "valid_range": (0.0, 100.0),
        "clusters": ["anvil"]  # Restrict to specific clusters
    }
}

# Feature engineering configuration
FEATURE_WINDOWS: Dict[str, List[int]] = {
    "short": [1, 2, 5],      # Short-term windows (minutes)
    "trend": [15, 30, 60]     # Trend windows (minutes)
}

# Memory thresholds for pressure detection
MEMORY_PRESSURE_THRESHOLDS: Dict[str, float] = {
    "warning": 0.80,   # 80% memory usage
    "critical": 0.95,  # 95% memory usage
    "extreme": 0.98    # 98% memory usage
}

# Sampling configuration
SAMPLING_CONFIG: Dict[str, Union[int, float]] = {
    "neg_pos_ratio": 3,           # Negatives per positive example
    "min_job_duration_minutes": 10,  # Minimum job duration to consider
    "max_examples_per_job": 5,    # Max examples from single job (avoid correlation)
    "validation_split": 0.2,     # Validation set fraction
}

# Model configuration
MODEL_CONFIG: Dict = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": 3  # Handle class imbalance
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "is_unbalance": True
    },
    "logistic_regression": {
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 1000,
        "n_jobs": -1
    }
}

# Evaluation thresholds
EVALUATION_CONFIG: Dict = {
    "max_fpr": 0.01,              # Maximum false positive rate
    "alert_budget_per_1k": 25,    # Max alerts per 1000 jobs
    "min_lead_time_minutes": 5,   # Minimum required lead time
    "calibration_bins": 10        # Number of calibration bins
}

# File system configuration  
# Base project directory
DEFAULT_DATA_ROOT = "/home/dynamo/a/jmckerra/projects/fresco-analysis/data"
PROJECT_ROOT = "/home/dynamo/a/jmckerra/projects/fresco-analysis/early-warning-failure-detection"
CACHE_DIR = f"{PROJECT_ROOT}/artifacts/cache"

OUTPUT_DIR = f"{PROJECT_ROOT}/artifacts"

# Processing configuration
PROCESSING_CONFIG: Dict = {
    "max_workers": 8,             # Parallel processing workers
    "memory_limit_gb": 20,        # Soft memory limit
    "chunk_size_mb": 100,         # Parquet chunk size
    "progress_report_interval": 1000,  # Log progress every N jobs
    "telemetry_engine": "polars", # Engine for telemetry reading: "polars" or "pandas"
}

# Logging configuration
LOGGING_CONFIG: Dict = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}


def get_cluster_from_path(file_path: Union[str, Path]) -> Optional[ClusterType]:
    """Extract cluster type from file path"""
    path_str = str(file_path).lower()
    
    for cluster in ClusterType:
        if cluster.value in path_str:
            return cluster
    return None


def validate_config() -> bool:
    """Validate configuration consistency"""
    # Check that all positive and negative classes are mutually exclusive
    if POSITIVE_CLASSES & NEGATIVE_CLASSES:
        raise ValueError("Positive and negative classes must be mutually exclusive")
    
    # Check prediction horizons are reasonable
    if not all(h > 0 for h in PREDICTION_HORIZONS):
        raise ValueError("All prediction horizons must be positive")
        
    # Check cluster configs
    for cluster_type, config in CLUSTER_CONFIGS.items():
        if config.memory_gb_per_core <= 0:
            raise ValueError(f"Invalid memory config for {cluster_type.value}")
    
    return True


# Validate configuration on import
validate_config()