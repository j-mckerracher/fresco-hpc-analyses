"""
Utility functions for FRESCO failure detection pipeline.

Includes memory management, logging, serialization, and common helper functions.
"""

import hashlib
import logging
import pickle
import psutil
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

from .config import LOGGING_CONFIG, PROCESSING_CONFIG


# Global console for rich output
console = Console()


def setup_logging(
    level: int = LOGGING_CONFIG["level"],
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging with rich formatting"""
    
    # Create logger
    logger = logging.getLogger("fresco_fd")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    rich_handler = RichHandler(console=console, show_time=True, show_path=False)
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]"
        )
    )
    logger.addHandler(rich_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                fmt=LOGGING_CONFIG["format"],
                datefmt=LOGGING_CONFIG["datefmt"]
            )
        )
        logger.addHandler(file_handler)
    
    return logger


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_gb": memory_info.rss / (1024**3),  # Resident Set Size
        "vms_gb": memory_info.vms / (1024**3),  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available_gb": psutil.virtual_memory().available / (1024**3)
    }


def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """Log current memory usage with context"""
    memory_stats = get_memory_usage()
    logger.info(
        f"Memory usage{' (' + context + ')' if context else ''}: "
        f"RSS={memory_stats['rss_gb']:.1f}GB, "
        f"Available={memory_stats['available_gb']:.1f}GB, "
        f"Process={memory_stats['percent']:.1f}%"
    )


def memory_limit_check(limit_gb: float = PROCESSING_CONFIG["memory_limit_gb"]) -> bool:
    """Check if memory usage exceeds soft limit"""
    memory_stats = get_memory_usage()
    return memory_stats["rss_gb"] > limit_gb


def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger = logging.getLogger("fresco_fd")
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper


def create_stable_hash(data: Union[str, Dict, List]) -> str:
    """Create a stable hash for caching purposes"""
    if isinstance(data, (dict, list)):
        # Convert to sorted string representation for stable hashing
        data_str = str(sorted(data.items()) if isinstance(data, dict) else sorted(data))
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def save_artifact(
    obj: Any,
    filepath: Union[str, Path],
    compress: bool = True
) -> None:
    """Save object to disk with optional compression"""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    if compress and filepath.suffix in ['.pkl', '.joblib']:
        joblib.dump(obj, filepath, compress=True)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    else:
        # For pandas objects
        if hasattr(obj, 'to_parquet'):
            obj.to_parquet(filepath)
        elif hasattr(obj, 'to_csv'):
            obj.to_csv(filepath)
        else:
            joblib.dump(obj, filepath)


def load_artifact(filepath: Union[str, Path]) -> Any:
    """Load object from disk"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    
    if filepath.suffix in ['.pkl', '.joblib']:
        return joblib.load(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath, index_col=0)
    else:
        # Try joblib first, then pickle
        try:
            return joblib.load(filepath)
        except:
            with open(filepath, 'rb') as f:
                return pickle.load(f)


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    context: str = ""
) -> None:
    """Validate that DataFrame has required columns"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns{' in ' + context if context else ''}: "
            f"{sorted(missing_columns)}"
        )


def safe_percentile(
    data: Union[np.ndarray, pd.Series],
    percentile: float,
    default: float = np.nan
) -> float:
    """Calculate percentile safely, handling empty data"""
    if len(data) == 0:
        return default
    return np.percentile(data, percentile)


def clean_numeric_column(
    series: pd.Series,
    valid_range: Optional[tuple] = None,
    fill_value: float = 0.0
) -> pd.Series:
    """Clean numeric column by handling outliers and missing values"""
    # Convert to numeric, coercing errors to NaN
    series = pd.to_numeric(series, errors='coerce')
    
    # Apply valid range filter if specified
    if valid_range:
        min_val, max_val = valid_range
        series = series.where(
            (series >= min_val) & (series <= max_val),
            np.nan
        )
    
    # Fill missing values
    series = series.fillna(fill_value)
    
    return series


def batch_process_with_progress(
    items: List[Any],
    process_func,
    batch_size: int = 100,
    desc: str = "Processing"
) -> List[Any]:
    """Process items in batches with progress bar"""
    results = []
    
    for i in track(range(0, len(items), batch_size), description=desc):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_memory(bytes_val: float) -> str:
    """Format memory size to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"


class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.last_report = 0
        self.report_interval = max(1, total // 100)  # Report every 1%
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps"""
        self.current += n
        
        # Report progress at intervals
        if self.current - self.last_report >= self.report_interval or self.current == self.total:
            self.report()
            self.last_report = self.current
    
    def report(self) -> None:
        """Report current progress"""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100
        rate = self.current / elapsed if elapsed > 0 else 0
        
        logger = logging.getLogger("fresco_fd")
        logger.info(
            f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%) "
            f"[{rate:.1f} items/s, {format_duration(elapsed)} elapsed]"
        )


def chunked(lst: List[Any], chunk_size: int):
    """Yield successive chunks of specified size from list"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)