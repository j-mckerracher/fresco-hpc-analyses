from .config import CACHE_DIR
"""
Streaming Parquet reader for FRESCO dataset.

Provides memory-efficient streaming access to telemetry data with 
job-level aggregation and intermediate caching.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Union, Any
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .config import (
    TELEMETRY_METRICS, 
    PROCESSING_CONFIG,
    ClusterType
)
from .io_discovery import DataFile
from .utils import (
    setup_logging,
    log_memory_usage, 
    memory_limit_check,
    save_artifact,
    load_artifact,
    create_stable_hash,
    ensure_directory,
    ProgressTracker,
    clean_numeric_column,
    validate_dataframe_columns
)


logger = logging.getLogger(__name__)


class ParquetStreamReader:
    """
    Memory-efficient streaming reader for FRESCO Parquet files.
    
    Features:
    - Lazy loading with column projection
    - Automatic job-level aggregation
    - Intermediate caching for performance
    - Robust error handling for corrupt files
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        memory_limit_gb: float = PROCESSING_CONFIG["memory_limit_gb"],
        chunk_size_rows: int = 200000,
        enable_caching: bool = True
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(CACHE_DIR)
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size_rows = chunk_size_rows
        self.enable_caching = enable_caching
        
        # Ensure cache directory exists
        if self.enable_caching:
            ensure_directory(self.cache_dir)
    
    def read_files_streaming(
        self,
        files: List[DataFile],
        required_columns: Optional[List[str]] = None,
        job_aggregation: bool = True,
        max_workers: int = PROCESSING_CONFIG["max_workers"]
    ) -> Iterator[pd.DataFrame]:
        """
        Stream data from multiple files with optional parallel processing.
        
        Args:
            files: List of data files to process
            required_columns: Columns to load (None = all available)
            job_aggregation: Whether to aggregate by job before yielding
            max_workers: Number of parallel workers
            
        Yields:
            DataFrames containing processed data
        """
        logger.info(f"Reading {len(files)} files with streaming")
        
        if max_workers > 1 and len(files) > 1:
            # Parallel processing for multiple files
            yield from self._read_files_parallel(
                files, required_columns, job_aggregation, max_workers
            )
        else:
            # Sequential processing
            progress = ProgressTracker(len(files), "Reading files")
            
            for file in files:
                try:
                    df = self.read_single_file(
                        file, 
                        required_columns=required_columns,
                        job_aggregation=job_aggregation
                    )
                    if not df.empty:
                        yield df
                    
                    progress.update()
                    
                    # Memory check
                    if memory_limit_check(self.memory_limit_gb):
                        logger.warning("Memory limit approached, forcing garbage collection")
                        import gc
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to read {file.path}: {e}")
                    progress.update()
                    continue
    
    def _read_files_parallel(
        self,
        files: List[DataFile],
        required_columns: Optional[List[str]],
        job_aggregation: bool,
        max_workers: int
    ) -> Iterator[pd.DataFrame]:
        """Parallel file processing with process pool"""
        
        # Create worker function with fixed parameters
        worker_func = partial(
            _read_file_worker,
            required_columns=required_columns,
            job_aggregation=job_aggregation,
            cache_dir=self.cache_dir if self.enable_caching else None,
            chunk_size_rows=self.chunk_size_rows
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files
            future_to_file = {
                executor.submit(worker_func, file): file 
                for file in files
            }
            
            progress = ProgressTracker(len(files), "Reading files (parallel)")
            
            # Process completed futures
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        yield df
                    
                    progress.update()
                    
                except Exception as e:
                    logger.error(f"Failed to process {file.path}: {e}")
                    progress.update()
                    continue
    
    def read_single_file(
        self,
        file: DataFile,
        required_columns: Optional[List[str]] = None,
        job_aggregation: bool = True
    ) -> pd.DataFrame:
        """
        Read a single Parquet file with optional caching.
        
        Args:
            file: File metadata
            required_columns: Columns to load
            job_aggregation: Whether to aggregate by job
            
        Returns:
            DataFrame with processed data
        """
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(file, required_columns, job_aggregation)
            cache_path = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_path.exists():
                try:
                    logger.debug(f"Loading from cache: {cache_path}")
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(f"Cache read failed for {cache_path}: {e}")
        
        # Read from source
        try:
            df = self._read_parquet_with_schema_detection(file.path, required_columns)
            
            if df.empty:
                logger.warning(f"Empty data from {file.path}")
                return df
            
            # Clean and validate data
            df = self._clean_telemetry_data(df, file.cluster)
            
            # Job-level aggregation if requested
            if job_aggregation and 'jid' in df.columns:
                df = self._aggregate_by_job(df)
            
            # Cache the result
            if self.enable_caching:
                try:
                    save_artifact(df, cache_path)
                    logger.debug(f"Cached result: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache {cache_path}: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read {file.path}: {e}")
            return pd.DataFrame()
    
    def _read_parquet_with_schema_detection(
        self,
        file_path: Path,
        required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Read Parquet file with automatic schema detection and column projection"""
        
        try:
            # Use PyArrow for efficient reading
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema
            
            # Determine which columns to read
            available_columns = schema.names
            columns_to_read = self._select_columns(available_columns, required_columns)
            
            if not columns_to_read:
                logger.warning(f"No valid columns found in {file_path}")
                return pd.DataFrame()
            
            logger.debug(f"Reading {len(columns_to_read)} columns from {file_path}")
            
            # Read with column projection
            table = parquet_file.read(columns=columns_to_read)
            df = table.to_pandas()
            
            logger.debug(f"Read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            # Fallback to pandas if PyArrow fails
            logger.warning(f"PyArrow read failed for {file_path}, trying pandas: {e}")
            try:
                return pd.read_parquet(file_path, columns=required_columns)
            except Exception as e2:
                logger.error(f"Pandas read also failed for {file_path}: {e2}")
                raise e2
    
    def _select_columns(
        self,
        available_columns: List[str],
        required_columns: Optional[List[str]] = None
    ) -> List[str]:
        """Select columns to read based on availability and requirements"""
        
        if required_columns is None:
            # Default columns to read
            default_cols = [
                'jid', 'time', 'submit_time', 'start_time', 'end_time',
                'exitcode', 'ncores', 'nhosts', 'queue', 'account',
                'username', 'job_username'
            ]
            # Add all telemetry metrics that are available
            for metric in TELEMETRY_METRICS.keys():
                default_cols.append(metric)
            
            required_columns = default_cols
        
        # Filter to only available columns
        columns_to_read = [col for col in required_columns if col in available_columns]
        
        # Log missing columns
        missing_cols = set(required_columns) - set(columns_to_read)
        if missing_cols:
            logger.debug(f"Missing columns: {sorted(missing_cols)}")
        
        return columns_to_read
    
    def _clean_telemetry_data(
        self,
        df: pd.DataFrame,
        cluster: Optional[ClusterType] = None
    ) -> pd.DataFrame:
        """Clean and validate telemetry data"""
        
        # Clean numeric telemetry columns
        for col, config in TELEMETRY_METRICS.items():
            if col in df.columns:
                # Check cluster restrictions
                if 'clusters' in config and cluster:
                    if cluster.value not in config['clusters']:
                        # Remove column not available for this cluster
                        df = df.drop(columns=[col])
                        continue
                
                # Clean numeric values
                valid_range = config.get('valid_range')
                missing_value = config.get('missing_value', 0.0)
                
                if missing_value is not None:
                    df[col] = clean_numeric_column(df[col], valid_range, missing_value)
                else:
                    # Optional metric, keep NaN for missing values
                    df[col] = clean_numeric_column(df[col], valid_range, np.nan)
        
        # Clean timestamp columns
        for time_col in ['time', 'submit_time', 'start_time', 'end_time']:
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        
        # Clean job ID
        if 'jid' in df.columns:
            df['jid'] = df['jid'].astype(str)
        
        # Remove rows with invalid timestamps (telemetry must have time)
        if 'time' in df.columns:
            df = df.dropna(subset=['time'])
        
        # Add cluster information if available
        if cluster:
            df['cluster'] = cluster.value
        
        return df
    
    def _aggregate_by_job(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate telemetry data by job for memory efficiency"""
        
        if 'jid' not in df.columns:
            return df
        
        # Separate job metadata from telemetry
        job_cols = [
            'jid', 'submit_time', 'start_time', 'end_time', 'exitcode',
            'ncores', 'nhosts', 'queue', 'account', 'username', 'job_username'
        ]
        available_job_cols = [col for col in job_cols if col in df.columns]
        
        # Get job metadata (should be constant per job)
        job_metadata = df[available_job_cols].drop_duplicates(subset=['jid'])
        
        # Get telemetry columns
        telemetry_cols = [col for col in TELEMETRY_METRICS.keys() if col in df.columns]
        
        if not telemetry_cols:
            return job_metadata
        
        # Aggregate telemetry by job
        agg_dict = {}
        for col in telemetry_cols:
            agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']
        
        # Add time for temporal features
        if 'time' in df.columns:
            agg_dict['time'] = ['min', 'max', 'count']
        
        # Perform aggregation
        telemetry_agg = df.groupby('jid').agg(agg_dict)
        
        # Flatten column names
        telemetry_agg.columns = [
            f"{col}_{agg}" for col, agg in telemetry_agg.columns
        ]
        telemetry_agg = telemetry_agg.reset_index()
        
        # Merge with job metadata
        result = job_metadata.merge(telemetry_agg, on='jid', how='left')
        
        logger.debug(f"Aggregated {len(df)} rows to {len(result)} jobs")
        return result
    
    def _generate_cache_key(
        self,
        file: DataFile,
        required_columns: Optional[List[str]],
        job_aggregation: bool
    ) -> str:
        """Generate stable cache key for file processing"""
        key_data = {
            'file_path': str(file.path),
            'file_size': file.size_bytes,
            'required_columns': sorted(required_columns) if required_columns else None,
            'job_aggregation': job_aggregation,
            'version': '1.0'  # Increment to invalidate old caches
        }
        return create_stable_hash(key_data)
    
    def clear_cache(self, pattern: str = "*") -> None:
        """Clear cached files matching pattern"""
        if not self.enable_caching or not self.cache_dir.exists():
            return
        
        import glob
        cache_files = list(self.cache_dir.glob(pattern))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.debug(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")


def _read_file_worker(
    file: DataFile,
    required_columns: Optional[List[str]] = None,
    job_aggregation: bool = True,
    cache_dir: Optional[Path] = None,
    chunk_size_rows: int = 200000
) -> Optional[pd.DataFrame]:
    """Worker function for parallel file processing"""
    
    # Set up logging for worker process
    logger = setup_logging()
    
    try:
        # Create reader instance for this worker
        reader = ParquetStreamReader(
            cache_dir=cache_dir,
            chunk_size_rows=chunk_size_rows,
            enable_caching=(cache_dir is not None)
        )
        
        # Process the file
        df = reader.read_single_file(
            file,
            required_columns=required_columns,
            job_aggregation=job_aggregation
        )
        
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"Worker failed to process {file.path}: {e}")
        return None


class DatasetBuilder:
    """
    Build training/test datasets from streamed telemetry data.
    
    Combines file discovery, streaming reading, and feature preparation.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None
    ):
        from .io_discovery import DataDiscovery
        
        self.discovery = DataDiscovery(data_root)
        self.reader = ParquetStreamReader(cache_dir=cache_dir)
    
    def build_dataset(
        self,
        clusters: Optional[Set[ClusterType]] = None,
        years: Optional[Set[int]] = None,
        months: Optional[Set[int]] = None,
        required_columns: Optional[List[str]] = None,
        job_aggregation: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build complete dataset from discovered files.
        
        Returns:
            Combined DataFrame with all processed data
        """
        
        # Discover files
        files = self.discovery.discover_files(
            clusters=clusters,
            years=years, 
            months=months
        )
        
        if not files:
            logger.warning("No files found matching criteria")
            return pd.DataFrame()
        
        # Validate files
        files = self.discovery.validate_files(files)
        
        if not files:
            logger.error("No valid files found")
            return pd.DataFrame()
        
        # Process files in streaming fashion
        dataframes = []
        
        for df_chunk in self.reader.read_files_streaming(
            files,
            required_columns=required_columns,
            job_aggregation=job_aggregation,
            max_workers=max_workers or PROCESSING_CONFIG["max_workers"]
        ):
            dataframes.append(df_chunk)
            
            # Memory management
            if memory_limit_check():
                logger.info("Combining chunks to manage memory")
                combined = pd.concat(dataframes, ignore_index=True)
                dataframes = [combined]
        
        # Combine all chunks
        if not dataframes:
            logger.warning("No data was successfully processed")
            return pd.DataFrame()
        
        final_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Built dataset with {len(final_df)} records")
        
        # Save if output path specified
        if output_path:
            save_artifact(final_df, output_path)
            logger.info(f"Saved dataset to {output_path}")
        
        return final_df

    def build_job_metadata(
        self,
        clusters: Optional[Set[ClusterType]] = None,
        years: Optional[Set[int]] = None,
        months: Optional[Set[int]] = None,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build job metadata with one row per job.
        
        Returns:
            DataFrame with unique job metadata (one row per jid)
        """
        # Get aggregated data (may have duplicates across files)
        job_data = self.build_dataset(
            clusters=clusters,
            years=years,
            months=months,
            job_aggregation=True,
            max_workers=max_workers
        )
        
        if job_data.empty:
            return job_data
            
        # Deduplicate by keeping the row with latest end_time per job
        logger.info(f"Deduplicating {len(job_data)} rows to unique jobs")
        
        # Sort by jid and end_time, keep last (latest) per jid
        if 'end_time' in job_data.columns:
            job_data = job_data.sort_values(['jid', 'end_time'])
        
        job_metadata = job_data.drop_duplicates(subset=['jid'], keep='last')
        logger.info(f"Reduced to {len(job_metadata)} unique jobs")
        
        return job_metadata
    
    def build_telemetry_timeseries(
        self,
        clusters: Optional[Set[ClusterType]] = None,
        years: Optional[Set[int]] = None,
        months: Optional[Set[int]] = None,
        target_jids: Optional[Set[str]] = None,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build raw telemetry time-series data.
        
        Args:
            clusters: Filter by clusters
            years: Filter by years  
            months: Filter by months
            target_jids: Only keep data for these job IDs (for memory efficiency)
            
        Returns:
            DataFrame with time-series telemetry (multiple rows per job)
        """
        files = self.discovery.discover_files(
            clusters=clusters,
            years=years,
            months=months
        )
        
        if not files:
            logger.warning("No files found matching criteria")
            return pd.DataFrame()
        
        # Validate files
        files = self.discovery.validate_files(files)
        
        # Process files in streaming fashion
        dataframes = []
        
        for df_chunk in self.reader.read_files_streaming(
            files,
            job_aggregation=False,  # Keep raw time-series
            max_workers=max_workers or PROCESSING_CONFIG["max_workers"]
        ):
            # Filter by target jids if specified
            if target_jids and 'jid' in df_chunk.columns:
                df_chunk = df_chunk[df_chunk['jid'].astype(str).isin(target_jids)]
            
            if not df_chunk.empty:
                dataframes.append(df_chunk)
            
            # Memory management
            if memory_limit_check():
                logger.info("Combining chunks to manage memory")
                combined = pd.concat(dataframes, ignore_index=True)
                dataframes = [combined]
        
        # Combine all chunks
        if not dataframes:
            logger.warning("No telemetry data was successfully processed")
            return pd.DataFrame()
        
        telemetry_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Built telemetry dataset with {len(telemetry_df)} time-series records")
        
        return telemetry_df
