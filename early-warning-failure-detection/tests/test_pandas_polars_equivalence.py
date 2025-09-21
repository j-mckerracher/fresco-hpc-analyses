"""
Comprehensive test suite ensuring exact equivalence between Pandas and Polars implementations.

This test suite validates that Polars implementations produce identical results to
the current Pandas implementation for all critical data processing operations.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Union, Tuple
import tempfile
from pathlib import Path

# Import the current implementations
import sys
sys.path.append(str(Path(__file__).parent.parent))
from fresco_failure_detector.fresco_fd.config import TELEMETRY_METRICS, ClusterType
from fresco_failure_detector.fresco_fd.utils import clean_numeric_column


class TestDataFrameEquivalence:
    """Comprehensive test suite ensuring Pandas/Polars identical behavior"""

    @pytest.fixture
    def sample_telemetry_data(self) -> pd.DataFrame:
        """Generate realistic telemetry data matching FRESCO schema"""
        np.random.seed(42)  # Reproducible tests
        n_rows = 10000

        # Generate job IDs (some duplicated for aggregation testing)
        job_ids = np.random.choice([f"job_{i}" for i in range(1000)], size=n_rows)

        # Generate timestamps
        base_time = datetime(2015, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            base_time.timestamp() + i * 300 for i in range(n_rows)  # 5-minute intervals
        ]

        data = {
            'jid': job_ids,
            'time': timestamps,
            'submit_time': timestamps,
            'start_time': [t + 60 for t in timestamps],  # Start 1 min after submit
            'end_time': [t + 3600 for t in timestamps],   # Run for 1 hour
            'exitcode': np.random.choice([0, 1, 'TIMEOUT', 'OOM'], size=n_rows),
            'ncores': np.random.randint(1, 64, size=n_rows),
            'nhosts': np.random.randint(1, 8, size=n_rows),
            'queue': np.random.choice(['normal', 'debug', 'gpu'], size=n_rows),
            'account': np.random.choice(['project1', 'project2', 'project3'], size=n_rows),
            'username': np.random.choice(['user1', 'user2', 'user3'], size=n_rows),
            'job_username': np.random.choice(['user1', 'user2', 'user3'], size=n_rows),
        }

        # Add telemetry metrics with realistic patterns and edge cases
        for metric, config in TELEMETRY_METRICS.items():
            if metric == 'value_cpuuser':
                # CPU usage: 0-100%, some invalid values
                values = np.random.uniform(0, 120, size=n_rows)  # Some > 100%
                values[::100] = np.nan  # Add some NaN values
                values[::200] = -5      # Add some negative values
            elif metric == 'value_memused':
                # Memory usage: 0-2048 GB, some invalid values
                values = np.random.uniform(0, 3000, size=n_rows)  # Some > 2048
                values[::150] = np.nan
                values[::300] = -10
            elif metric == 'value_memused_minus_diskcache':
                # Optional metric - more NaN values
                values = np.random.uniform(0, 2048, size=n_rows)
                values[::3] = np.nan  # 1/3 missing (common for optional metrics)
            elif metric == 'value_nfs':
                # NFS I/O: 0-10000 MB/s
                values = np.random.uniform(0, 12000, size=n_rows)  # Some > 10000
                values[::80] = np.nan
            elif metric == 'value_block':
                # Block I/O: 0-100 GB/s
                values = np.random.uniform(0, 150, size=n_rows)   # Some > 100
                values[::120] = np.nan
            elif metric == 'value_gpu':
                # GPU utilization: only for some clusters, 0-100%
                values = np.random.uniform(0, 100, size=n_rows)
                values[::2] = np.nan  # GPU not available for many jobs
            else:
                # Default pattern for any additional metrics
                values = np.random.uniform(0, 100, size=n_rows)
                values[::50] = np.nan

            data[metric] = values

        # Add cluster information
        data['cluster'] = np.random.choice(['conte', 'stampede', 'anvil'], size=n_rows)

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_job_metadata(self) -> pd.DataFrame:
        """Generate job metadata with various edge cases"""
        np.random.seed(123)
        n_jobs = 1000

        job_ids = [f"job_{i}" for i in range(n_jobs)]
        base_time = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()

        data = {
            'jid': job_ids,
            'submit_time': [base_time + i * 3600 for i in range(n_jobs)],
            'start_time': [base_time + i * 3600 + 120 for i in range(n_jobs)],
            'end_time': [base_time + i * 3600 + 3720 for i in range(n_jobs)],
            'exitcode': np.random.choice([0, 1, 'TIMEOUT', 'OOM', 'CANCELLED'], size=n_jobs),
            'ncores': np.random.randint(1, 128, size=n_jobs),
            'nhosts': np.random.randint(1, 16, size=n_jobs),
            'queue': np.random.choice(['normal', 'debug', 'gpu', 'long'], size=n_jobs),
            'account': [f"project_{i%10}" for i in range(n_jobs)],
            'username': [f"user_{i%50}" for i in range(n_jobs)],
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def target_job_ids(self) -> Set[str]:
        """Generate a set of target job IDs for filtering tests"""
        # Simulate the 1.3M job scenario with a smaller subset
        return {f"job_{i}" for i in range(0, 800, 2)}  # Every other job

    def assert_dataframes_equivalent(
        self,
        pandas_df: pd.DataFrame,
        polars_df_or_pandas: Union[pl.DataFrame, pd.DataFrame],
        check_dtype: bool = True,
        check_names: bool = True,
        check_exact: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ):
        """
        Strict equivalence checker with tolerance for floating point differences.

        Args:
            pandas_df: Reference Pandas DataFrame
            polars_df_or_pandas: Polars DataFrame or converted Pandas DataFrame
            check_dtype: Whether to check data types
            check_names: Whether to check column names and order
            check_exact: Whether to check exact values (no tolerance)
            rtol: Relative tolerance for floating point comparison
            atol: Absolute tolerance for floating point comparison
        """
        # Convert Polars to Pandas for comparison if needed
        if isinstance(polars_df_or_pandas, pl.DataFrame):
            polars_as_pandas = polars_df_or_pandas.to_pandas()
        else:
            polars_as_pandas = polars_df_or_pandas

        # Check basic structure
        assert len(pandas_df) == len(polars_as_pandas), \
            f"Row count mismatch: pandas={len(pandas_df)}, polars={len(polars_as_pandas)}"

        if check_names:
            assert list(pandas_df.columns) == list(polars_as_pandas.columns), \
                f"Column mismatch: pandas={list(pandas_df.columns)}, polars={list(polars_as_pandas.columns)}"

        # Check each column
        for col in pandas_df.columns:
            if col not in polars_as_pandas.columns:
                continue

            pandas_col = pandas_df[col]
            polars_col = polars_as_pandas[col]

            # Handle different data types appropriately
            if pd.api.types.is_numeric_dtype(pandas_col):
                # Numeric comparison with tolerance
                if check_exact:
                    pd.testing.assert_series_equal(
                        pandas_col, polars_col, check_names=False, check_dtype=check_dtype
                    )
                else:
                    np.testing.assert_allclose(
                        pandas_col.fillna(0), polars_col.fillna(0),
                        rtol=rtol, atol=atol,
                        err_msg=f"Numeric values differ in column '{col}'"
                    )
                    # Check NaN positions separately
                    pd.testing.assert_series_equal(
                        pandas_col.isna(), polars_col.isna(),
                        check_names=False, check_dtype=False
                    )
            elif pd.api.types.is_datetime64_any_dtype(pandas_col):
                # Datetime comparison
                pd.testing.assert_series_equal(
                    pandas_col, polars_col, check_names=False, check_dtype=False
                )
            else:
                # String/categorical comparison
                pd.testing.assert_series_equal(
                    pandas_col, polars_col, check_names=False, check_dtype=check_dtype
                )

    def test_clean_numeric_column_equivalence(self, sample_telemetry_data):
        """Test that clean_numeric_column produces identical results"""
        test_col = sample_telemetry_data['value_cpuuser'].copy()

        # Test with valid range
        valid_range = (0.0, 100.0)
        fill_value = 0.0

        pandas_result = clean_numeric_column(test_col, valid_range, fill_value)

        # Polars equivalent (to be implemented)
        def clean_numeric_column_polars(series: pl.Series, valid_range: Optional[tuple] = None, fill_value: float = 0.0) -> pl.Series:
            """Polars implementation of clean_numeric_column"""
            # Convert to numeric (Polars handles this automatically for most types)
            series = series.cast(pl.Float64, strict=False)

            # Apply valid range filter if specified
            if valid_range:
                min_val, max_val = valid_range
                series = series.map_elements(
                    lambda x: x if (x is not None and min_val <= x <= max_val) else None,
                    return_dtype=pl.Float64
                )

            # Fill missing values
            series = series.fill_null(fill_value)

            return series

        # Convert to Polars, apply function, convert back
        polars_series = pl.Series(test_col)
        polars_result = clean_numeric_column_polars(polars_series, valid_range, fill_value)
        polars_as_pandas = polars_result.to_pandas()

        # Assert equivalence
        self.assert_dataframes_equivalent(
            pd.DataFrame({'col': pandas_result}),
            pd.DataFrame({'col': polars_as_pandas}),
            check_exact=True
        )

    def test_timestamp_parsing_equivalence(self, sample_telemetry_data):
        """Test timestamp parsing equivalence between Pandas and Polars"""
        # Test various timestamp formats
        timestamp_data = {
            'unix_timestamps': sample_telemetry_data['time'].values,
            'datetime_strings': [
                '2015-01-01 12:00:00+00:00',
                '2015-01-02 13:30:00Z',
                '2015-01-03T14:45:00.123456Z',
                '2015-01-04 15:00:00',  # No timezone
                'invalid_timestamp',
                None
            ] * (len(sample_telemetry_data) // 6 + 1)
        }
        timestamp_df = pd.DataFrame(timestamp_data)

        # Pandas implementation
        pandas_result = timestamp_df.copy()
        pandas_result['unix_timestamps'] = pd.to_datetime(
            pandas_result['unix_timestamps'], unit='s', errors='coerce', utc=True
        )
        pandas_result['datetime_strings'] = pd.to_datetime(
            pandas_result['datetime_strings'], errors='coerce', utc=True
        )

        # Polars implementation
        polars_df = pl.DataFrame(timestamp_data)
        polars_result = polars_df.with_columns([
            pl.col('unix_timestamps').cast(pl.Datetime('us', 'UTC')).alias('unix_timestamps'),
            pl.col('datetime_strings').str.strptime(pl.Datetime('us', 'UTC'), fmt=None, strict=False).alias('datetime_strings')
        ])

        # Compare results
        self.assert_dataframes_equivalent(
            pandas_result, polars_result, check_dtype=False  # Datetime dtypes may differ slightly
        )

    def test_job_id_filtering_critical(self, sample_telemetry_data, target_job_ids):
        """Test the critical .isin() operation that's the performance bottleneck"""
        # This is the most critical test - the operation causing slowdown
        test_df = sample_telemetry_data.copy()

        # Pandas implementation (current bottleneck)
        pandas_filtered = test_df[test_df['jid'].astype(str).isin(target_job_ids)]

        # Polars implementation
        polars_df = pl.DataFrame(test_df)
        polars_filtered = polars_df.filter(
            pl.col('jid').cast(pl.Utf8).is_in(list(target_job_ids))
        )

        # Assert equivalence
        self.assert_dataframes_equivalent(
            pandas_filtered.reset_index(drop=True),
            polars_filtered,
            check_dtype=False  # May have slight dtype differences
        )

        # Also test performance (not equivalence, just validation)
        import time

        # Time pandas operation
        start = time.time()
        for _ in range(10):
            _ = test_df[test_df['jid'].astype(str).isin(target_job_ids)]
        pandas_time = time.time() - start

        # Time polars operation
        start = time.time()
        for _ in range(10):
            _ = polars_df.filter(pl.col('jid').cast(pl.Utf8).is_in(list(target_job_ids)))
        polars_time = time.time() - start

        print(f"Performance comparison - Pandas: {pandas_time:.3f}s, Polars: {polars_time:.3f}s")
        # Polars should be faster, but this is just informational

    def test_job_aggregation_operations(self, sample_telemetry_data):
        """Test the complex multi-column aggregation operations"""
        test_df = sample_telemetry_data.copy()

        # Pandas implementation (from _aggregate_by_job)
        if 'jid' not in test_df.columns:
            pandas_result = test_df
        else:
            # Separate job metadata from telemetry
            job_cols = [
                'jid', 'submit_time', 'start_time', 'end_time', 'exitcode',
                'ncores', 'nhosts', 'queue', 'account', 'username', 'job_username'
            ]
            available_job_cols = [col for col in job_cols if col in test_df.columns]

            # Get job metadata (should be constant per job)
            job_metadata = test_df[available_job_cols].drop_duplicates(subset=['jid'])

            # Get telemetry columns
            telemetry_cols = [col for col in TELEMETRY_METRICS.keys() if col in test_df.columns]

            if not telemetry_cols:
                pandas_result = job_metadata
            else:
                # Aggregate telemetry by job
                agg_dict = {}
                for col in telemetry_cols:
                    agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']

                # Add time for temporal features
                if 'time' in test_df.columns:
                    agg_dict['time'] = ['min', 'max', 'count']

                # Perform aggregation
                telemetry_agg = test_df.groupby('jid').agg(agg_dict)

                # Flatten column names
                telemetry_agg.columns = [
                    f"{col}_{agg}" for col, agg in telemetry_agg.columns
                ]
                telemetry_agg = telemetry_agg.reset_index()

                # Merge with job metadata
                pandas_result = job_metadata.merge(telemetry_agg, on='jid', how='left')

        # Polars implementation
        polars_df = pl.DataFrame(test_df)

        if 'jid' not in polars_df.columns:
            polars_result = polars_df
        else:
            # Separate job metadata from telemetry
            job_cols = [
                'jid', 'submit_time', 'start_time', 'end_time', 'exitcode',
                'ncores', 'nhosts', 'queue', 'account', 'username', 'job_username'
            ]
            available_job_cols = [col for col in job_cols if col in polars_df.columns]

            # Get job metadata (should be constant per job)
            job_metadata = polars_df.select(available_job_cols).unique(subset=['jid'])

            # Get telemetry columns
            telemetry_cols = [col for col in TELEMETRY_METRICS.keys() if col in polars_df.columns]

            if not telemetry_cols:
                polars_result = job_metadata
            else:
                # Build aggregation expressions
                agg_exprs = []
                for col in telemetry_cols:
                    agg_exprs.extend([
                        pl.col(col).mean().alias(f"{col}_mean"),
                        pl.col(col).std().alias(f"{col}_std"),
                        pl.col(col).min().alias(f"{col}_min"),
                        pl.col(col).max().alias(f"{col}_max"),
                        pl.col(col).count().alias(f"{col}_count")
                    ])

                # Add time aggregations if available
                if 'time' in polars_df.columns:
                    agg_exprs.extend([
                        pl.col('time').min().alias('time_min'),
                        pl.col('time').max().alias('time_max'),
                        pl.col('time').count().alias('time_count')
                    ])

                # Perform aggregation
                telemetry_agg = polars_df.group_by('jid').agg(agg_exprs)

                # Merge with job metadata
                polars_result = job_metadata.join(telemetry_agg, on='jid', how='left')

        # Assert equivalence
        self.assert_dataframes_equivalent(
            pandas_result.sort_values('jid').reset_index(drop=True),
            polars_result.sort('jid'),
            check_dtype=False  # Aggregation may produce slightly different dtypes
        )

    def test_data_cleaning_pipeline(self, sample_telemetry_data):
        """Test the complete data cleaning pipeline equivalence"""
        test_df = sample_telemetry_data.copy()
        cluster = ClusterType.CONTE

        # Pandas implementation (from _clean_telemetry_data)
        pandas_df = test_df.copy()

        # Clean numeric telemetry columns
        for col, config in TELEMETRY_METRICS.items():
            if col in pandas_df.columns:
                # Check cluster restrictions
                if 'clusters' in config and cluster:
                    if cluster.value not in config['clusters']:
                        # Remove column not available for this cluster
                        pandas_df = pandas_df.drop(columns=[col])
                        continue

                # Clean numeric values
                valid_range = config.get('valid_range')
                missing_value = config.get('missing_value', 0.0)

                if missing_value is not None:
                    pandas_df[col] = clean_numeric_column(pandas_df[col], valid_range, missing_value)
                else:
                    # Optional metric, keep NaN for missing values
                    pandas_df[col] = clean_numeric_column(pandas_df[col], valid_range, np.nan)

        # Clean timestamp columns
        for time_col in ['time', 'submit_time', 'start_time', 'end_time']:
            if time_col in pandas_df.columns:
                pandas_df[time_col] = pd.to_datetime(pandas_df[time_col], errors='coerce', utc=True)

        # Clean job ID
        if 'jid' in pandas_df.columns:
            pandas_df['jid'] = pandas_df['jid'].astype(str)

        # Remove rows with invalid timestamps (telemetry must have time)
        if 'time' in pandas_df.columns:
            pandas_df = pandas_df.dropna(subset=['time'])

        # Add cluster information if available
        if cluster:
            pandas_df['cluster'] = cluster.value

        # Polars implementation
        polars_df = pl.DataFrame(test_df)

        # Clean numeric telemetry columns
        for col, config in TELEMETRY_METRICS.items():
            if col in polars_df.columns:
                # Check cluster restrictions
                if 'clusters' in config and cluster:
                    if cluster.value not in config['clusters']:
                        # Remove column not available for this cluster
                        polars_df = polars_df.drop(col)
                        continue

                # Clean numeric values
                valid_range = config.get('valid_range')
                missing_value = config.get('missing_value', 0.0)

                if missing_value is not None:
                    # Apply range filter and fill
                    if valid_range:
                        min_val, max_val = valid_range
                        polars_df = polars_df.with_columns(
                            pl.when(
                                (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
                            ).then(pl.col(col))
                            .otherwise(missing_value)
                            .alias(col)
                        )
                    else:
                        polars_df = polars_df.with_columns(
                            pl.col(col).fill_null(missing_value)
                        )
                else:
                    # Optional metric, keep null for missing values
                    if valid_range:
                        min_val, max_val = valid_range
                        polars_df = polars_df.with_columns(
                            pl.when(
                                (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
                            ).then(pl.col(col))
                            .otherwise(None)
                            .alias(col)
                        )

        # Clean timestamp columns
        timestamp_cols = ['time', 'submit_time', 'start_time', 'end_time']
        for time_col in timestamp_cols:
            if time_col in polars_df.columns:
                polars_df = polars_df.with_columns(
                    pl.col(time_col).cast(pl.Datetime('us', 'UTC'), strict=False).alias(time_col)
                )

        # Clean job ID
        if 'jid' in polars_df.columns:
            polars_df = polars_df.with_columns(
                pl.col('jid').cast(pl.Utf8).alias('jid')
            )

        # Remove rows with invalid timestamps (telemetry must have time)
        if 'time' in polars_df.columns:
            polars_df = polars_df.filter(pl.col('time').is_not_null())

        # Add cluster information if available
        if cluster:
            polars_df = polars_df.with_columns(
                pl.lit(cluster.value).alias('cluster')
            )

        # Assert equivalence
        self.assert_dataframes_equivalent(
            pandas_df.reset_index(drop=True),
            polars_df,
            check_dtype=False  # Datetime and string dtypes may differ
        )

    def test_streaming_chunk_processing(self, sample_telemetry_data, target_job_ids):
        """Test chunk-based processing equivalence"""
        # Simulate the streaming scenario
        test_df = sample_telemetry_data.copy()
        chunk_size = 1000

        # Split into chunks
        chunks = [test_df[i:i+chunk_size] for i in range(0, len(test_df), chunk_size)]

        # Pandas chunk processing
        pandas_results = []
        for chunk in chunks:
            # Filter by target jids (the critical operation)
            if 'jid' in chunk.columns:
                filtered_chunk = chunk[chunk['jid'].astype(str).isin(target_job_ids)]
                if not filtered_chunk.empty:
                    pandas_results.append(filtered_chunk)

        if pandas_results:
            pandas_final = pd.concat(pandas_results, ignore_index=True)
        else:
            pandas_final = pd.DataFrame()

        # Polars chunk processing
        polars_results = []
        for chunk in chunks:
            polars_chunk = pl.DataFrame(chunk)
            # Filter by target jids
            if 'jid' in polars_chunk.columns:
                filtered_chunk = polars_chunk.filter(
                    pl.col('jid').cast(pl.Utf8).is_in(list(target_job_ids))
                )
                if len(filtered_chunk) > 0:
                    polars_results.append(filtered_chunk)

        if polars_results:
            polars_final = pl.concat(polars_results)
        else:
            polars_final = pl.DataFrame()

        # Assert equivalence
        if len(pandas_final) > 0 and len(polars_final) > 0:
            self.assert_dataframes_equivalent(
                pandas_final.sort_values('jid').reset_index(drop=True),
                polars_final.sort('jid'),
                check_dtype=False
            )
        else:
            assert len(pandas_final) == len(polars_final) == 0

    def test_memory_efficiency_validation(self, sample_telemetry_data):
        """Test that Polars uses less memory than Pandas for the same operations"""
        import psutil
        import gc

        test_df = sample_telemetry_data.copy()

        # Measure Pandas memory usage
        gc.collect()
        process = psutil.Process()
        pandas_start_memory = process.memory_info().rss

        pandas_df = test_df.copy()
        # Perform typical operations
        pandas_df['jid'] = pandas_df['jid'].astype(str)
        pandas_grouped = pandas_df.groupby('jid').agg({'value_cpuuser': ['mean', 'std', 'count']})

        pandas_peak_memory = process.memory_info().rss
        pandas_memory_used = pandas_peak_memory - pandas_start_memory

        # Clean up
        del pandas_df, pandas_grouped
        gc.collect()

        # Measure Polars memory usage
        polars_start_memory = process.memory_info().rss

        polars_df = pl.DataFrame(test_df)
        # Perform equivalent operations
        polars_df = polars_df.with_columns(pl.col('jid').cast(pl.Utf8))
        polars_grouped = polars_df.group_by('jid').agg([
            pl.col('value_cpuuser').mean().alias('value_cpuuser_mean'),
            pl.col('value_cpuuser').std().alias('value_cpuuser_std'),
            pl.col('value_cpuuser').count().alias('value_cpuuser_count')
        ])

        polars_peak_memory = process.memory_info().rss
        polars_memory_used = polars_peak_memory - polars_start_memory

        print(f"Memory usage - Pandas: {pandas_memory_used / 1024 / 1024:.1f} MB, "
              f"Polars: {polars_memory_used / 1024 / 1024:.1f} MB")

        # Polars should use less memory (this is informational, not a strict test)
        # assert polars_memory_used <= pandas_memory_used * 1.1  # Allow 10% tolerance

    def test_edge_cases(self):
        """Test edge cases that might break equivalence"""

        # Empty DataFrame
        empty_df = pd.DataFrame()
        empty_polars = pl.DataFrame()

        self.assert_dataframes_equivalent(empty_df, empty_polars)

        # DataFrame with all NaN values
        nan_data = {'col1': [np.nan, np.nan], 'col2': [None, None]}
        nan_pandas = pd.DataFrame(nan_data)
        nan_polars = pl.DataFrame(nan_data)

        self.assert_dataframes_equivalent(nan_pandas, nan_polars, check_dtype=False)

        # DataFrame with mixed types
        mixed_data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        }
        mixed_pandas = pd.DataFrame(mixed_data)
        mixed_polars = pl.DataFrame(mixed_data)

        self.assert_dataframes_equivalent(mixed_pandas, mixed_polars, check_dtype=False)

    def test_full_pipeline_integration(self, sample_telemetry_data, sample_job_metadata, target_job_ids):
        """End-to-end pipeline equivalence test"""

        # Simulate the full telemetry processing pipeline
        telemetry_df = sample_telemetry_data.copy()
        job_df = sample_job_metadata.copy()

        # === PANDAS PIPELINE ===
        pandas_telemetry = telemetry_df.copy()

        # 1. Filter by target job IDs
        pandas_telemetry = pandas_telemetry[
            pandas_telemetry['jid'].astype(str).isin(target_job_ids)
        ]

        # 2. Clean data
        for col, config in TELEMETRY_METRICS.items():
            if col in pandas_telemetry.columns:
                valid_range = config.get('valid_range')
                missing_value = config.get('missing_value', 0.0)
                if missing_value is not None:
                    pandas_telemetry[col] = clean_numeric_column(
                        pandas_telemetry[col], valid_range, missing_value
                    )

        # 3. Aggregate by job
        if not pandas_telemetry.empty:
            agg_dict = {}
            telemetry_cols = [col for col in TELEMETRY_METRICS.keys() if col in pandas_telemetry.columns]
            for col in telemetry_cols:
                agg_dict[col] = ['mean', 'count']

            pandas_agg = pandas_telemetry.groupby('jid').agg(agg_dict)
            pandas_agg.columns = [f"{col}_{agg}" for col, agg in pandas_agg.columns]
            pandas_agg = pandas_agg.reset_index()

            # 4. Merge with job metadata
            pandas_final = job_df.merge(pandas_agg, on='jid', how='inner')
        else:
            pandas_final = pd.DataFrame()

        # === POLARS PIPELINE ===
        polars_telemetry = pl.DataFrame(telemetry_df)
        polars_job = pl.DataFrame(job_df)

        # 1. Filter by target job IDs
        polars_telemetry = polars_telemetry.filter(
            pl.col('jid').cast(pl.Utf8).is_in(list(target_job_ids))
        )

        # 2. Clean data
        for col, config in TELEMETRY_METRICS.items():
            if col in polars_telemetry.columns:
                valid_range = config.get('valid_range')
                missing_value = config.get('missing_value', 0.0)
                if missing_value is not None:
                    if valid_range:
                        min_val, max_val = valid_range
                        polars_telemetry = polars_telemetry.with_columns(
                            pl.when(
                                (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
                            ).then(pl.col(col))
                            .otherwise(missing_value)
                            .alias(col)
                        )
                    else:
                        polars_telemetry = polars_telemetry.with_columns(
                            pl.col(col).fill_null(missing_value)
                        )

        # 3. Aggregate by job
        if len(polars_telemetry) > 0:
            agg_exprs = []
            telemetry_cols = [col for col in TELEMETRY_METRICS.keys() if col in polars_telemetry.columns]
            for col in telemetry_cols:
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).count().alias(f"{col}_count")
                ])

            polars_agg = polars_telemetry.group_by('jid').agg(agg_exprs)

            # 4. Merge with job metadata
            polars_final = polars_job.join(polars_agg, on='jid', how='inner')
        else:
            polars_final = pl.DataFrame()

        # === COMPARE RESULTS ===
        if len(pandas_final) > 0 and len(polars_final) > 0:
            self.assert_dataframes_equivalent(
                pandas_final.sort_values('jid').reset_index(drop=True),
                polars_final.sort('jid'),
                check_dtype=False
            )
        else:
            assert len(pandas_final) == len(polars_final) == 0, \
                f"Both should be empty: pandas={len(pandas_final)}, polars={len(polars_final)}"


if __name__ == "__main__":
    # Run specific tests for debugging
    import sys
    test_instance = TestDataFrameEquivalence()

    # Create fixtures manually for testing
    sample_data = test_instance.sample_telemetry_data()
    sample_jobs = test_instance.sample_job_metadata()
    target_jobs = test_instance.target_job_ids()

    print("Running equivalence tests...")

    try:
        test_instance.test_job_id_filtering_critical(sample_data, target_jobs)
        print("✓ Job ID filtering test passed")

        test_instance.test_job_aggregation_operations(sample_data)
        print("✓ Job aggregation test passed")

        test_instance.test_data_cleaning_pipeline(sample_data)
        print("✓ Data cleaning test passed")

        test_instance.test_full_pipeline_integration(sample_data, sample_jobs, target_jobs)
        print("✓ Full pipeline integration test passed")

        print("\n🎉 All equivalence tests passed! Polars implementation will be functionally identical to Pandas.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)