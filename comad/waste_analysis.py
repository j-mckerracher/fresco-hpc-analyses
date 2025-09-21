"""
FRESCO HPC Resource Waste Analysis Script

This script analyzes resource waste in HPC jobs using the FRESCO dataset
to answer RQ2: "How prevalent and severe is resource waste across different
job types and user behaviors?"

This version has been optimized for maximum performance using multiprocessing
with early aggregation to dramatically reduce memory usage and processing time.
"""
from scipy import stats as scipy_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
from datetime import datetime
from tqdm import tqdm
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def _collect_all_job_ids(files):
    ids = set()
    for fp in tqdm(files, desc="Enumerating job IDs"):
        try:
            df = pd.read_parquet(fp, columns=['jid'])
            if 'jid' in df.columns:
                ids.update(df['jid'].dropna().unique())
        except Exception:
            continue
    return ids

def _sample_job_ids(files, sample_fraction, rng=42):
    all_ids = _collect_all_job_ids(files)
    n = int(len(all_ids) * sample_fraction)
    rng = np.random.default_rng(rng)
    return set(rng.choice(list(all_ids), size=n, replace=False))



def _process_and_aggregate_file(file_with_sys, sampled_job_ids=None):
    """
    Process a single file and return aggregated job-level data.
    This function runs in a separate process for true parallelism.

    Args:
        file_with_sys (tuple[Path,str] | Path): (parquet path, system label) or just a Path
        sampled_job_ids (set, optional): Set of job IDs to filter by

    Returns:
        pd.DataFrame or None: Aggregated job data or None if error/empty
    """
    try:
        # Support both (Path, system_label) and Path
        if isinstance(file_with_sys, tuple):
            file_path, system_label = file_with_sys
        else:
            file_path, system_label = file_with_sys, None

        df = pd.read_parquet(file_path)
        if df.empty:
            return None

        df = _clean_dataframe(df)
        if df.empty:
            return None

        # Attach system label from filename-derived input
        if system_label is not None:
            df['system'] = system_label

        # Apply job sampling if provided
        if sampled_job_ids and 'jid' in df.columns:
            df = df[df['jid'].isin(sampled_job_ids)]
            if df.empty:
                return None

        # Aggregate to job-level right away
        job_aggregations = {
            'start_time': 'first',
            'end_time': 'first',
            'job_exitcode': 'first',
            'queue': 'first',
            'job_username': 'first',
            'ncores': 'first',
            'nhosts': 'first',
            'duration_hours': 'first',
            'job_cpu_usage': ['mean', 'count'],
            'value_memused': ['mean', 'count'],
            'system': 'first'  # keep the system label with the job
        }

        aggregated_df = df.groupby('jid').agg(job_aggregations)

        # Flatten multi-index columns
        aggregated_df.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col
            for col in aggregated_df.columns.values
        ]
        aggregated_df = aggregated_df.reset_index()

        return aggregated_df

    except Exception as e:
        logger.warning(f"Could not process {file_with_sys}: {e}")
        return None



def _clean_dataframe(df):
    """
    Clean and standardize a single dataframe.
    Optimized version with early returns for better performance.

    Args:
        df (pd.DataFrame): Raw dataframe from parquet file

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df

    # Detect data format based on content (heuristic)
    is_2013_format = False
    if 'jobname' in df.columns and 'exitcode' in df.columns:
        jobname_sample = df['jobname'].dropna().head(5)
        exitcode_sample = df['exitcode'].dropna().head(5)
        if len(jobname_sample) > 0 and len(exitcode_sample) > 0:
            # 2013 format: jobname has COMPLETED/FAILED; exitcode looks like {NODE...}
            if any(str(val) in ['COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED'] for val in jobname_sample):
                if any(str(val).startswith('{') for val in exitcode_sample):
                    is_2013_format = True

    # Column mapping
    if is_2013_format:
        # Exitcode / username
        if 'jobname' in df.columns:
            df['job_exitcode'] = df['jobname']
        if 'username' in df.columns:
            df['job_username'] = df['username']

        # CPU usage: pick a valid numeric column
        cpu_cols = [c for c in ['value_cpuuser', 'cpu_user', 'cpu_user_pct'] if c in df.columns]
        if cpu_cols:
            df['job_cpu_usage'] = pd.to_numeric(df[cpu_cols[0]], errors='coerce')
    else:
        # Newer (2023+) format
        if 'exitcode' in df.columns:
            df['job_exitcode'] = df['exitcode']
        if 'username' in df.columns:
            df['job_username'] = df['username']
        if 'value_cpuuser' in df.columns:
            df['job_cpu_usage'] = pd.to_numeric(df['value_cpuuser'], errors='coerce')

    # Required columns
    required_columns = [
        'time', 'start_time', 'end_time', 'job_exitcode', 'queue',
        'job_username', 'job_cpu_usage', 'value_memused', 'ncores', 'nhosts', 'jid'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return pd.DataFrame()  # Early return if we can't proceed

    # Type conversions
    try:
        for col in ['time', 'start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col])
        for col in ['job_cpu_usage', 'value_memused', 'ncores', 'nhosts']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception:
        return pd.DataFrame()

    # Duration
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['duration_hours'] = df['duration_seconds'] / 3600

    # Valid rows
    valid_mask = (
        (df['duration_seconds'] > 0) &
        (df['job_cpu_usage'].notna()) &
        (df['value_memused'].notna()) &
        (df['ncores'] > 0)
    )
    return df[valid_mask]



class FrescoResourceWasteAnalyzer:
    """
    Optimized analyzer for resource waste in HPC jobs using FRESCO dataset.

    This version uses multiprocessing with early aggregation for maximum performance.
    """

    def __init__(self, data_dir="/home/fresco-analysis/data/", num_workers=None):
        """
        Initialize the analyzer with data directory path.

        Args:
            data_dir (str): Path to the FRESCO data chunks directory.
            num_workers (int, optional): Number of worker processes for file processing.
                                        Defaults to the number of CPU cores.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./fresco_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        # Initialize data containers
        self.job_data = None
        self.waste_metrics = None

        # Performance metrics tracking
        self.total_jobs_processed = 0
        self.total_files_processed = 0

        logger.info(f"Initialized Optimized FRESCO Resource Waste Analyzer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Process pool size configured to {self.num_workers}")

    def _enumerate_job_ids_from_file(self, file_path):
        """
        Read a single Parquet file and return the set of unique job IDs it contains.
        Reads only the 'jid' column for efficiency.
        """
        try:
            df = pd.read_parquet(file_path, columns=['jid'])
            if 'jid' not in df.columns or df.empty:
                return set()
            # Drop NA and convert to Python types to prevent dtype surprises
            return set(df['jid'].dropna().astype(object).unique().tolist())
        except Exception as e:
            logger.warning(f"Failed to enumerate jids from {file_path}: {e}")
            return set()

    def _collect_all_job_ids_parallel(self, file_paths):
        """
        Collect the global set of unique job IDs across ALL files in parallel.
        NOTE: This holds the set of unique jids in memory.
        """
        logger.info(f"Enumerating job IDs from {len(file_paths)} files (jid column only)...")
        all_ids = set()
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {ex.submit(self._enumerate_job_ids_from_file, fp): fp for fp in file_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Enumerating jids"):
                try:
                    part_ids = fut.result()
                    if part_ids:
                        all_ids.update(part_ids)
                except Exception as e:
                    logger.warning(f"Enumeration task failed: {e}")
        logger.info(f"Enumerated {len(all_ids):,} unique job IDs")
        return all_ids

    def _sample_job_ids_srs(self, files_with_system, sample_fraction, random_state=42, min_samples=1):
        """
        True SRS of jobs:
          1) Enumerate ALL unique job IDs across the provided files.
          2) Draw an exact simple random sample of size floor(fraction * N).

        Args:
            files_with_system (list[tuple[Path,str]]): (file, system) tuples
            sample_fraction (float): 0 < fraction <= 1
            random_state (int): RNG seed
            min_samples (int): lower bound to avoid empty samples in tiny tests

        Returns:
            set: sampled job IDs
        """
        assert 0 < sample_fraction <= 1.0, "sample_fraction must be in (0, 1]"
        file_paths = [fp for (fp, _sys) in files_with_system]
        all_ids = self._collect_all_job_ids_parallel(file_paths)
        N = len(all_ids)
        if N == 0:
            logger.warning("No job IDs found during enumeration; returning empty sample.")
            return set()

        k = max(int(np.floor(sample_fraction * N)), min_samples)
        if k >= N:
            logger.info("Requested sample >= population size; taking all job IDs.")
            return set(all_ids)

        rng = np.random.default_rng(random_state)
        # numpy choice on a list for reproducibility and speed
        sampled = rng.choice(list(all_ids), size=k, replace=False)
        sampled_set = set(sampled.tolist())
        logger.info(f"SRS selected {len(sampled_set):,} jobs out of {N:,} ({sample_fraction:.1%})")
        return sampled_set

    def discover_data_files(self, limit_years=None, limit_files_per_day=None, hpc_system=None):
        """
        Discover all parquet files in the data directory, optionally filtered by HPC system.

        Args:
            limit_years (list): List of years to limit analysis to (e.g., [2013, 2014])
            limit_files_per_day (int): Limit number of files per day for testing
            hpc_system (str): Filter by HPC system. Options:
                - 'stampede': Files containing '_S' in filename
                - 'conte':    Files containing '_C' in filename
                - 'anvil':    Files without '_S' or '_C' in filename
                - 'all' or None: All files (no filtering)

        Returns:
            list[tuple[Path,str]]: List of (file path, system label) tuples
        """
        logger.info(f"Discovering data files for HPC system: {hpc_system or 'all'}...")
        files = []

        for year_dir in sorted(self.data_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            if limit_years and int(year) not in limit_years:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue

                    day_files = list(day_dir.glob("*.parquet"))

                    # Filter by HPC system (using filename)
                    if hpc_system:
                        hsys = hpc_system.lower()
                        if hsys == 'stampede':
                            day_files = [f for f in day_files if '_S' in f.name]
                        elif hsys == 'conte':
                            day_files = [f for f in day_files if '_C' in f.name]
                        elif hsys == 'anvil':
                            day_files = [f for f in day_files if ('_S' not in f.name and '_C' not in f.name)]
                        elif hsys == 'all':
                            pass  # keep all
                        else:
                            logger.warning(f"Unknown HPC system '{hpc_system}'. Valid: stampede|conte|anvil|all")

                    if limit_files_per_day:
                        day_files = day_files[:limit_files_per_day]

                    # Attach system label by filename
                    for f in day_files:
                        sys = 'stampede' if '_S' in f.name else ('conte' if '_C' in f.name else 'anvil')
                        files.append((f, sys))

        logger.info(f"Discovered {len(files)} data files for HPC system: {hpc_system or 'all'}")
        return files

    def _efficient_job_sampling(self, files, sample_fraction):
        """
        More efficient job sampling that doesn't require reading all files twice.

        Args:
            files (list): List of file paths
            sample_fraction (float): Fraction of jobs to sample

        Returns:
            set: Set of sampled job IDs
        """
        logger.info(f"Collecting job IDs for {sample_fraction:.1%} sampling...")

        # Sample a subset of files to get job IDs from (much more efficient)
        sample_files = files[::max(1, len(files) // 20)]  # Take every Nth file for sampling
        all_job_ids = set()

        for file_path in tqdm(sample_files, desc="Collecting job IDs"):
            try:
                df = pd.read_parquet(file_path)
                df = _clean_dataframe(df)
                if not df.empty and 'jid' in df.columns:
                    all_job_ids.update(df['jid'].unique())
            except Exception as e:
                logger.warning(f"Error reading {file_path} for job ID collection: {e}")
                continue

        # Sample the job IDs
        sampled_job_ids = set(pd.Series(list(all_job_ids)).sample(
            frac=sample_fraction, random_state=42
        ).tolist())

        logger.info(f"Sampled {len(sampled_job_ids)} jobs from {len(all_job_ids)} total jobs")
        return sampled_job_ids

    def load_and_process_parallel(self, files, sample_jobs_fraction=None):
        """
        Load and process files in parallel with early aggregation.
        Args:
            files (list[tuple[Path,str]]): (file, system) tuples from discover_data_files
            sample_jobs_fraction (float): Fraction of jobs to sample (0<frac<=1)
        Returns:
            pd.DataFrame: Aggregated job-level data
        """
        logger.info(f"Processing {len(files)} files in parallel using {self.num_workers} workers...")

        sampled_job_ids = None
        if sample_jobs_fraction and sample_jobs_fraction < 1.0:
            sampled_job_ids = self._sample_job_ids_srs(files, sample_jobs_fraction, random_state=42)

        pre_aggregated_results = []
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            process_func = partial(_process_and_aggregate_file, sampled_job_ids=sampled_job_ids)
            future_to_file = {executor.submit(process_func, fws): fws for fws in files}

            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        pre_aggregated_results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Process failed for {file_path}: {e}")

        # Count successfully processed shards (post-filter)
        self.total_files_processed = len(pre_aggregated_results)
        if not pre_aggregated_results:
            raise ValueError("No data could be processed from the specified files")

        logger.info(f"Successfully processed {self.total_files_processed} shard aggregates")
        logger.info("Combining pre-aggregated results...")

        combined_df = pd.concat(pre_aggregated_results, ignore_index=True)
        gc.collect()

        logger.info("Performing final aggregation for jobs spanning multiple files...")

        def weighted_mean(group, value_col, weight_col):
            weights = group[weight_col]
            denom = weights.sum()
            if denom == 0:
                return 0.0
            return (group[value_col] * weights).sum() / denom

        # Final job-level aggregation across shards:
        # Use min(start), max(end) and recompute duration; carry a single system label
        final_agg = combined_df.groupby('jid').apply(
            lambda g: pd.Series({
                'start_time': g['start_time_first'].min(),
                'end_time': g['end_time_first'].max(),
                'job_exitcode': g['job_exitcode_first'].iloc[0],
                'queue': g['queue_first'].iloc[0],
                'job_username': g['job_username_first'].iloc[0],
                'ncores': g['ncores_first'].iloc[0],
                'nhosts': g['nhosts_first'].iloc[0],
                'job_cpu_usage': weighted_mean(g, 'job_cpu_usage_mean', 'job_cpu_usage_count'),
                'value_memused': weighted_mean(g, 'value_memused_mean', 'value_memused_count'),
                'system': g['system_first'].iloc[0]
            })
        ).reset_index()

        # Recompute duration
        final_agg['duration_hours'] = (final_agg['end_time'] - final_agg['start_time']).dt.total_seconds() / 3600.0

        self.total_jobs_processed = len(final_agg)
        logger.info(f"Final aggregated data contains {self.total_jobs_processed:,} unique jobs")

        return final_agg

    def calculate_resource_waste(self, job_df):
        """
        Calculate resource waste metrics on already-aggregated job data.

        Args:
            job_df (pd.DataFrame): Aggregated job-level data

        Returns:
            pd.DataFrame: Job data with waste metrics added
        """
        logger.info("Calculating resource waste metrics...")
        job_df = job_df.copy()

        # CPU waste
        job_df['cpu_waste'] = 1.0 - (job_df['job_cpu_usage'] / 100.0)
        job_df['cpu_waste'] = job_df['cpu_waste'].clip(0, 1)

        # Ensure system column exists; if not, fallback to queue-based inference
        if 'system' not in job_df.columns:
            job_df['system'] = job_df['queue'].apply(
                lambda x: 'stampede' if '_S' in str(x) else ('conte' if '_C' in str(x) else 'anvil')
            )

        # System-specific memory per core (GB)
        memory_per_core = {
            'stampede': 2.0,  # ~32GB / 16-core node
            'conte': 4.0,  # ~64GB / 16-core node
            'anvil': 2.0  # ~256GB / 128-core node
        }

        job_df['mem_per_core'] = job_df['system'].map(memory_per_core)
        job_df['estimated_requested_mem_gb'] = job_df['ncores'] * job_df['mem_per_core']

        # Memory waste
        job_df['mem_waste'] = 1.0 - (job_df['value_memused'] / job_df['estimated_requested_mem_gb'])
        job_df['mem_waste'] = job_df['mem_waste'].clip(0, 1)

        # Composite waste (equal-weight)
        job_df['composite_waste'] = 0.5 * job_df['cpu_waste'] + 0.5 * job_df['mem_waste']

        # Economic impact
        job_df['cpu_hours_wasted'] = job_df['cpu_waste'] * job_df['ncores'] * job_df['duration_hours']
        job_df['mem_gb_hours_wasted'] = job_df['mem_waste'] * job_df['estimated_requested_mem_gb'] * job_df[
            'duration_hours']

        # Duration/size/waste categories
        job_df['duration_category'] = pd.cut(
            job_df['duration_hours'],
            bins=[0, 1, 8, 24, float('inf')],
            labels=['Short (<1h)', 'Medium (1-8h)', 'Long (8-24h)', 'Very Long (>24h)']
        )
        job_df['size_category'] = pd.cut(
            job_df['ncores'],
            bins=[0, 16, 64, 256, float('inf')],
            labels=['Small (1-16)', 'Medium (17-64)', 'Large (65-256)', 'Very Large (>256)']
        )
        job_df['waste_category'] = pd.cut(
            job_df['composite_waste'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
        )

        logger.info(f"Resource waste calculation completed for {len(job_df)} jobs")
        return job_df

    def generate_statistical_summaries(self, df):
        """
        Generate comprehensive statistical summaries of resource waste with significance tests.

        Args:
            df (pd.DataFrame): Job data with waste metrics

        Returns:
            dict: Dictionary of summary statistics
        """
        logger.info("Generating statistical summaries...")

        stats = {}

        # RQ2 Severity Analysis: Overall waste statistics with confidence intervals
        stats['overall'] = {
            'total_jobs': len(df),
            'cpu_waste_mean': df['cpu_waste'].mean(),
            'cpu_waste_median': df['cpu_waste'].median(),
            'cpu_waste_std': df['cpu_waste'].std(),
            'cpu_waste_95ci': scipy_stats.t.interval(0.95, len(df)-1,
                                                    loc=df['cpu_waste'].mean(),
                                                    scale=scipy_stats.sem(df['cpu_waste'])),
            'mem_waste_mean': df['mem_waste'].mean(),
            'mem_waste_95ci': scipy_stats.t.interval(0.95, len(df)-1,
                                                    loc=df['mem_waste'].mean(),
                                                    scale=scipy_stats.sem(df['mem_waste'])),
            'composite_waste_mean': df['composite_waste'].mean(),
            'composite_waste_95ci': scipy_stats.t.interval(0.95, len(df)-1,
                                                            loc=df['composite_waste'].mean(),
                                                            scale=scipy_stats.sem(df['composite_waste'])),
        }

        # RQ2 Prevalence Analysis: Waste thresholds
        stats['waste_thresholds'] = {}
        for threshold in [0.5, 0.75, 0.9]:
            threshold_key = f'>{int(threshold * 100)}%'
            stats['waste_thresholds'][threshold_key] = {
                'composite_waste_pct': (df['composite_waste'] > threshold).mean() * 100,
                'cpu_waste_pct': (df['cpu_waste'] > threshold).mean() * 100,
                'mem_waste_pct': (df['mem_waste'] > threshold).mean() * 100,
            }

        # RQ2 Economic Impact: Total resource waste
        stats['total_waste'] = {
            'total_cpu_hours_wasted': df['cpu_hours_wasted'].sum(),
            'avg_cpu_hours_wasted_per_job': df['cpu_hours_wasted'].mean(),
            'total_mem_gb_hours_wasted': df['mem_gb_hours_wasted'].sum(),
            'avg_mem_gb_hours_wasted_per_job': df['mem_gb_hours_wasted'].mean(),
        }

        # RQ2 Job Type Analysis with ANOVA tests
        stats['by_exitcode'] = df.groupby('job_exitcode')['composite_waste'].agg(['mean', 'std', 'count']).round(4)

        # ANOVA test for exit codes
        exitcode_groups = [group['composite_waste'].values for name, group in df.groupby('job_exitcode') if len(group) > 5]
        if len(exitcode_groups) > 1:
            f_stat, p_value = scipy_stats.f_oneway(*exitcode_groups)
            stats['exitcode_anova'] = {'f_statistic': f_stat, 'p_value': p_value}

        stats['by_queue'] = df.groupby('queue')['composite_waste'].agg(['mean', 'std', 'count']).round(4)
        stats['by_duration'] = df.groupby('duration_category')['composite_waste'].agg(['mean', 'std', 'count']).round(4)
        stats['by_size'] = df.groupby('size_category')['composite_waste'].agg(['mean', 'std', 'count']).round(4)

        # Kruskal-Wallis test for job sizes (non-parametric)
        size_groups = [group['composite_waste'].values for name, group in df.groupby('size_category') if len(group) > 5]
        if len(size_groups) > 1:
            h_stat, p_value = scipy_stats.kruskal(*size_groups)
            stats['size_kruskal'] = {'h_statistic': h_stat, 'p_value': p_value}

        # RQ2 User Behavior Analysis: Top wasting users
        user_waste = df.groupby('job_username').agg(
            job_count=('composite_waste', 'count'),
            avg_waste=('composite_waste', 'mean'),
            std_waste=('composite_waste', 'std'),
            total_cpu_hours_wasted=('cpu_hours_wasted', 'sum')
        ).round(4)
        stats['top_wasting_users'] = user_waste[user_waste['job_count'] >= 5].nlargest(20, 'avg_waste')

        # Correlation analysis between waste types
        stats['waste_correlations'] = {
            'cpu_mem_correlation': df[['cpu_waste', 'mem_waste']].corr().iloc[0, 1],
            'cpu_duration_correlation': df[['cpu_waste', 'duration_hours']].corr().iloc[0, 1],
            'cpu_size_correlation': df[['cpu_waste', 'ncores']].corr().iloc[0, 1],
        }

        logger.info("Statistical summaries completed")
        return stats

    def create_visualizations(self, df, stats):
        """
        Create comprehensive visualizations of resource waste patterns.
        Enhanced with additional plots for deeper insights.

        Args:
            df (pd.DataFrame): Job data with waste metrics
            stats (dict): Statistical summaries
        """
        logger.info("Creating visualizations...")
        plt.style.use('seaborn-v0_8')

        # 1. Distribution and category analysis (existing plot, enhanced)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Enhanced histogram with percentile markers
        sns.histplot(df['composite_waste'], bins=50, kde=True, ax=axes[0])
        axes[0].set_title('Distribution of Composite Waste Score', fontsize=14)
        axes[0].set_xlabel('Composite Waste Score', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)

        # Add percentile markers
        percentiles = [25, 50, 75, 90]
        colors = ['green', 'yellow', 'orange', 'red']
        for p, c in zip(percentiles, colors):
            val = np.percentile(df['composite_waste'], p)
            axes[0].axvline(val, color=c, linestyle='--', alpha=0.7,
                        label=f'{p}th percentile: {val:.3f}')
        axes[0].axvline(df['composite_waste'].mean(), color='red', linestyle='-',
                    linewidth=2, label=f'Mean: {df["composite_waste"].mean():.3f}')
        axes[0].legend()

        # Enhanced pie chart
        waste_counts = df['waste_category'].value_counts()
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        axes[1].pie(waste_counts.values, labels=waste_counts.index, autopct='%1.1f%%',
                    startangle=90, colors=colors_pie)
        axes[1].set_title('Job Distribution by Waste Category', fontsize=14)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'waste_distributions_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Waste by job characteristics (new)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # By exit code
        exitcode_order = df.groupby('job_exitcode')['composite_waste'].mean().sort_values(ascending=False).index
        sns.boxplot(data=df, x='job_exitcode', y='composite_waste', order=exitcode_order, ax=axes[0, 0])
        axes[0, 0].set_title('Resource Waste by Job Exit Code', fontsize=14)
        axes[0, 0].set_xlabel('Exit Code', fontsize=12)
        axes[0, 0].set_ylabel('Composite Waste Score', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # By duration category
        duration_order = ['Short (<1h)', 'Medium (1-8h)', 'Long (8-24h)', 'Very Long (>24h)']
        sns.boxplot(data=df, x='duration_category', y='composite_waste', order=duration_order, ax=axes[0, 1])
        axes[0, 1].set_title('Resource Waste by Job Duration', fontsize=14)
        axes[0, 1].set_xlabel('Duration Category', fontsize=12)
        axes[0, 1].set_ylabel('Composite Waste Score', fontsize=12)

        # By size category
        if 'size_category' in df.columns:
            size_order = ['Small (1-16)', 'Medium (17-64)', 'Large (65-256)', 'Very Large (>256)']
            size_order = [s for s in size_order if s in df['size_category'].unique()]
            sns.boxplot(data=df, x='size_category', y='composite_waste', order=size_order, ax=axes[1, 0])
            axes[1, 0].set_title('Resource Waste by Job Size', fontsize=14)
            axes[1, 0].set_xlabel('Size Category (cores)', fontsize=12)
            axes[1, 0].set_ylabel('Composite Waste Score', fontsize=12)

        # CPU vs Memory waste scatter
        scatter = axes[1, 1].scatter(df['cpu_waste'], df['mem_waste'],
                                    c=df['composite_waste'], cmap='RdYlGn_r',
                                    alpha=0.6, s=50)
        axes[1, 1].set_xlabel('CPU Waste', fontsize=12)
        axes[1, 1].set_ylabel('Memory Waste', fontsize=12)
        axes[1, 1].set_title('CPU vs Memory Waste Correlation', fontsize=14)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Add correlation coefficient
        corr = df[['cpu_waste', 'mem_waste']].corr().iloc[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Composite Waste Score', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'waste_by_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Time series analysis (new)
        if 'start_time' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Aggregate by day - handle mixed timezone data
            # Convert to datetime and strip timezone info to avoid mixing issues
            df['date'] = pd.to_datetime(df['start_time'], utc=True).dt.tz_localize(None).dt.date
            daily_waste = df.groupby('date').agg({
                'composite_waste': ['mean', 'std', 'count'],
                'cpu_hours_wasted': 'sum'
            })

            # Plot rolling average
            rolling_mean = daily_waste['composite_waste']['mean'].rolling(window=7, center=True).mean()

            ax.plot(daily_waste.index, daily_waste['composite_waste']['mean'],
                    alpha=0.3, color='blue', label='Daily Average')
            ax.plot(daily_waste.index, rolling_mean, color='red', linewidth=2,
                    label='7-day Moving Average')

            # Add confidence band
            rolling_std = daily_waste['composite_waste']['std'].rolling(window=7, center=True).mean()
            ax.fill_between(daily_waste.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2, color='red')

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Average Composite Waste Score', fontsize=12)
            ax.set_title('Resource Waste Trends Over Time', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'waste_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. User behavior heatmap (new)
        # Select top users by total waste impact
        top_users_impact = df.groupby('job_username')['cpu_hours_wasted'].sum().nlargest(20).index
        user_df = df[df['job_username'].isin(top_users_impact)]

        if len(user_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create pivot table for heatmap
            user_pivot = user_df.pivot_table(
                values='composite_waste',
                index='job_username',
                columns='duration_category',
                aggfunc='mean'
            )

            # Sort by average waste
            user_pivot['avg_waste'] = user_pivot.mean(axis=1)
            user_pivot = user_pivot.sort_values('avg_waste', ascending=False).drop('avg_waste', axis=1)

            # Create heatmap
            sns.heatmap(user_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'Average Composite Waste Score'},
                    ax=ax)
            ax.set_title('User Waste Patterns by Job Duration', fontsize=14)
            ax.set_xlabel('Job Duration Category', fontsize=12)
            ax.set_ylabel('Username', fontsize=12)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'user_waste_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("Visualizations completed")

    def save_results(self, df, stats):
        """
        Save analysis results to CSV files and summary reports.

        Args:
            df (pd.DataFrame): Job data with waste metrics
            stats (dict): Statistical summaries
        """
        logger.info("Saving results...")
        df.to_csv(self.output_dir / 'job_data_with_waste_metrics.csv.gz', index=False, compression='gzip')
        logger.info(f"Saved job data to {self.output_dir / 'job_data_with_waste_metrics.csv.gz'}")

        # Save stats as JSON for easy reading
        import json
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info("All results saved successfully")

    def print_key_findings(self, stats):
        """
        Print key findings suitable for inclusion in an academic paper.

        Args:
            stats (dict): Statistical summaries
        """
        print("\n" + "="*80)
        print("KEY FINDINGS FOR RQ2: Resource Waste Analysis")
        print("="*80)
        print(f"Total jobs analyzed: {stats['overall']['total_jobs']:,}")
        print(f"Average CPU waste: {stats['overall']['cpu_waste_mean']:.1%}")
        print(f"Average memory waste: {stats['overall']['mem_waste_mean']:.1%}")
        print(f"Average composite waste: {stats['overall']['composite_waste_mean']:.1%}")

        for threshold_key, threshold_stats in stats['waste_thresholds'].items():
            print(f"Jobs with composite waste {threshold_key}: {threshold_stats['composite_waste_pct']:.1f}%")

        print(f"Total CPU-hours wasted: {stats['total_waste']['total_cpu_hours_wasted']:,.0f}")
        print("="*80)

    def run_full_analysis(self, limit_years=None, sample_jobs_fraction=None, test_mode=False, hpc_system=None):
        """
        Run the complete resource waste analysis pipeline.

        Args:
            limit_years (list): Years to analyze (for focused analysis)
            sample_jobs_fraction (float): Fraction of JOBS to sample (for testing)
            test_mode (bool): If True, limit to small subset for testing
            hpc_system (str): 'stampede', 'conte', 'anvil', or 'all'/None
        """
        try:
            hpc_label = hpc_system or 'all'
            logger.info(f"Starting Optimized FRESCO Resource Waste Analysis for RQ2 - HPC System: {hpc_label}...")

            limit_files_per_day = 2 if test_mode else None
            if test_mode and sample_jobs_fraction is None:
                sample_jobs_fraction = 0.1

            files = self.discover_data_files(limit_years, limit_files_per_day, hpc_system)
            if not files:
                raise ValueError(f"No data files found for HPC system: {hpc_label}")

            self.job_data = self.load_and_process_parallel(files, sample_jobs_fraction=sample_jobs_fraction)
            self.job_data = self.calculate_resource_waste(self.job_data)
            stats = self.generate_statistical_summaries(self.job_data)

            # Report selected configuration details
            stats['hpc_system'] = hpc_label
            # E) Successfully processed files, not just discovered
            stats['files_processed'] = self.total_files_processed

            self.create_visualizations(self.job_data, stats)
            self.save_results(self.job_data, stats)
            self.print_key_findings(stats)

            logger.info(f"RQ2 Analysis completed successfully for HPC system: {hpc_label}!")
            return stats

        except Exception as e:
            logger.error(f"RQ2 Analysis failed for HPC system {hpc_label}: {e}", exc_info=True)
            raise


def main():
    """
    Main function with examples for running HPC system-specific analyses.
    """
    # Initialize analyzer
    analyzer = FrescoResourceWasteAnalyzer(num_workers=None)

    # Analyze only Stampede data (files containing "_S")
    print("Running analysis for Stampede data only...")
    stampede_stats = analyzer.run_full_analysis(
        test_mode=False,
        sample_jobs_fraction=0.1,
        hpc_system='stampede'
    )

    # Analyze only Conte data (files containing "_C")
    print("\nRunning analysis for Conte data only...")
    conte_stats = analyzer.run_full_analysis(
        test_mode=False,
        sample_jobs_fraction=0.1,
        hpc_system='conte'
    )

    # Analyze only Anvil data (files without "_S" or "_C")
    print("\nRunning analysis for Anvil data only...")
    anvil_stats = analyzer.run_full_analysis(
        test_mode=False,
        sample_jobs_fraction=0.1,
        hpc_system='anvil'
    )

    # Analyze all data (no filtering)
    print("\nRunning analysis for all HPC systems...")
    all_stats = analyzer.run_full_analysis(
        test_mode=False,
        sample_jobs_fraction=0.1,
        hpc_system='all'  # or hpc_system=None
    )

    # Compare results across systems
    print("\n" + "=" * 80)
    print("COMPARISON ACROSS HPC SYSTEMS")
    print("=" * 80)

    systems = [
        ('Conte', conte_stats),
        ('Anvil', anvil_stats),
        ('Stampede', stampede_stats),
        ('All Systems', all_stats)
    ]

    for system_name, stats in systems:
        if stats:
            print(f"\n{system_name}:")
            print(f"  Total jobs: {stats['overall']['total_jobs']:,}")
            print(f"  Files processed: {stats.get('files_processed', 'N/A')}")
            print(f"  Avg CPU waste: {stats['overall']['cpu_waste_mean']:.1%}")
            print(f"  Avg memory waste: {stats['overall']['mem_waste_mean']:.1%}")
            print(f"  Avg composite waste: {stats['overall']['composite_waste_mean']:.1%}")


def run_single_system_analysis(system_name):
    """
    Convenience function to run analysis for a single HPC system.

    Args:
        system_name (str): 'stampede', 'conte', 'anvil', or 'all'
    """
    analyzer = FrescoResourceWasteAnalyzer(num_workers=None)

    print(f"Running analysis for {system_name.upper()} system...")
    stats = analyzer.run_full_analysis(
        test_mode=False,
        sample_jobs_fraction=0.25,
        hpc_system=system_name
    )

    return stats


if __name__ == "__main__":
    # Run the main comparison analysis
    main()

    # Or run individual system analysis:
    # stampede_stats = run_single_system_analysis('stampede')
    # conte_stats = run_single_system_analysis('conte')
    # anvil_stats = run_single_system_analysis('anvil')
    # all_stats = run_single_system_analysis('all')