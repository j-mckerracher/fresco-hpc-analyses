"""
FRESCO HPC Resource Waste Analysis Script (Optimized Multiprocessing Version)

This script analyzes resource waste in HPC jobs using the FRESCO dataset
to answer RQ2: "How prevalent and severe is resource waste across different
job types and user behaviors?"

This version has been optimized for maximum performance using multiprocessing
with early aggregation to dramatically reduce memory usage and processing time.
"""

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


def _process_and_aggregate_file(file_path, sampled_job_ids=None):
    """
    Process a single file and return aggregated job-level data.
    This function runs in a separate process for true parallelism.

    Args:
        file_path (Path): Path to the parquet file
        sampled_job_ids (set, optional): Set of job IDs to filter by

    Returns:
        pd.DataFrame or None: Aggregated job data or None if error/empty
    """
    try:
        # Read the file
        df = pd.read_parquet(file_path)

        if df.empty:
            return None

        # Clean the dataframe
        df = _clean_dataframe(df)

        if df.empty:
            return None

        # Apply sampling filter if provided
        if sampled_job_ids and 'jid' in df.columns:
            df = df[df['jid'].isin(sampled_job_ids)]
            if df.empty:
                return None

        # Perform aggregation immediately in this process
        job_aggregations = {
            'start_time': 'first',
            'end_time': 'first',
            'job_exitcode': 'first',
            'queue': 'first',
            'job_username': 'first',
            'ncores': 'first',
            'nhosts': 'first',
            'duration_hours': 'first',
            'job_cpu_usage': ['mean', 'count'],  # Need count for weighted average later
            'value_memused': ['mean', 'count']   # Need count for weighted average later
        }

        aggregated_df = df.groupby('jid').agg(job_aggregations)

        # Flatten multi-index columns
        aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                                for col in aggregated_df.columns.values]
        aggregated_df = aggregated_df.reset_index()

        return aggregated_df

    except Exception as e:
        logger.warning(f"Could not process {file_path}: {e}")
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

    # Detect data format based on content
    is_2013_format = False
    if 'jobname' in df.columns and 'exitcode' in df.columns:
        # Quick sample check for 2013 format
        jobname_sample = df['jobname'].dropna().head(5)
        exitcode_sample = df['exitcode'].dropna().head(5)

        if len(jobname_sample) > 0 and len(exitcode_sample) > 0:
            # 2013 format: jobname has COMPLETED/FAILED, exitcode has {NODE...}
            if any(str(val) in ['COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED'] for val in jobname_sample):
                if any(str(val).startswith('{') for val in exitcode_sample):
                    is_2013_format = True

    # Handle column mapping based on format
    if is_2013_format:
        if 'jobname' in df.columns:
            df['job_exitcode'] = df['jobname']
        if 'host_list' in df.columns:
            df['job_username'] = df['host_list']
        if 'username' in df.columns:
            df['job_cpu_usage'] = pd.to_numeric(df['username'], errors='coerce')
    else:
        # Newer data format (2023+)
        if 'exitcode' in df.columns:
            df['job_exitcode'] = df['exitcode']
        if 'username' in df.columns:
            df['job_username'] = df['username']
        if 'value_cpuuser' in df.columns:
            df['job_cpu_usage'] = pd.to_numeric(df['value_cpuuser'], errors='coerce')

    # Check for required columns
    required_columns = ['time', 'start_time', 'end_time', 'job_exitcode', 'queue',
                       'job_username', 'job_cpu_usage', 'value_memused', 'ncores', 'nhosts', 'jid']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return pd.DataFrame()  # Early return for missing columns

    # Convert data types efficiently
    try:
        # Convert datetime columns
        for col in ['time', 'start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col])

        # Convert numeric columns
        numeric_columns = ['job_cpu_usage', 'value_memused', 'ncores', 'nhosts']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    except Exception as e:
        logger.warning(f"Error converting data types: {e}")
        return pd.DataFrame()

    # Calculate job duration
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['duration_hours'] = df['duration_seconds'] / 3600

    # Filter out invalid jobs with vectorized operations
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

    def __init__(self, data_dir="/home/dynamo/a/jmckerra/projects/fresco-analysis/data", num_workers=None):
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

    def discover_data_files(self, limit_years=None, limit_files_per_day=None):
        """
        Discover all parquet files in the data directory.

        Args:
            limit_years (list): List of years to limit analysis to (e.g., [2013, 2014])
            limit_files_per_day (int): Limit number of files per day for testing

        Returns:
            list: List of file paths
        """
        logger.info("Discovering data files...")
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
                    if limit_files_per_day:
                        day_files = day_files[:limit_files_per_day]
                    files.extend(day_files)

        logger.info(f"Discovered {len(files)} data files")
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
        This replaces the old load_data_chunked method with a much faster approach.

        Args:
            files (list): List of file paths to process
            sample_jobs_fraction (float): Fraction of jobs to sample (for testing)

        Returns:
            pd.DataFrame: Aggregated job-level data
        """
        logger.info(f"Processing {len(files)} files in parallel using {self.num_workers} workers...")

        # Handle sampling if requested
        sampled_job_ids = None
        if sample_jobs_fraction and sample_jobs_fraction < 1.0:
            sampled_job_ids = self._efficient_job_sampling(files, sample_jobs_fraction)

        # Process files in parallel with early aggregation
        pre_aggregated_results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Create partial function with sampled_job_ids
            process_func = partial(_process_and_aggregate_file, sampled_job_ids=sampled_job_ids)

            # Submit all tasks
            future_to_file = {executor.submit(process_func, f): f for f in files}

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        pre_aggregated_results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Process failed for {file_path}: {e}")

        self.total_files_processed = len(pre_aggregated_results)

        if not pre_aggregated_results:
            raise ValueError("No data could be processed from the specified files")

        logger.info(f"Successfully processed {len(pre_aggregated_results)} files")
        logger.info("Combining pre-aggregated results...")

        # Combine all pre-aggregated results
        combined_df = pd.concat(pre_aggregated_results, ignore_index=True)
        gc.collect()  # Clean up memory

        # Final aggregation to handle jobs that span multiple files
        logger.info("Performing final aggregation for jobs spanning multiple files...")

        def weighted_mean(group, value_col, weight_col):
            """Calculate weighted mean for a group"""
            weights = group[weight_col]
            if weights.sum() == 0:
                return 0
            return (group[value_col] * weights).sum() / weights.sum()

        # Group by job ID and aggregate
        final_agg = combined_df.groupby('jid').apply(
            lambda g: pd.Series({
                'start_time': g['start_time_first'].iloc[0],
                'end_time': g['end_time_first'].iloc[0],
                'job_exitcode': g['job_exitcode_first'].iloc[0],
                'queue': g['queue_first'].iloc[0],
                'job_username': g['job_username_first'].iloc[0],
                'ncores': g['ncores_first'].iloc[0],
                'nhosts': g['nhosts_first'].iloc[0],
                'duration_hours': g['duration_hours_first'].iloc[0],
                'job_cpu_usage': weighted_mean(g, 'job_cpu_usage_mean', 'job_cpu_usage_count'),
                'value_memused': weighted_mean(g, 'value_memused_mean', 'value_memused_count'),
            })
        ).reset_index()

        self.total_jobs_processed = len(final_agg)
        logger.info(f"Final aggregated data contains {self.total_jobs_processed:,} unique jobs")

        return final_agg

    def calculate_resource_waste(self, job_df):
        """
        Calculate resource waste metrics on already-aggregated job data.
        Much simpler now since aggregation is already done.

        Args:
            job_df (pd.DataFrame): Aggregated job-level data

        Returns:
            pd.DataFrame: Job data with waste metrics added
        """
        logger.info("Calculating resource waste metrics...")

        # The input is already aggregated, so we just calculate waste metrics
        job_df = job_df.copy()

        # RQ2 Core Metric 1: CPU Waste
        job_df['cpu_waste'] = 1.0 - (job_df['job_cpu_usage'] / 100.0)
        job_df['cpu_waste'] = job_df['cpu_waste'].clip(0, 1)

        # RQ2 Core Metric 2: Memory Waste
        job_df['estimated_requested_mem_gb'] = job_df['ncores'] * 4.0
        job_df['mem_waste'] = 1.0 - (job_df['value_memused'] / job_df['estimated_requested_mem_gb'])
        job_df['mem_waste'] = job_df['mem_waste'].clip(0, 1)

        # RQ2 Severity Metric: Composite Waste Score
        job_df['composite_waste'] = (0.6 * job_df['cpu_waste']) + (0.4 * job_df['mem_waste'])

        # RQ2 Economic Impact Metrics
        job_df['cpu_hours_wasted'] = job_df['cpu_waste'] * job_df['ncores'] * job_df['duration_hours']
        job_df['mem_gb_hours_wasted'] = job_df['mem_waste'] * job_df['estimated_requested_mem_gb'] * job_df['duration_hours']

        # RQ2 Job Type Classification
        job_df['duration_category'] = pd.cut(
            job_df['duration_hours'],
            bins=[0, 1, 8, 24, float('inf')],
            labels=['Short (<1h)', 'Medium (1-8h)', 'Long (8-24h)', 'Very Long (>24h)']
        )

        # RQ2 Severity Classification
        job_df['waste_category'] = pd.cut(
            job_df['composite_waste'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
        )

        logger.info(f"Resource waste calculation completed for {len(job_df)} jobs")
        return job_df

    def generate_statistical_summaries(self, df):
        """
        Generate comprehensive statistical summaries of resource waste.

        Args:
            df (pd.DataFrame): Job data with waste metrics

        Returns:
            dict: Dictionary of summary statistics
        """
        logger.info("Generating statistical summaries...")

        stats = {}

        # RQ2 Severity Analysis: Overall waste statistics
        stats['overall'] = {
            'total_jobs': len(df),
            'cpu_waste_mean': df['cpu_waste'].mean(),
            'cpu_waste_median': df['cpu_waste'].median(),
            'mem_waste_mean': df['mem_waste'].mean(),
            'composite_waste_mean': df['composite_waste'].mean(),
        }

        # RQ2 Prevalence Analysis: Waste thresholds
        stats['waste_thresholds'] = {}
        for threshold in [0.5, 0.75, 0.9]:
            threshold_key = f'>{int(threshold * 100)}%'
            stats['waste_thresholds'][threshold_key] = {
                'composite_waste_pct': (df['composite_waste'] > threshold).mean() * 100,
            }

        # RQ2 Economic Impact: Total resource waste
        stats['total_waste'] = {
            'total_cpu_hours_wasted': df['cpu_hours_wasted'].sum(),
            'avg_cpu_hours_wasted_per_job': df['cpu_hours_wasted'].mean(),
        }

        # RQ2 Job Type Analysis
        stats['by_exitcode'] = df.groupby('job_exitcode')['composite_waste'].agg(['mean']).round(4)
        stats['by_queue'] = df.groupby('queue')['composite_waste'].agg(['mean']).round(4)
        stats['by_duration'] = df.groupby('duration_category')['composite_waste'].agg(['mean']).round(4)

        # RQ2 User Behavior Analysis: Top wasting users
        user_waste = df.groupby('job_username').agg(
            job_count=('composite_waste', 'count'),
            avg_waste=('composite_waste', 'mean')
        ).round(4)
        stats['top_wasting_users'] = user_waste[user_waste['job_count'] >= 5].nlargest(20, 'avg_waste')

        logger.info("Statistical summaries completed")
        return stats

    def create_visualizations(self, df, stats):
        """
        Create comprehensive visualizations of resource waste patterns.

        Args:
            df (pd.DataFrame): Job data with waste metrics
            stats (dict): Statistical summaries
        """
        logger.info("Creating visualizations...")
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)

        # Distribution of waste scores
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        sns.histplot(df['composite_waste'], bins=50, kde=True, ax=axes[0])
        axes[0].set_title('Distribution of Composite Waste Score')
        axes[0].axvline(df['composite_waste'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["composite_waste"].mean():.3f}')
        axes[0].legend()

        waste_counts = df['waste_category'].value_counts()
        axes[1].pie(waste_counts.values, labels=waste_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Job Distribution by Waste Category')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'waste_distributions.png', dpi=300)
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

    def run_full_analysis(self, limit_years=None, sample_jobs_fraction=None, test_mode=False):
        """
        Run the complete resource waste analysis pipeline.

        Args:
            limit_years (list): Years to analyze (for focused analysis)
            sample_jobs_fraction (float): Fraction of JOBS to sample (for testing)
            test_mode (bool): If True, limit to small subset for testing
        """
        try:
            logger.info("Starting Optimized FRESCO Resource Waste Analysis for RQ2...")
            limit_files_per_day = 2 if test_mode else None
            if test_mode and sample_jobs_fraction is None:
                sample_jobs_fraction = 0.1

            files = self.discover_data_files(limit_years, limit_files_per_day)
            if not files:
                raise ValueError("No data files found")

            self.job_data = self.load_and_process_parallel(files, sample_jobs_fraction=sample_jobs_fraction)
            self.job_data = self.calculate_resource_waste(self.job_data)
            stats = self.generate_statistical_summaries(self.job_data)
            self.create_visualizations(self.job_data, stats)
            self.save_results(self.job_data, stats)
            self.print_key_findings(stats)

            logger.info("RQ2 Analysis completed successfully!")
            return stats

        except Exception as e:
            logger.error(f"RQ2 Analysis failed: {e}", exc_info=True)
            raise


def main():
    """
    Main function to run the optimized FRESCO resource waste analysis for RQ2.
    """
    # Initialize analyzer with configurable number of worker processes
    # Default uses all CPU cores for maximum parallelism
    analyzer = FrescoResourceWasteAnalyzer(num_workers=None)  # Use all CPU cores

    # Set test_mode=True for quick testing with subset of data
    # Set test_mode=False for full analysis
    stats = analyzer.run_full_analysis(test_mode=False, sample_jobs_fraction=0.1)

    if stats:
        print(f"\nRQ2 Analysis complete! Results saved to: {analyzer.output_dir}")
        print("For full analysis, set sample_jobs_fraction=None or 1.0")


if __name__ == "__main__":
    main()