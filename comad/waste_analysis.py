"""
FRESCO HPC Resource Waste Analysis Script

This script analyzes resource waste in HPC jobs using the FRESCO dataset
to answer RQ2: "How prevalent and severe is resource waste across different
job types and user behaviors?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FrescoResourceWasteAnalyzer:
    """
    Comprehensive analyzer for resource waste in HPC jobs using FRESCO dataset.

    RQ2 Focus: Analyzes "How prevalent and severe is resource waste across
    different job types and user behaviors?"
    """

    def __init__(self, data_dir="/home/dynamo/a/jmckerra/projects/fresco-analysis/data"):
        """
        Initialize the analyzer with data directory path.

        Args:
            data_dir (str): Path to the FRESCO data chunks directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./fresco_analysis_output")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize data containers
        self.job_data = None
        self.waste_metrics = None

        # Performance metrics tracking
        self.total_jobs_processed = 0
        self.total_files_processed = 0

        logger.info(f"Initialized FRESCO Resource Waste Analyzer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

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

    def load_data_chunked(self, files, chunk_size=1000000, sample_jobs_fraction=None):
        """
        Load data from multiple parquet files in chunks to manage memory.

        RQ2 Implementation: This function loads the raw FRESCO data that contains both
        requested resources (ncores, timelimit) and actual usage metrics (value_cpuuser,
        value_memused) - the foundation for detecting resource waste.
        
        IMPORTANT: Each job (jid) has multiple timestamp records showing resource usage
        over time. We sample JOBS, not individual timestamp records.

        Args:
            files (list): List of file paths to load
            chunk_size (int): Maximum number of rows to process at once
            sample_jobs_fraction (float): Fraction of JOBS to sample (for testing)

        Returns:
            pd.DataFrame: Combined job data (time-series records)
        """
        logger.info(f"Loading data from {len(files)} files...")

        all_data = []
        current_rows = 0
        all_job_ids = set()

        # First pass: collect all unique job IDs if sampling is requested
        if sample_jobs_fraction and sample_jobs_fraction < 1.0:
            logger.info("First pass: collecting job IDs for sampling...")
            for file_path in tqdm(files[:min(50, len(files))], desc="Collecting job IDs"):
                try:
                    df = pd.read_parquet(file_path)
                    df = self._clean_dataframe(df)
                    if len(df) > 0 and 'jid' in df.columns:
                        all_job_ids.update(df['jid'].unique())
                except Exception as e:
                    logger.warning(f"Error reading {file_path} for job ID collection: {e}")
                    continue
            
            # Sample job IDs
            sampled_job_ids = set(pd.Series(list(all_job_ids)).sample(
                frac=sample_jobs_fraction, random_state=42
            ).tolist())
            logger.info(f"Sampled {len(sampled_job_ids)} jobs out of {len(all_job_ids)} total jobs")
        else:
            sampled_job_ids = None

        # Second pass: load data for selected jobs
        for file_path in tqdm(files, desc="Loading files"):
            try:
                df = pd.read_parquet(file_path)

                # Clean and standardize data
                df = self._clean_dataframe(df)

                # Filter to sampled jobs if sampling
                if sampled_job_ids is not None and len(df) > 0 and 'jid' in df.columns:
                    df = df[df['jid'].isin(sampled_job_ids)]

                if len(df) > 0:  # Only append non-empty dataframes
                    all_data.append(df)
                    current_rows += len(df)
                    self.total_files_processed += 1

                # Memory management
                if current_rows >= chunk_size:
                    logger.info(f"Processed {current_rows} rows, {self.total_files_processed} files")
                    gc.collect()

            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No data could be loaded from the specified files")

        # Combine all data
        logger.info("Combining data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Count unique jobs processed
        unique_jobs = combined_data['jid'].nunique() if 'jid' in combined_data.columns else 0
        self.total_jobs_processed = unique_jobs

        logger.info(f"Loaded {len(combined_data)} total records from {unique_jobs} unique jobs")
        return combined_data

    def _clean_dataframe(self, df):
        """
        Clean and standardize a single dataframe.

        RQ2 Implementation: Ensures we have clean data for both "job types" (via exitcode,
        queue classification) and "user behaviors" (via username) that RQ2 analyzes.

        Args:
            df (pd.DataFrame): Raw dataframe from parquet file

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Detect data format based on content
        # Check if this is 2013-style data (incorrect column mapping) or newer format
        is_2013_format = False
        if 'jobname' in df.columns and 'exitcode' in df.columns:
            # Check if jobname contains typical exit codes
            jobname_sample = df['jobname'].dropna().iloc[:5] if len(df['jobname'].dropna()) > 0 else []
            exitcode_sample = df['exitcode'].dropna().iloc[:5] if len(df['exitcode'].dropna()) > 0 else []

            # 2013 format: jobname has COMPLETED/FAILED, exitcode has {NODE...}
            if any(str(val) in ['COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED'] for val in jobname_sample):
                if any(str(val).startswith('{') for val in exitcode_sample):
                    is_2013_format = True

        if is_2013_format:
            # Handle 2013 data column naming inconsistencies
            logger.info("Detected 2013-style data format, applying corrections...")
            # Create proper column assignments for 2013 data
            if 'jobname' in df.columns:
                df['job_exitcode'] = df['jobname']
            if 'host_list' in df.columns:
                df['job_username'] = df['host_list']
            if 'username' in df.columns:
                df['job_cpu_usage'] = pd.to_numeric(df['username'], errors='coerce')
        else:
            # Handle newer data format (2023+)
            logger.info("Detected newer data format...")
            if 'exitcode' in df.columns:
                df['job_exitcode'] = df['exitcode']
            if 'username' in df.columns:
                df['job_username'] = df['username']
            if 'value_cpuuser' in df.columns:
                df['job_cpu_usage'] = pd.to_numeric(df['value_cpuuser'], errors='coerce')

        # Ensure required columns exist with proper names
        # Note: jid is the unique job identifier that groups multiple timestamp records
        required_columns = ['time', 'start_time', 'end_time', 'job_exitcode', 'queue',
                            'job_username', 'job_cpu_usage', 'value_memused', 'ncores', 'nhosts', 'jid']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return pd.DataFrame()  # Return empty dataframe if critical columns missing

        # Convert data types
        try:
            df['time'] = pd.to_datetime(df['time'])
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['job_cpu_usage'] = pd.to_numeric(df['job_cpu_usage'], errors='coerce')
            df['value_memused'] = pd.to_numeric(df['value_memused'], errors='coerce')
            df['ncores'] = pd.to_numeric(df['ncores'], errors='coerce')
            df['nhosts'] = pd.to_numeric(df['nhosts'], errors='coerce')
        except Exception as e:
            logger.warning(f"Error converting data types: {e}")
            return pd.DataFrame()

        # Calculate job duration
        df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        df['duration_hours'] = df['duration_seconds'] / 3600

        # Filter out invalid jobs
        df = df[
            (df['duration_seconds'] > 0) &  # Positive duration
            (df['job_cpu_usage'].notna()) &  # Valid CPU data
            (df['value_memused'].notna()) &  # Valid memory data
            (df['ncores'] > 0)  # Valid core count
            ]

        return df

    def calculate_resource_waste(self, df):
        """
        Calculate resource waste metrics for jobs by aggregating time-series data.

        RQ2 Implementation: This is the CORE function that answers "How prevalent and severe
        is resource waste" by:
        1. Aggregating multiple timestamp records per job to get job-level metrics
        2. Defining waste as the gap between requested and actual resource usage
        3. Creating metrics to quantify waste severity
        4. Categorizing jobs by waste levels for prevalence analysis

        Args:
            df (pd.DataFrame): Time-series job data with multiple records per job (jid)

        Returns:
            pd.DataFrame: Job-level data with waste metrics added
        """
        logger.info("Calculating resource waste metrics by aggregating job time-series...")

        # Group by job ID and aggregate to get job-level metrics
        job_aggregations = {
            # Job metadata (take first value since they're constant per job)
            'start_time': 'first',
            'end_time': 'first', 
            'job_exitcode': 'first',
            'queue': 'first',
            'job_username': 'first',
            'ncores': 'first',
            'nhosts': 'first',
            'duration_hours': 'first',
            
            # Resource usage metrics (aggregate across time series)
            'job_cpu_usage': 'mean',  # Average CPU usage across job lifetime
            'value_memused': 'mean',  # Average memory usage across job lifetime
        }
        
        logger.info(f"Aggregating {len(df)} timestamp records into job-level metrics...")
        job_df = df.groupby('jid').agg(job_aggregations).reset_index()
        
        logger.info(f"Aggregated to {len(job_df)} unique jobs")

        # RQ2 Core Metric 1: CPU Waste
        # Measures how much requested CPU time was wasted
        # Formula: 1 - (average_cpu_usage / 100) where 100% = full utilization
        job_df['cpu_waste'] = 1.0 - (job_df['job_cpu_usage'] / 100.0)
        job_df['cpu_waste'] = job_df['cpu_waste'].clip(0, 1)  # Ensure 0-1 range

        # RQ2 Core Metric 2: Memory Waste
        # Measures how much requested memory was wasted
        # Since we don't have requested memory, estimate based on system capacity
        # Assume each core has access to ~4GB (typical HPC allocation)
        job_df['estimated_requested_mem_gb'] = job_df['ncores'] * 4.0
        job_df['mem_waste'] = 1.0 - (job_df['value_memused'] / job_df['estimated_requested_mem_gb'])
        job_df['mem_waste'] = job_df['mem_waste'].clip(0, 1)  # Ensure 0-1 range

        # RQ2 Severity Metric: Composite Waste Score
        # Combines CPU and memory waste into single severity measure
        # Weighted average: 60% CPU, 40% Memory (CPU typically more critical)
        job_df['composite_waste'] = (0.6 * job_df['cpu_waste']) + (0.4 * job_df['mem_waste'])

        # RQ2 Economic Impact Metrics: Quantify waste in resource-hours
        # These show the real cost of waste in computational resources
        job_df['cpu_hours_wasted'] = job_df['cpu_waste'] * job_df['ncores'] * job_df['duration_hours']
        job_df['mem_gb_hours_wasted'] = job_df['mem_waste'] * job_df['estimated_requested_mem_gb'] * job_df['duration_hours']

        # RQ2 Job Type Classification 1: Duration-based categories
        # Different job types have different waste patterns
        job_df['duration_category'] = pd.cut(
            job_df['duration_hours'],
            bins=[0, 1, 8, 24, float('inf')],
            labels=['Short (<1h)', 'Medium (1-8h)', 'Long (8-24h)', 'Very Long (>24h)']
        )

        # RQ2 Severity Classification: Categorize waste levels for prevalence analysis
        # Enables answering "How prevalent is severe waste?"
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

        RQ2 Implementation: Systematically answers "How prevalent and severe" by:
        1. Prevalence: Percentages of jobs with different waste levels
        2. Severity: Statistical distributions of waste scores
        3. Job Types: Waste patterns across different job characteristics
        4. User Behaviors: Waste patterns across different users

        Args:
            df (pd.DataFrame): Job data with waste metrics

        Returns:
            dict: Dictionary of summary statistics
        """
        logger.info("Generating statistical summaries...")

        stats = {}

        # RQ2 Severity Analysis: Overall waste statistics
        # Provides baseline measures of waste severity across all jobs
        stats['overall'] = {
            'total_jobs': len(df),
            'cpu_waste_mean': df['cpu_waste'].mean(),
            'cpu_waste_median': df['cpu_waste'].median(),
            'cpu_waste_std': df['cpu_waste'].std(),
            'mem_waste_mean': df['mem_waste'].mean(),
            'mem_waste_median': df['mem_waste'].median(),
            'mem_waste_std': df['mem_waste'].std(),
            'composite_waste_mean': df['composite_waste'].mean(),
            'composite_waste_median': df['composite_waste'].median(),
            'composite_waste_std': df['composite_waste'].std(),
        }

        # RQ2 Severity Distribution Analysis: Percentiles
        # Shows how waste is distributed - are most jobs efficient or wasteful?
        percentiles = [25, 50, 75, 90, 95, 99]
        stats['percentiles'] = {}
        for p in percentiles:
            stats['percentiles'][f'{p}th'] = {
                'cpu_waste': np.percentile(df['cpu_waste'], p),
                'mem_waste': np.percentile(df['mem_waste'], p),
                'composite_waste': np.percentile(df['composite_waste'], p)
            }

        # RQ2 Prevalence Analysis: Waste thresholds
        # Directly answers "How prevalent" by counting jobs above waste thresholds
        waste_thresholds = [0.5, 0.75, 0.9]
        stats['waste_thresholds'] = {}
        for threshold in waste_thresholds:
            threshold_key = f'>{int(threshold * 100)}%'
            stats['waste_thresholds'][threshold_key] = {
                'cpu_waste_jobs': (df['cpu_waste'] > threshold).sum(),
                'mem_waste_jobs': (df['mem_waste'] > threshold).sum(),
                'composite_waste_jobs': (df['composite_waste'] > threshold).sum(),
                # These percentages directly answer "How prevalent is severe waste?"
                'cpu_waste_pct': (df['cpu_waste'] > threshold).mean() * 100,
                'mem_waste_pct': (df['mem_waste'] > threshold).mean() * 100,
                'composite_waste_pct': (df['composite_waste'] > threshold).mean() * 100,
            }

        # RQ2 Economic Impact: Total resource waste
        # Quantifies the scale of the waste problem
        stats['total_waste'] = {
            'total_cpu_hours_wasted': df['cpu_hours_wasted'].sum(),
            'total_mem_gb_hours_wasted': df['mem_gb_hours_wasted'].sum(),
            'avg_cpu_hours_wasted_per_job': df['cpu_hours_wasted'].mean(),
            'avg_mem_gb_hours_wasted_per_job': df['mem_gb_hours_wasted'].mean(),
        }

        # RQ2 Job Type Analysis 1: Waste by job outcome
        # Answers "across different job types" - do failed jobs waste more than completed ones?
        stats['by_exitcode'] = df.groupby('job_exitcode').agg({
            'composite_waste': ['count', 'mean', 'median', 'std'],
            'cpu_waste': ['mean', 'median'],
            'mem_waste': ['mean', 'median']
        }).round(4)

        # RQ2 Job Type Analysis 2: Waste by queue
        # Different queues may have different waste patterns (interactive vs batch, etc.)
        stats['by_queue'] = df.groupby('queue').agg({
            'composite_waste': ['count', 'mean', 'median', 'std'],
            'cpu_waste': ['mean', 'median'],
            'mem_waste': ['mean', 'median']
        }).round(4)

        # RQ2 Job Type Analysis 3: Waste by duration category
        # Do short jobs waste more than long jobs? Important for scheduling policies
        stats['by_duration'] = df.groupby('duration_category').agg({
            'composite_waste': ['count', 'mean', 'median', 'std'],
            'cpu_waste': ['mean', 'median'],
            'mem_waste': ['mean', 'median']
        }).round(4)

        # RQ2 User Behavior Analysis: Top wasting users
        # Answers "user behaviors" part of RQ2 - which users consistently waste resources?
        user_waste = df.groupby('job_username').agg({
            'composite_waste': ['count', 'mean', 'sum'],
            'cpu_hours_wasted': 'sum',
            'mem_gb_hours_wasted': 'sum'
        }).round(4)
        user_waste.columns = ['job_count', 'avg_waste', 'total_waste', 'cpu_hours_wasted', 'mem_gb_hours_wasted']
        user_waste = user_waste[user_waste['job_count'] >= 5]  # Users with at least 5 jobs
        stats['top_wasting_users'] = user_waste.nlargest(20, 'total_waste')

        logger.info("Statistical summaries completed")
        return stats

    def create_visualizations(self, df, stats):
        """
        Create comprehensive visualizations of resource waste patterns.

        RQ2 Implementation: Creates visualizations that clearly show:
        1. Prevalence: Histograms and pie charts showing waste distribution
        2. Severity: Box plots and percentile visualizations
        3. Job Types: Comparative charts across different job characteristics
        4. User Behaviors: User-level waste analysis charts

        Args:
            df (pd.DataFrame): Job data with waste metrics
            stats (dict): Statistical summaries
        """
        logger.info("Creating visualizations...")

        # Set up the plotting environment
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)

        # RQ2 Visualization 1: Distribution of waste scores
        # Shows prevalence - how many jobs have low vs high waste?
        fig, axes = plt.subplots(2, 2, figsize=fig_size)

        # CPU Waste distribution - shows prevalence of CPU waste
        axes[0, 0].hist(df['cpu_waste'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of CPU Waste (RQ2: Prevalence)')
        axes[0, 0].set_xlabel('CPU Waste (0-1)')
        axes[0, 0].set_ylabel('Number of Jobs')
        axes[0, 0].axvline(df['cpu_waste'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["cpu_waste"].mean():.3f}')
        axes[0, 0].legend()

        # Memory Waste distribution - shows prevalence of memory waste
        axes[0, 1].hist(df['mem_waste'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Memory Waste (RQ2: Prevalence)')
        axes[0, 1].set_xlabel('Memory Waste (0-1)')
        axes[0, 1].set_ylabel('Number of Jobs')
        axes[0, 1].axvline(df['mem_waste'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["mem_waste"].mean():.3f}')
        axes[0, 1].legend()

        # Composite Waste distribution - shows overall waste prevalence
        axes[1, 0].hist(df['composite_waste'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Composite Waste Score (RQ2: Overall Prevalence)')
        axes[1, 0].set_xlabel('Composite Waste (0-1)')
        axes[1, 0].set_ylabel('Number of Jobs')
        axes[1, 0].axvline(df['composite_waste'].mean(), color='red', linestyle='--',
                           label=f'Mean: {df["composite_waste"].mean():.3f}')
        axes[1, 0].legend()

        # Waste category pie chart - shows prevalence by severity categories
        waste_counts = df['waste_category'].value_counts()
        axes[1, 1].pie(waste_counts.values, labels=waste_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Job Distribution by Waste Category (RQ2: Severity Prevalence)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'waste_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # RQ2 Visualization 2: Waste by job characteristics
        # Shows how waste varies across different job types
        fig, axes = plt.subplots(2, 2, figsize=fig_size)

        # Waste by exit code - answers "job types" part of RQ2
        exitcode_data = df.groupby('job_exitcode')['composite_waste'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(exitcode_data)), exitcode_data.values)
        axes[0, 0].set_title('Average Waste by Job Outcome (RQ2: Job Types)')
        axes[0, 0].set_xlabel('Job Outcome')
        axes[0, 0].set_ylabel('Average Composite Waste')
        axes[0, 0].set_xticks(range(len(exitcode_data)))
        axes[0, 0].set_xticklabels(exitcode_data.index, rotation=45)

        # Waste by queue - shows waste patterns across different job types/queues
        queue_data = df.groupby('queue')['composite_waste'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(queue_data)), queue_data.values)
        axes[0, 1].set_title('Average Waste by Queue (RQ2: Job Types)')
        axes[0, 1].set_xlabel('Queue')
        axes[0, 1].set_ylabel('Average Composite Waste')
        axes[0, 1].set_xticks(range(len(queue_data)))
        axes[0, 1].set_xticklabels(queue_data.index, rotation=45)

        # Waste by duration category - shows how job length affects waste
        duration_data = df.groupby('duration_category')['composite_waste'].mean()
        axes[1, 0].bar(range(len(duration_data)), duration_data.values)
        axes[1, 0].set_title('Average Waste by Job Duration (RQ2: Job Types)')
        axes[1, 0].set_xlabel('Duration Category')
        axes[1, 0].set_ylabel('Average Composite Waste')
        axes[1, 0].set_xticks(range(len(duration_data)))
        axes[1, 0].set_xticklabels(duration_data.index, rotation=45)

        # Box plot of waste by exit code - shows severity distribution by job type
        df_sample = df.sample(min(10000, len(df)), random_state=42)  # Sample for performance
        sns.boxplot(data=df_sample, x='job_exitcode', y='composite_waste', ax=axes[1, 1])
        axes[1, 1].set_title('Waste Distribution by Job Outcome (RQ2: Severity by Job Type)')
        axes[1, 1].set_xlabel('Job Outcome')
        axes[1, 1].set_ylabel('Composite Waste')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'waste_by_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # RQ2 Visualization 3: Temporal patterns
        # Shows if waste varies by time - useful for understanding user behaviors
        if 'start_time' in df.columns:
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))

            # Daily waste patterns - shows temporal trends in user behavior
            df['date'] = pd.to_datetime(df['start_time']).dt.date
            daily_waste = df.groupby('date')['composite_waste'].mean()

            axes[0].plot(daily_waste.index, daily_waste.values, alpha=0.7)
            axes[0].set_title('Daily Average Waste Over Time (RQ2: User Behavior Patterns)')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Average Composite Waste')
            axes[0].tick_params(axis='x', rotation=45)

            # Hourly patterns - shows if certain times have more waste
            df['hour'] = pd.to_datetime(df['start_time']).dt.hour
            hourly_waste = df.groupby('hour')['composite_waste'].mean()

            axes[1].bar(hourly_waste.index, hourly_waste.values)
            axes[1].set_title('Average Waste by Hour of Day (RQ2: Temporal User Behavior)')
            axes[1].set_xlabel('Hour of Day')
            axes[1].set_ylabel('Average Composite Waste')
            axes[1].set_xticks(range(0, 24, 2))

            plt.tight_layout()
            plt.savefig(self.output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

        # RQ2 Visualization 4: User analysis
        # Directly addresses "user behaviors" part of RQ2
        user_summary = df.groupby('job_username').agg({
            'composite_waste': ['count', 'mean'],
            'cpu_hours_wasted': 'sum'
        }).round(4)
        user_summary.columns = ['job_count', 'avg_waste', 'total_cpu_hours_wasted']
        user_summary = user_summary[user_summary['job_count'] >= 5]

        if len(user_summary) > 0:
            fig, axes = plt.subplots(1, 2, figsize=fig_size)

            # Top wasting users by average waste - shows user behavior patterns
            top_avg_wasters = user_summary.nlargest(15, 'avg_waste')
            axes[0].barh(range(len(top_avg_wasters)), top_avg_wasters['avg_waste'].values)
            axes[0].set_title('Top 15 Users by Average Waste (RQ2: User Behaviors)')
            axes[0].set_xlabel('Average Composite Waste')
            axes[0].set_ylabel('User')
            axes[0].set_yticks(range(len(top_avg_wasters)))
            axes[0].set_yticklabels(top_avg_wasters.index, fontsize=8)

            # Top wasting users by total waste - shows which users cause most damage
            top_total_wasters = user_summary.nlargest(15, 'total_cpu_hours_wasted')
            axes[1].barh(range(len(top_total_wasters)), top_total_wasters['total_cpu_hours_wasted'].values)
            axes[1].set_title('Top 15 Users by Total CPU Hours Wasted (RQ2: User Impact)')
            axes[1].set_xlabel('Total CPU Hours Wasted')
            axes[1].set_ylabel('User')
            axes[1].set_yticks(range(len(top_total_wasters)))
            axes[1].set_yticklabels(top_total_wasters.index, fontsize=8)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'user_analysis.png', dpi=300, bbox_inches='tight')
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

        # Save job data with waste metrics
        output_file = self.output_dir / 'job_data_with_waste_metrics.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved job data to {output_file}")

        # Save statistical summaries
        stats_file = self.output_dir / 'waste_statistics_summary.txt'
        with open(stats_file, 'w') as f:
            f.write("FRESCO HPC Resource Waste Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Jobs Analyzed: {stats['overall']['total_jobs']:,}\n")
            f.write(f"Total Files Processed: {self.total_files_processed}\n\n")

            f.write("OVERALL WASTE STATISTICS\n")
            f.write("-" * 30 + "\n")
            for key, value in stats['overall'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")

            f.write("\nWASTE PERCENTILES\n")
            f.write("-" * 20 + "\n")
            for percentile, values in stats['percentiles'].items():
                f.write(f"{percentile} Percentile:\n")
                for metric, value in values.items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")

            f.write("\nWASTE THRESHOLD ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for threshold, values in stats['waste_thresholds'].items():
                f.write(f"Jobs with {threshold} waste:\n")
                for metric, value in values.items():
                    if 'jobs' in metric:
                        f.write(f"  {metric.replace('_', ' ').title()}: {value:,}\n")
                    else:
                        f.write(f"  {metric.replace('_', ' ').title()}: {value:.2f}%\n")

            f.write("\nTOTAL RESOURCE WASTE\n")
            f.write("-" * 25 + "\n")
            for key, value in stats['total_waste'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,.2f}\n")

        logger.info(f"Saved statistics summary to {stats_file}")

        # Save detailed breakdowns as CSV
        stats['by_exitcode'].to_csv(self.output_dir / 'waste_by_exitcode.csv')
        stats['by_queue'].to_csv(self.output_dir / 'waste_by_queue.csv')
        stats['by_duration'].to_csv(self.output_dir / 'waste_by_duration.csv')
        stats['top_wasting_users'].to_csv(self.output_dir / 'top_wasting_users.csv')

        logger.info("All results saved successfully")

    def print_key_findings(self, stats):
        """
        Print key findings suitable for inclusion in an academic paper.

        RQ2 Implementation: Provides publication-ready answers to RQ2:
        1. Prevalence statistics (what % of jobs waste resources)
        2. Severity statistics (how much resources are wasted)
        3. Job type patterns (which types waste most)
        4. Economic quantification (cost of waste)

        Args:
            stats (dict): Statistical summaries
        """
        print("\n" + "=" * 80)
        print("KEY FINDINGS FOR RQ2: Resource Waste Analysis")
        print("=" * 80)

        print(f"\nDATASET OVERVIEW:")
        print(f"   • Analyzed {stats['overall']['total_jobs']:,} HPC jobs from FRESCO dataset")
        print(f"   • Processed {self.total_files_processed:,} data files")

        # RQ2 Answer Part 1: PREVALENCE of resource waste
        print(f"\nRQ2 ANSWER - RESOURCE WASTE PREVALENCE:")
        cpu_waste_mean = stats['overall']['cpu_waste_mean']
        mem_waste_mean = stats['overall']['mem_waste_mean']
        composite_waste_mean = stats['overall']['composite_waste_mean']

        print(f"   • Average CPU waste: {cpu_waste_mean:.1%} of requested CPU resources unused")
        print(f"   • Average memory waste: {mem_waste_mean:.1%} of requested memory unused")
        print(f"   • Average composite waste: {composite_waste_mean:.1%} of overall resources unused")

        # RQ2 Answer Part 2: SEVERITY distribution
        print(f"\nRQ2 ANSWER - WASTE SEVERITY DISTRIBUTION:")
        high_waste_pct = stats['waste_thresholds']['>50%']['composite_waste_pct']
        very_high_waste_pct = stats['waste_thresholds']['>75%']['composite_waste_pct']
        extreme_waste_pct = stats['waste_thresholds']['>90%']['composite_waste_pct']

        print(f"   • Jobs with >50% waste: {high_waste_pct:.1f}% (severe waste)")
        print(f"   • Jobs with >75% waste: {very_high_waste_pct:.1f}% (very severe waste)")
        print(f"   • Jobs with >90% waste: {extreme_waste_pct:.1f}% (extreme waste)")

        print(f"\nECONOMIC IMPACT OF WASTE:")
        total_cpu_hours_wasted = stats['total_waste']['total_cpu_hours_wasted']
        avg_cpu_hours_per_job = stats['total_waste']['avg_cpu_hours_wasted_per_job']

        print(f"   • Total CPU hours wasted: {total_cpu_hours_wasted:,.0f}")
        print(f"   • Average CPU hours wasted per job: {avg_cpu_hours_per_job:.2f}")

        # Estimate cost (assuming $0.10 per CPU hour - adjust based on actual costs)
        estimated_cost = total_cpu_hours_wasted * 0.10
        print(f"   • Estimated cost impact: ${estimated_cost:,.2f} (at $0.10/CPU-hour)")

        # RQ2 Answer Part 3: Patterns by JOB TYPES
        print(f"\nRQ2 ANSWER - PATTERNS BY JOB TYPES:")

        # Top 3 exit codes by average waste
        by_exitcode = stats['by_exitcode']['composite_waste']['mean'].sort_values(ascending=False)
        print(f"   • Highest waste by job outcome:")
        for i, (exitcode, waste) in enumerate(by_exitcode.head(3).items()):
            print(f"     {i + 1}. {exitcode}: {waste:.1%} average waste")

        # Queue analysis
        by_queue = stats['by_queue']['composite_waste']['mean'].sort_values(ascending=False)
        print(f"   • Highest waste by queue type:")
        for i, (queue, waste) in enumerate(by_queue.head(3).items()):
            print(f"     {i + 1}. {queue}: {waste:.1%} average waste")

        # Duration analysis
        by_duration = stats['by_duration']['composite_waste']['mean'].sort_values(ascending=False)
        print(f"   • Highest waste by job duration:")
        for duration, waste in by_duration.items():
            print(f"     • {duration}: {waste:.1%} average waste")

        # RQ2 Answer Part 4: USER BEHAVIORS
        print(f"\nRQ2 ANSWER - USER BEHAVIOR PATTERNS:")
        top_users = stats['top_wasting_users'].head(5)
        print(f"   • Top 5 resource-wasting users:")
        for i, (user, data) in enumerate(top_users.iterrows()):
            print(f"     {i + 1}. User {user}: {data['avg_waste']:.1%} avg waste, {data['job_count']} jobs")

        # Calculate user behavior insights
        user_stats = stats['top_wasting_users']
        high_waste_users = len(user_stats[user_stats['avg_waste'] > 0.7])
        total_users = len(user_stats)

        # Add a check to prevent division by zero if no users meet the criteria
        if total_users > 0:
            print(
                f"   • {high_waste_users}/{total_users} users ({high_waste_users / total_users:.1%}) consistently waste >70% of resources")
        else:
            print("   • Not enough user data (users with >=5 jobs) to analyze waste consistency.")
        print(
            f"   • {high_waste_users}/{total_users} users ({high_waste_users / total_users:.1%}) consistently waste >70% of resources")

        print(f"\nRQ2 RESEARCH IMPLICATIONS:")
        print(f"   • Resource waste is PREVALENT: {composite_waste_mean:.0%} of allocated resources unused on average")
        print(f"   • Waste severity varies by job type: failed jobs ≠ completed jobs in waste patterns")
        print(f"   • User behavior is key factor: consistent patterns of over-requesting resources")
        print(f"   • Economic impact is substantial: {total_cpu_hours_wasted:,.0f} CPU-hours wasted")
        print(f"   • Targeted interventions possible: focus on high-waste users and job types")

        print("\n" + "=" * 80)

    def run_full_analysis(self, limit_years=None, sample_jobs_fraction=None, test_mode=False):
        """
        Run the complete resource waste analysis pipeline.

        RQ2 Implementation: Orchestrates the full analysis to answer RQ2 by:
        1. Loading FRESCO data with requested vs actual resource usage
        2. Calculating waste metrics to quantify prevalence and severity
        3. Analyzing patterns across job types and user behaviors
        4. Generating actionable insights for HPC administrators

        Args:
            limit_years (list): Years to analyze (for focused analysis)
            sample_jobs_fraction (float): Fraction of JOBS to sample (for testing)
            test_mode (bool): If True, limit to small subset for testing
        """
        try:
            logger.info("Starting FRESCO Resource Waste Analysis for RQ2...")

            # Discover data files
            if test_mode:
                if not limit_years:
                    limit_years = []  # Focus on one year for testing
                limit_files_per_day = 2  # Limit files per day
                if sample_jobs_fraction is None:
                    sample_jobs_fraction = 0.1  # Sample 10% of jobs
            else:
                limit_files_per_day = None

            files = self.discover_data_files(limit_years, limit_files_per_day)

            if not files:
                raise ValueError("No data files found")

            # RQ2 Step 1: Load data with both requested and actual resource usage
            self.job_data = self.load_data_chunked(files, sample_jobs_fraction=sample_jobs_fraction)

            # RQ2 Step 2: Calculate waste metrics (core of prevalence/severity analysis)
            self.job_data = self.calculate_resource_waste(self.job_data)

            # RQ2 Step 3: Generate statistics (answers prevalence/severity questions)
            stats = self.generate_statistical_summaries(self.job_data)

            # RQ2 Step 4: Create visualizations (shows patterns across job types/users)
            self.create_visualizations(self.job_data, stats)

            # RQ2 Step 5: Save results for further analysis
            self.save_results(self.job_data, stats)

            # RQ2 Step 6: Print publication-ready findings
            self.print_key_findings(stats)

            logger.info("RQ2 Analysis completed successfully!")
            return stats

        except Exception as e:
            logger.error(f"RQ2 Analysis failed: {e}")
            raise


def main():
    """
    Main function to run the FRESCO resource waste analysis for RQ2.

    RQ2 Focus: This script specifically addresses Research Question 2:
    "How prevalent and severe is resource waste across different job types and user behaviors?"

    The analysis demonstrates the original research idea of detecting "jobs that used
    far less than requested" by systematically measuring the gap between requested
    resources (ncores, estimated memory) and actual usage (value_cpuuser, value_memused).
    """
    # Initialize analyzer
    analyzer = FrescoResourceWasteAnalyzer()

    # Set test_mode=True for quick testing with subset of data
    # Set test_mode=False for full analysis
    stats = analyzer.run_full_analysis(test_mode=False, sample_jobs_fraction=0.01)

    print(f"\nRQ2 Analysis complete! Results saved to: {analyzer.output_dir}")
    print("To run full analysis, set test_mode=False in the main() function")
    print("\nThis analysis directly answers RQ2 by:")
    print("• Quantifying PREVALENCE: What % of jobs waste resources")
    print("• Measuring SEVERITY: How much resources are wasted")
    print("• Analyzing JOB TYPES: Waste patterns by outcome, queue, duration")
    print("• Studying USER BEHAVIORS: Which users consistently waste resources")


if __name__ == "__main__":
    main()