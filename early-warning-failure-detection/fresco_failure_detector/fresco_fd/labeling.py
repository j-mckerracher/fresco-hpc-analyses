"""
Label construction and exitcode mapping for FRESCO failure detection.

Handles conversion of job exit codes to canonical failure classes and
creates positive/negative examples for training with strict no-leakage guarantees.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ExitCodeClass,
    EXITCODE_MAPPINGS,
    POSITIVE_CLASSES,
    NEGATIVE_CLASSES,
    PREDICTION_HORIZONS,
    SAMPLING_CONFIG
)
from .utils import validate_dataframe_columns


logger = logging.getLogger(__name__)


@dataclass
class LabelExample:
    """Single training example with label and metadata"""
    jid: str
    timestamp: datetime  # Time at which features should be computed
    label: int  # 0=negative, 1=positive
    horizon_minutes: int  # Prediction horizon
    failure_time: Optional[datetime] = None  # Actual failure time (for positives)
    exitcode_class: Optional[ExitCodeClass] = None
    
    @property
    def lead_time_minutes(self) -> Optional[float]:
        """Lead time in minutes (for positive examples)"""
        if self.failure_time and self.label == 1:
            return (self.failure_time - self.timestamp).total_seconds() / 60.0
        return None


class ExitCodeMapper:
    """Maps job exit codes to canonical failure classifications"""
    
    def __init__(self, custom_mappings: Optional[Dict] = None):
        self.mappings = EXITCODE_MAPPINGS.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)
    
    def map_exitcode(self, exitcode: Union[int, str, float]) -> ExitCodeClass:
        """Map exit code to canonical classification"""
        
        # Handle NaN/None
        if pd.isna(exitcode) or exitcode is None:
            return ExitCodeClass.UNKNOWN
        
        # Convert to string for lookup
        exitcode_str = str(exitcode).strip().upper()
        
        # Direct lookup
        if exitcode_str in self.mappings:
            return self.mappings[exitcode_str]
        
        # Try as integer
        try:
            exitcode_int = int(float(exitcode))
            if exitcode_int in self.mappings:
                return self.mappings[exitcode_int]
        except (ValueError, TypeError):
            pass
        
        # Pattern matching for complex exit codes
        exitcode_lower = exitcode_str.lower()
        
        if any(pattern in exitcode_lower for pattern in ['fail', 'error', 'abort']):
            return ExitCodeClass.FAILED
        elif 'timeout' in exitcode_lower or 'dl' in exitcode_lower:
            return ExitCodeClass.TIMEOUT
        elif 'oom' in exitcode_lower or 'memory' in exitcode_lower:
            return ExitCodeClass.OOM
        elif any(pattern in exitcode_lower for pattern in ['cancel', 'preempt']):
            return ExitCodeClass.CANCELLED
        elif any(pattern in exitcode_lower for pattern in ['complete', 'success', 'done']):
            return ExitCodeClass.COMPLETED
        
        logger.debug(f"Unknown exitcode pattern: {exitcode}")
        return ExitCodeClass.UNKNOWN
    
    def map_exitcodes_batch(self, exitcodes: pd.Series) -> pd.Series:
        """Map a series of exit codes efficiently"""
        return exitcodes.apply(self.map_exitcode)
    
    def is_failure(self, exitcode_class: ExitCodeClass) -> bool:
        """Check if exit code class represents a failure"""
        return exitcode_class in POSITIVE_CLASSES
    
    def is_success(self, exitcode_class: ExitCodeClass) -> bool:
        """Check if exit code class represents success"""
        return exitcode_class in NEGATIVE_CLASSES
    
    def get_mapping_stats(self, exitcodes: pd.Series) -> Dict[ExitCodeClass, int]:
        """Get statistics on exit code mappings"""
        mapped = self.map_exitcodes_batch(exitcodes)
        return dict(mapped.value_counts())


class LabelGenerator:
    """
    Generates training labels from job data with strict no-leakage guarantees.
    
    For each failing job:
    - Creates positive examples at t = t_failure - H for each horizon H
    
    For each successful job:
    - Creates negative examples at random times well before completion
    - Excludes the last H minutes to avoid "end signature" leakage
    """
    
    def __init__(
        self,
        horizons: List[int] = PREDICTION_HORIZONS,
        neg_pos_ratio: int = SAMPLING_CONFIG["neg_pos_ratio"],
        min_job_duration_minutes: int = SAMPLING_CONFIG["min_job_duration_minutes"],
        max_examples_per_job: int = SAMPLING_CONFIG["max_examples_per_job"]
    ):
        self.horizons = horizons
        self.neg_pos_ratio = neg_pos_ratio
        self.min_job_duration_minutes = min_job_duration_minutes
        self.max_examples_per_job = max_examples_per_job
        self.exitcode_mapper = ExitCodeMapper()
    
    def generate_labels(
        self,
        job_data: pd.DataFrame,
        deduplicate: bool = True
    ) -> Dict[int, List[LabelExample]]:
        """
        Generate training labels for all horizons.
        
        Args:
            job_data: DataFrame with job information
            deduplicate: Whether to deduplicate examples from same job
            
        Returns:
            Dictionary mapping horizon -> list of label examples
        """
        # Validate input data
        required_cols = ['jid', 'start_time', 'end_time', 'exitcode']
        validate_dataframe_columns(job_data, required_cols, "job_data")
        
        # Prepare job data
        job_data = self._prepare_job_data(job_data)
        
        # Filter to valid jobs
        valid_jobs = self._filter_valid_jobs(job_data)
        logger.info(f"Processing {len(valid_jobs)} valid jobs out of {len(job_data)}")
        
        # Generate examples for each horizon
        all_examples = {}
        
        for horizon in self.horizons:
            logger.info(f"Generating examples for {horizon}-minute horizon")
            
            examples = []
            
            # Generate positive examples (failures)
            positive_examples = self._generate_positive_examples(valid_jobs, horizon)
            examples.extend(positive_examples)
            
            # Generate negative examples (successes)
            negative_examples = self._generate_negative_examples(
                valid_jobs, horizon, len(positive_examples)
            )
            examples.extend(negative_examples)
            
            # Deduplicate if requested
            if deduplicate:
                examples = self._deduplicate_examples(examples)
            
            all_examples[horizon] = examples
            
            logger.info(
                f"Horizon {horizon}min: {len(positive_examples)} positive, "
                f"{len(negative_examples)} negative examples"
            )
        
        return all_examples
    
    def _prepare_job_data(self, job_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean job data"""
        df = job_data.copy()
        
        # Convert timestamps
        for col in ['start_time', 'end_time', 'submit_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        
        # Map exit codes
        df['exitcode_class'] = self.exitcode_mapper.map_exitcodes_batch(df['exitcode'])
        
        # Calculate job duration
        df['duration_minutes'] = (
            df['end_time'] - df['start_time']
        ).dt.total_seconds() / 60.0
        
        # Add failure/success flags
        df['is_failure'] = df['exitcode_class'].apply(self.exitcode_mapper.is_failure)
        df['is_success'] = df['exitcode_class'].apply(self.exitcode_mapper.is_success)
        
        return df
    
    def _filter_valid_jobs(self, job_data: pd.DataFrame) -> pd.DataFrame:
        """Filter to jobs suitable for label generation"""
        df = job_data.copy()
        
        initial_count = len(df)
        
        # Remove jobs with missing critical fields
        df = df.dropna(subset=['jid', 'start_time', 'end_time'])
        logger.debug(f"After removing missing timestamps: {len(df)}")
        
        # Remove jobs with invalid timestamps
        df = df[df['start_time'] < df['end_time']]
        logger.debug(f"After removing invalid timestamps: {len(df)}")
        
        # Remove very short jobs
        df = df[df['duration_minutes'] >= self.min_job_duration_minutes]
        logger.debug(f"After removing short jobs: {len(df)}")
        
        # Remove jobs with unknown exit codes (can't classify)
        df = df[df['exitcode_class'] != ExitCodeClass.UNKNOWN]
        logger.debug(f"After removing unknown exit codes: {len(df)}")
        
        # Only keep jobs we can classify as failure or success
        df = df[(df['is_failure'].astype(bool)) | (df['is_success'].astype(bool))]
        logger.debug(f"After filtering to classifiable jobs: {len(df)}")
        
        logger.info(f"Filtered {initial_count} -> {len(df)} valid jobs")
        return df
    
    def _generate_positive_examples(
        self, 
        job_data: pd.DataFrame, 
        horizon: int
    ) -> List[LabelExample]:
        """Generate positive examples from failing jobs"""
        
        failing_jobs = job_data[job_data['is_failure']].copy()
        logger.debug(f"Found {len(failing_jobs)} failing jobs for horizon {horizon}")
        
        examples = []
        
        for _, job in failing_jobs.iterrows():
            failure_time = job['end_time']
            start_time = job['start_time']
            
            # Example time is exactly H minutes before failure
            example_time = failure_time - timedelta(minutes=horizon)
            
            # Skip if example time is before job start
            if example_time <= start_time:
                logger.debug(f"Skipping job {job['jid']}: example time before start")
                continue
            
            # Create positive example
            example = LabelExample(
                jid=str(job['jid']),
                timestamp=example_time,
                label=1,  # Positive
                horizon_minutes=horizon,
                failure_time=failure_time,
                exitcode_class=job['exitcode_class']
            )
            
            examples.append(example)
        
        logger.info(f"Generated {len(examples)} positive examples for horizon {horizon}")
        return examples
    
    def _generate_negative_examples(
        self,
        job_data: pd.DataFrame,
        horizon: int,
        num_positives: int
    ) -> List[LabelExample]:
        """Generate negative examples from successful jobs"""
        
        successful_jobs = job_data[job_data['is_success']].copy()
        logger.debug(f"Found {len(successful_jobs)} successful jobs")
        
        # Calculate target number of negatives
        target_negatives = num_positives * self.neg_pos_ratio
        
        if len(successful_jobs) == 0:
            logger.warning("No successful jobs found for negative examples")
            return []
        
        examples = []
        examples_per_job = min(
            self.max_examples_per_job,
            max(1, target_negatives // len(successful_jobs))
        )
        
        for _, job in successful_jobs.iterrows():
            start_time = job['start_time']
            end_time = job['end_time']
            
            # Define safe sampling window (exclude last H minutes)
            safe_end_time = end_time - timedelta(minutes=horizon)
            
            # Skip if safe window is too small
            safe_duration = (safe_end_time - start_time).total_seconds() / 60.0
            if safe_duration < self.min_job_duration_minutes:
                continue
            
            # Sample random times within safe window
            job_examples = self._sample_negative_times(
                job,
                start_time,
                safe_end_time,
                examples_per_job,
                horizon
            )
            
            examples.extend(job_examples)
            
            # Stop if we have enough examples
            if len(examples) >= target_negatives:
                break
        
        # Randomly subsample if we have too many
        if len(examples) > target_negatives:
            np.random.shuffle(examples)
            examples = examples[:target_negatives]
        
        logger.info(f"Generated {len(examples)} negative examples for horizon {horizon}")
        return examples
    
    def _sample_negative_times(
        self,
        job: pd.Series,
        start_time: datetime,
        end_time: datetime,
        num_samples: int,
        horizon: int
    ) -> List[LabelExample]:
        """Sample negative example times for a single job"""
        
        examples = []
        duration_seconds = (end_time - start_time).total_seconds()
        
        for _ in range(num_samples):
            # Random time within safe window
            random_seconds = np.random.uniform(0, duration_seconds)
            sample_time = start_time + timedelta(seconds=random_seconds)
            
            example = LabelExample(
                jid=str(job['jid']),
                timestamp=sample_time,
                label=0,  # Negative
                horizon_minutes=horizon,
                failure_time=None,
                exitcode_class=job['exitcode_class']
            )
            
            examples.append(example)
        
        return examples
    
    def _deduplicate_examples(self, examples: List[LabelExample]) -> List[LabelExample]:
        """Remove highly correlated examples from the same job"""
        
        if len(examples) <= 1:
            return examples
        
        # Group by job ID
        job_groups = {}
        for example in examples:
            if example.jid not in job_groups:
                job_groups[example.jid] = []
            job_groups[example.jid].append(example)
        
        deduplicated = []
        min_time_gap_minutes = 5  # Minimum time gap between examples from same job
        
        for jid, job_examples in job_groups.items():
            # Sort by timestamp
            job_examples.sort(key=lambda x: x.timestamp)
            
            # Keep first example
            if job_examples:
                deduplicated.append(job_examples[0])
                last_timestamp = job_examples[0].timestamp
                
                # Add subsequent examples if they're far enough apart
                for example in job_examples[1:]:
                    time_gap = (example.timestamp - last_timestamp).total_seconds() / 60.0
                    
                    if time_gap >= min_time_gap_minutes:
                        deduplicated.append(example)
                        last_timestamp = example.timestamp
        
        removed_count = len(examples) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"Deduplicated: removed {removed_count} examples")
        
        return deduplicated
    
    def examples_to_dataframe(self, examples: List[LabelExample]) -> pd.DataFrame:
        """Convert examples to DataFrame format"""
        
        records = []
        for example in examples:
            record = {
                'jid': example.jid,
                'timestamp': example.timestamp,
                'label': example.label,
                'horizon_minutes': example.horizon_minutes,
                'failure_time': example.failure_time,
                'exitcode_class': example.exitcode_class.value if example.exitcode_class else None,
                'lead_time_minutes': example.lead_time_minutes
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_label_statistics(
        self, 
        examples_by_horizon: Dict[int, List[LabelExample]]
    ) -> pd.DataFrame:
        """Generate summary statistics for generated labels"""
        
        stats = []
        
        for horizon, examples in examples_by_horizon.items():
            if not examples:
                continue
            
            positives = [ex for ex in examples if ex.label == 1]
            negatives = [ex for ex in examples if ex.label == 0]
            
            lead_times = [ex.lead_time_minutes for ex in positives if ex.lead_time_minutes is not None]
            
            stat = {
                'horizon_minutes': horizon,
                'total_examples': len(examples),
                'positive_examples': len(positives),
                'negative_examples': len(negatives),
                'positive_rate': len(positives) / len(examples) if examples else 0,
                'unique_jobs': len(set(ex.jid for ex in examples)),
                'median_lead_time': np.median(lead_times) if lead_times else None,
                'min_lead_time': np.min(lead_times) if lead_times else None,
                'max_lead_time': np.max(lead_times) if lead_times else None
            }
            
            stats.append(stat)
        
        return pd.DataFrame(stats)