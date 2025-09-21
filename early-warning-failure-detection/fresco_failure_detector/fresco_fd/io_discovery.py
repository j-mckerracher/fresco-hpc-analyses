"""
File discovery and cluster tagging for FRESCO dataset.

Handles lazy discovery of hour-chunked Parquet files and automatic cluster detection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Union
import re
from datetime import datetime, date

from .config import ClusterType, get_cluster_from_path


logger = logging.getLogger(__name__)


@dataclass
class DataFile:
    """Metadata for a discovered data file"""
    path: Path
    cluster: Optional[ClusterType]
    year: int
    month: int
    day: int
    hour: Optional[int] = None
    size_bytes: int = 0
    
    @property
    def date_key(self) -> str:
        """Date key for sorting/grouping"""
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
    
    @property
    def datetime_key(self) -> str:
        """Datetime key including hour if available"""
        if self.hour is not None:
            return f"{self.date_key}-{self.hour:02d}"
        return self.date_key


class DataDiscovery:
    """
    Discovers and indexes FRESCO dataset files with lazy loading.
    
    Expected structure: {data_root}/{cluster?}/year/month/day/*.parquet
    """
    
    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self._file_cache: Optional[List[DataFile]] = None
        self._index_cache: Optional[Dict] = None
        
        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
    
    def discover_files(
        self,
        clusters: Optional[Set[ClusterType]] = None,
        years: Optional[Set[int]] = None,
        months: Optional[Set[int]] = None,
        file_pattern: str = "*.parquet",
        force_refresh: bool = False
    ) -> List[DataFile]:
        """
        Discover data files matching criteria.
        
        Args:
            clusters: Filter by cluster types (None = all)
            years: Filter by years (None = all)
            months: Filter by months (None = all)
            file_pattern: File pattern to match
            force_refresh: Force refresh of cached file list
            
        Returns:
            List of discovered data files
        """
        # Use cache if available and not forcing refresh
        if not force_refresh and self._file_cache is not None:
            return self._filter_files(self._file_cache, clusters, years, months)
        
        logger.info(f"Discovering files in {self.data_root} with pattern {file_pattern}")
        
        discovered_files = []
        
        # Search for files matching the expected structure
        for file_path in self.data_root.rglob(file_pattern):
            try:
                data_file = self._parse_file_path(file_path)
                if data_file:
                    discovered_files.append(data_file)
            except Exception as e:
                logger.debug(f"Failed to parse file path {file_path}: {e}")
                continue
        
        # Sort by datetime for consistent processing order
        discovered_files.sort(key=lambda f: (f.year, f.month, f.day, f.hour or 0))
        
        # Cache the results
        self._file_cache = discovered_files
        
        logger.info(f"Discovered {len(discovered_files)} files")
        self._log_discovery_summary(discovered_files)
        
        return self._filter_files(discovered_files, clusters, years, months)
    
    def _parse_file_path(self, file_path: Path) -> Optional[DataFile]:
        """Parse file path to extract metadata"""
        try:
            # Get relative path from data root
            rel_path = file_path.relative_to(self.data_root)
            path_parts = rel_path.parts
            
            # Try to extract date components from path
            year, month, day, hour = self._extract_date_from_path(path_parts)
            
            if year is None or month is None or day is None:
                return None
            
            # Extract cluster from path
            cluster = get_cluster_from_path(file_path)
            
            # Get file size
            size_bytes = file_path.stat().st_size if file_path.exists() else 0
            
            return DataFile(
                path=file_path,
                cluster=cluster,
                year=year,
                month=month,
                day=day,
                hour=hour,
                size_bytes=size_bytes
            )
            
        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_date_from_path(self, path_parts: tuple) -> tuple:
        """Extract year, month, day, hour from path parts"""
        year = month = day = hour = None
        
        # Look for numeric path components that could be dates
        for i, part in enumerate(path_parts):
            # Try to find year (4 digits, 20xx)
            if re.match(r'^20\d{2}$', part):
                year = int(part)
                
                # Look for month in next part
                if i + 1 < len(path_parts):
                    month_part = path_parts[i + 1]
                    if re.match(r'^(0?[1-9]|1[0-2])$', month_part):
                        month = int(month_part)
                        
                        # Look for day in next part
                        if i + 2 < len(path_parts):
                            day_part = path_parts[i + 2]
                            if re.match(r'^(0?[1-9]|[12]\d|3[01])$', day_part):
                                day = int(day_part)
                                break
        
        # Try to extract hour from filename if present
        filename = path_parts[-1]
        hour_match = re.search(r'(\d{2})\.\w+$', filename)
        if hour_match:
            potential_hour = int(hour_match.group(1))
            if 0 <= potential_hour <= 23:
                hour = potential_hour
        
        # Alternative: look for hour patterns in filename
        if hour is None:
            hour_patterns = [
                r'hour[_-]?(\d{1,2})',
                r'h(\d{1,2})',
                r'_(\d{2})\.parquet$'
            ]
            for pattern in hour_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    potential_hour = int(match.group(1))
                    if 0 <= potential_hour <= 23:
                        hour = potential_hour
                        break
        
        return year, month, day, hour
    
    def _filter_files(
        self,
        files: List[DataFile],
        clusters: Optional[Set[ClusterType]] = None,
        years: Optional[Set[int]] = None,
        months: Optional[Set[int]] = None
    ) -> List[DataFile]:
        """Filter files based on criteria"""
        filtered = files
        
        if clusters is not None:
            filtered = [f for f in filtered if f.cluster in clusters]
        
        if years is not None:
            filtered = [f for f in filtered if f.year in years]
            
        if months is not None:
            filtered = [f for f in filtered if f.month in months]
        
        return filtered
    
    def _log_discovery_summary(self, files: List[DataFile]) -> None:
        """Log summary of discovered files"""
        if not files:
            logger.warning("No files discovered")
            return
        
        # Summarize by cluster
        cluster_counts = {}
        for file in files:
            cluster_name = file.cluster.value if file.cluster else "unknown"
            cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + 1
        
        logger.info("Discovery summary:")
        for cluster, count in sorted(cluster_counts.items()):
            logger.info(f"  {cluster}: {count} files")
        
        # Date range
        min_date = min(files, key=lambda f: (f.year, f.month, f.day))
        max_date = max(files, key=lambda f: (f.year, f.month, f.day))
        logger.info(f"  Date range: {min_date.date_key} to {max_date.date_key}")
        
        # Total size
        total_size_gb = sum(f.size_bytes for f in files) / (1024**3)
        logger.info(f"  Total size: {total_size_gb:.1f} GB")
    
    def get_files_by_date(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> List[DataFile]:
        """Get files within date range"""
        files = self.discover_files(**kwargs)
        
        if start_date or end_date:
            filtered = []
            for file in files:
                file_date = date(file.year, file.month, file.day)
                
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                    
                filtered.append(file)
            files = filtered
        
        return files
    
    def get_files_by_cluster(self, cluster: ClusterType, **kwargs) -> List[DataFile]:
        """Get files for specific cluster"""
        return self.discover_files(clusters={cluster}, **kwargs)
    
    def group_files_by_date(self, files: List[DataFile]) -> Dict[str, List[DataFile]]:
        """Group files by date key"""
        groups = {}
        for file in files:
            key = file.date_key
            if key not in groups:
                groups[key] = []
            groups[key].append(file)
        return groups
    
    def group_files_by_cluster(self, files: List[DataFile]) -> Dict[str, List[DataFile]]:
        """Group files by cluster"""
        groups = {}
        for file in files:
            key = file.cluster.value if file.cluster else "unknown"
            if key not in groups:
                groups[key] = []
            groups[key].append(file)
        return groups
    
    def estimate_processing_batches(
        self,
        files: List[DataFile],
        target_batch_size_gb: float = 1.0
    ) -> List[List[DataFile]]:
        """Create processing batches based on estimated memory usage"""
        batches = []
        current_batch = []
        current_size = 0.0
        
        for file in files:
            file_size_gb = file.size_bytes / (1024**3)
            
            # Start new batch if this would exceed target size
            if current_batch and current_size + file_size_gb > target_batch_size_gb:
                batches.append(current_batch)
                current_batch = []
                current_size = 0.0
            
            current_batch.append(file)
            current_size += file_size_gb
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} processing batches "
                   f"(target: {target_batch_size_gb:.1f} GB per batch)")
        
        return batches
    
    def validate_files(self, files: List[DataFile]) -> List[DataFile]:
        """Validate that files exist and are readable"""
        valid_files = []
        
        for file in files:
            try:
                if not file.path.exists():
                    logger.warning(f"File does not exist: {file.path}")
                    continue
                
                if file.path.stat().st_size == 0:
                    logger.warning(f"File is empty: {file.path}")
                    continue
                
                # Try to get basic file info
                file.path.stat()
                valid_files.append(file)
                
            except Exception as e:
                logger.warning(f"File validation failed for {file.path}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_files)}/{len(files)} files")
        return valid_files