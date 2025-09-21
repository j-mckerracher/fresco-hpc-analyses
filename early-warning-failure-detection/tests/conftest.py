"""
Pytest configuration for FRESCO failure detection tests.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files"""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def ensure_test_data_dir(test_data_dir):
    """Ensure test data directory exists"""
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir

# Set random seeds for reproducible tests
np.random.seed(42)
pl.set_random_seed(42)

# Configure pandas display options for better test output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)