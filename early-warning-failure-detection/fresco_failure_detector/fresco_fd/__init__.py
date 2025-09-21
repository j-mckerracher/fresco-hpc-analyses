"""
FRESCO Failure Detection Pipeline

A production-grade early-warning failure detection system for FRESCO HPC dataset.
Predicts job failures within configurable horizons (5, 15, 60 minutes) using 
telemetry signals and machine learning models.

Research Question: Can early-warning signals in telemetry reliably predict 
imminent job failures?
"""

__version__ = "1.0.0"
__author__ = "FRESCO Analysis Team"

from . import config
from . import utils

__all__ = ["config", "utils"]