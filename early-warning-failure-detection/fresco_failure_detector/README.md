# FRESCO Failure Detection Pipeline

A production-grade early-warning failure detection system for FRESCO HPC dataset. This pipeline answers the research question:

> **RQ-R1:** *Can early-warning signals in telemetry reliably predict imminent job failures?*

## Key Features

- **Multi-horizon prediction**: 5, 15, and 60-minute warning windows
- **No data leakage**: Strict temporal boundaries for realistic evaluation
- **Memory-efficient processing**: Streams large datasets without loading everything into memory
- **Multiple model types**: Rule-based baselines, XGBoost, Logistic Regression, LightGBM
- **Comprehensive evaluation**: AUROC, AUPRC, lead-time analysis, calibration metrics
- **Production-ready**: CLI interface, caching, robust error handling

## Installation

### Requirements

- Python 3.10+ (3.11 recommended)
- 90 GB RAM available
- ~35 GB free disk space (pipeline uses <25 GB working set)
- Red Hat Linux (developed on WSL2)

### Setup

```bash
# Clone the repository
cd fresco_failure_detector

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e .[dev]

# Verify installation
fresco-fd status
```

### Dependencies

Core dependencies (automatically installed):
- pandas, pyarrow, numpy (data processing)
- scikit-learn, xgboost, lightgbm (machine learning)
- typer, rich (CLI interface)
- matplotlib, seaborn (visualization)

## Quick Start

### 1. Check System Status

```bash
fresco-fd status
```

This validates your installation and shows configuration details.

### 2. Prepare Dataset

```bash
fresco-fd prepare \
  --data-root /home/dynamo/a/jmckerra/projects/fresco-analysis/data \
  --clusters all \
  --years 2022,2023 \
  --horizons 5,15,60 \
  --output ./artifacts/datasets/
```

### 3. Train Models

```bash
fresco-fd train \
  --dataset ./artifacts/datasets/ \
  --models xgb,logreg,rule \
  --calibrate \
  --output ./artifacts/models/
```

### 4. Evaluate Models

```bash
fresco-fd eval \
  --models ./artifacts/models/ \
  --dataset ./artifacts/datasets/ \
  --output ./artifacts/reports/
```

## Data Format

### Expected Directory Structure

```
/home/dynamo/a/jmckerra/projects/fresco-analysis/data/
├── [cluster]/
│   └── year/
│       └── month/
│           └── day/
│               └── *.parquet
```

### Required Columns

**Job Metadata:**
- `jid` - Job ID (unique identifier)
- `submit_time`, `start_time`, `end_time` - ISO timestamps
- `exitcode` - Job exit code (mapped to failure classes)
- `ncores`, `nhosts`, `queue`, `account` - Resource allocation
- `username` or `job_username` - User information

**Telemetry Time Series:**
- `time` - Measurement timestamp
- `value_cpuuser` - CPU utilization (%)
- `value_memused` - Memory usage (GB)
- `value_nfs` - NFS I/O (MB/s)  
- `value_block` - Block I/O (GB/s)
- `value_gpu` - GPU utilization (%) [Anvil only]

## CLI Commands

### Data Preparation

```bash
fresco-fd prepare [OPTIONS]
```

**Key Options:**
- `--data-root PATH` - Root directory containing FRESCO data
- `--clusters LIST` - Target clusters (all, conte, stampede, anvil)
- `--years LIST` - Year filter (e.g., 2022,2023)
- `--horizons LIST` - Prediction horizons in minutes (5,15,60)
- `--neg-multiplier INT` - Negative examples per positive (default: 3)
- `--dry-run` - Show what would be done without executing

### Model Training

```bash
fresco-fd train [OPTIONS]
```

**Key Options:**
- `--dataset PATH` - Directory with prepared datasets
- `--models LIST` - Model types (xgb, logreg, lgb, rule)
- `--calibrate/--no-calibrate` - Apply probability calibration
- `--output PATH` - Output directory for trained models

### Status and Diagnostics

```bash
fresco-fd status
```

Shows installation status, configuration, and system resources.

## Architecture

### Core Components

1. **Data Discovery** (`io_discovery.py`) - Lazy file discovery and cluster detection
2. **Streaming Reader** (`io_reader.py`) - Memory-efficient parquet processing
3. **Label Generation** (`labeling.py`) - No-leakage example creation
4. **Feature Engineering** (`features.py`) - Rolling window statistics
5. **Model Training** (`models.py`) - ML models with calibration
6. **Evaluation** (`eval.py`) - Comprehensive metrics and analysis

### Feature Engineering

**Per-metric features** (CPU, Memory, I/O, GPU):
- **Level**: mean, median, last values
- **Dynamics**: slope, acceleration, percent change
- **Variability**: standard deviation, IQR, coefficient of variation
- **Extremes**: min, max, time over thresholds, spike counts
- **Temporal**: autocorrelation, burstiness patterns

**Cross-metric couplings**:
- Memory pressure + CPU idle + I/O stall patterns
- GPU utilization vs CPU patterns
- Correlated anomalies across metrics

**Context features**:
- Job size (cores, hosts), cluster type
- Job age and estimated completion ratio

### No-Leakage Guarantee

The pipeline ensures strict temporal boundaries:

- **Positive examples**: Created at `t = failure_time - H` for horizon H
- **Negative examples**: Sampled from safe windows excluding last H minutes
- **Feature computation**: Only uses data up to time t
- **Temporal splits**: Train/validation/test based on time, not random

## Success Criteria

The pipeline demonstrates reliable failure prediction when achieving:

**Primary Metrics (at least one horizon):**
- AUPRC ≥ 0.20 and AUROC ≥ 0.75
- Recall ≥ 50-70% at FPR ≤ 1%
- Median lead time ≥ 10-15 minutes

**Robustness:**
- Consistent performance on temporal hold-out
- Reasonable cross-cluster generalization  
- Well-calibrated probabilities (Brier < 0.25, ECE < 0.1)

**Interpretability:**
- Feature importance aligns with domain knowledge
- Top signals: memory pressure, I/O stalls, CPU collapse

## Configuration

### Key Configuration Files

- `config.py` - Core configuration parameters
- `pyproject.toml` - Package and dependency configuration
- `Makefile` - Common development tasks

### Customization Points

- **Exit code mappings**: Add new failure classes in `EXITCODE_MAPPINGS`
- **Feature windows**: Modify time windows in `FEATURE_WINDOWS`
- **Model parameters**: Adjust hyperparameters in `MODEL_CONFIG`
- **Memory thresholds**: Cluster-specific limits in `CLUSTER_CONFIGS`

## Memory Management

The pipeline is designed for systems with limited memory:

- **Streaming processing**: Never loads entire dataset
- **Lazy file discovery**: Only processes files as needed
- **Intermediate caching**: Temporary results saved as Parquet/Feather
- **Memory monitoring**: Automatic garbage collection and warnings
- **Configurable limits**: Soft memory caps and worker limits

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce `--max-workers` or set memory limits
- Enable `--keep-cache` to avoid recomputation
- Process smaller date ranges

**Missing Data:**
- Check data root path exists and has correct permissions
- Verify expected directory structure (year/month/day)
- Use `--dry-run` to validate file discovery

**Model Training Failures:**
- Ensure sufficient training examples (>100 per class)
- Check for missing feature columns
- Try simpler models (rule, logreg) first

**Performance Issues:**
- Use SSD storage for better I/O performance
- Enable caching for repeated experiments
- Parallelize with appropriate worker count

### Debug Commands

```bash
# Validate installation and data access
fresco-fd status

# Test data discovery without processing
fresco-fd prepare --dry-run --data-root /path/to/data

# Check dataset split sizes
ls -la ./artifacts/datasets/
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run full test suite
make test

# Run tests without coverage
make test-fast

# Code quality checks
make lint
make type-check
```

### Project Structure

```
fresco_failure_detector/
├── fresco_fd/           # Main package
│   ├── config.py        # Configuration
│   ├── io_discovery.py  # File discovery
│   ├── io_reader.py     # Streaming reader
│   ├── labeling.py      # Label generation
│   ├── features.py      # Feature engineering
│   ├── models.py        # ML models
│   └── cli.py          # CLI interface
├── tests/              # Test suite
├── artifacts/          # Generated outputs
└── scripts/           # Utility scripts
```

## Performance Expectations

**Typical Processing Times:**
- Data preparation: 2-4 hours for 1 year of data
- Feature engineering: 1-2 hours per horizon
- Model training: 10-30 minutes per model
- Evaluation: 5-10 minutes per horizon

**Resource Usage:**
- Peak memory: 15-25 GB (depending on dataset size)
- Disk usage: 10-20 GB temporary files
- CPU utilization: Scales with `--max-workers`

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{fresco_failure_detection,
  title={FRESCO Early-Warning Failure Detection Pipeline},
  author={FRESCO Analysis Team},
  year={2024},
  url={https://github.com/your-repo/fresco-failure-detector}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check this README and the troubleshooting section
2. Review the implementation plan in `IMPLEMENTATION_PLAN.md`
3. Submit issues with detailed error messages and configuration
4. Include system information and data characteristics