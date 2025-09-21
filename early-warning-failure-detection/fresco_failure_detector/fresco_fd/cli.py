"""
Command-line interface for FRESCO failure detection pipeline.

Provides commands for data preparation, model training, evaluation, and analysis.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import track
import pandas as pd

from . import __version__
from .config import (
    DEFAULT_DATA_ROOT,
    OUTPUT_DIR, 
    PREDICTION_HORIZONS, 
    PROCESSING_CONFIG,
    ClusterType
)
from .utils import setup_logging, log_memory_usage
from .io_discovery import DataDiscovery
from .io_reader import DatasetBuilder
from .labeling import LabelGenerator
from .features import FeatureEngineering
from .sampler import DatasetSampler
from .models import ModelFactory


# Create CLI app
app = typer.Typer(
    name="fresco-fd",
    help="FRESCO Failure Detection Pipeline - Early warning system for HPC job failures",
    add_completion=False
)

# Console for rich output
console = Console()


def version_callback(value: bool):
    if value:
        print(f"FRESCO Failure Detection Pipeline v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose logging"
    )
):
    """FRESCO Failure Detection Pipeline"""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)


@app.command()
def prepare(
    data_root: str = typer.Option(
        DEFAULT_DATA_ROOT,
    OUTPUT_DIR,
        "--data-root",
        help="Root directory containing FRESCO data"
    ),
    clusters: str = typer.Option(
        "all",
        "--clusters", 
        help="Comma-separated cluster names (all, conte, stampede, anvil)"
    ),
    years: Optional[str] = typer.Option(
        None,
        "--years",
        help="Comma-separated years to include (e.g., 2022,2023)"
    ),
    months: Optional[str] = typer.Option(
        None,
        "--months", 
        help="Comma-separated months to include (e.g., 1,2,3)"
    ),
    horizons: str = typer.Option(
        "5,15,60",
        "--horizons",
        help="Comma-separated prediction horizons in minutes"
    ),
    neg_multiplier: int = typer.Option(
        3,
        "--neg-multiplier",
        help="Negative examples per positive example"
    ),
    output_dir: str = typer.Option(
        f"{OUTPUT_DIR}/datasets",
        "--output", "-o",
        help="Output directory for prepared datasets"
    ),
    max_workers: int = typer.Option(
        PROCESSING_CONFIG["max_workers"],
        "--max-workers", 
        help="Number of parallel workers"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing"
    ),
    keep_cache: bool = typer.Option(
        False,
        "--keep-cache",
        help="Keep intermediate cache files"
    )
):
    """Prepare training/test datasets from FRESCO telemetry data"""
    
    logger = logging.getLogger("fresco_fd")
    console.print("[bold blue]FRESCO Failure Detection - Data Preparation[/bold blue]")
    
    try:
        # Parse parameters
        cluster_list = _parse_clusters(clusters)
        year_list = _parse_years(years) if years else None
        month_list = _parse_months(months) if months else None
        horizon_list = _parse_horizons(horizons)
        
        console.print(f"[green]Configuration:[/green]")
        console.print(f"  Data root: {data_root}")
        console.print(f"  Clusters: {cluster_list}")
        console.print(f"  Years: {year_list}")
        console.print(f"  Months: {month_list}")
        console.print(f"  Horizons: {horizon_list} minutes")
        console.print(f"  Negative multiplier: {neg_multiplier}")
        console.print(f"  Output: {output_dir}")
        
        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
            return
        
        # Initialize components
        logger.info("Initializing data pipeline components")
        
        # Data discovery
        discovery = DataDiscovery(data_root)
        dataset_builder = DatasetBuilder(data_root)
        sampler = DatasetSampler()
        
        # Discover files
        console.print("\n[bold]Discovering data files...[/bold]")
        files = discovery.discover_files(
            clusters=set(cluster_list) if cluster_list != ["all"] else None,
            years=set(year_list) if year_list else None,
            months=set(month_list) if month_list else None
        )
        
        if not files:
            console.print("[red]No data files found matching criteria[/red]")
            raise typer.Exit(1)
        
        console.print(f"Found {len(files)} data files")
        
        # Build dataset
        console.print("\n[bold]Building dataset...[/bold]")
        log_memory_usage(logger, "before dataset build")
        
        # Build job metadata (one row per job)
        job_data = dataset_builder.build_job_metadata(
            clusters=set(cluster_list) if cluster_list != ["all"] else None,
            years=set(year_list) if year_list else None,
            months=set(month_list) if month_list else None,
            max_workers=max_workers
        )
        
        if job_data.empty:
            console.print("[red]No job data was successfully processed[/red]")
            raise typer.Exit(1)
        
        console.print(f"Built job metadata for {len(job_data)} unique jobs")
        
        # Build telemetry time-series data (multiple rows per job)
        console.print("\n[bold]Building telemetry time-series...[/bold]")
        telemetry_data = dataset_builder.build_telemetry_timeseries(
            clusters=set(cluster_list) if cluster_list != ["all"] else None,
            years=set(year_list) if year_list else None,
            months=set(month_list) if month_list else None,
            target_jids=set(job_data['jid'].astype(str)),  # Only load telemetry for jobs we have
            max_workers=max_workers
        )
        
        if telemetry_data.empty:
            console.print("[red]No telemetry data was successfully processed[/red]")
            raise typer.Exit(1)
            
        console.print(f"Built telemetry dataset with {len(telemetry_data)} time-series records")
        log_memory_usage(logger, "after dataset build")
        
        # Create temporal splits
        console.print("\n[bold]Creating temporal dataset splits...[/bold]")
        
        splits = sampler.create_temporal_splits(
            job_data=job_data,
            telemetry_data=telemetry_data,
            horizons=horizon_list
        )
        
        if not splits:
            console.print("[red]Failed to create dataset splits[/red]")
            raise typer.Exit(1)
        
        # Save splits
        console.print(f"\n[bold]Saving dataset splits to {output_dir}...[/bold]")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sampler.save_dataset_splits(splits, str(output_path))
        
        # Summary table
        _display_dataset_summary(splits)
        
        console.print(f"\n[green]✓ Dataset preparation completed successfully[/green]")
        console.print(f"[green]  Results saved to: {output_dir}[/green]")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    dataset_dir: str = typer.Option(
        f"{OUTPUT_DIR}/datasets",
        "--dataset", "-d",
        help="Directory containing prepared datasets"
    ),
    models: str = typer.Option(
        "xgb,logreg,rule",
        "--models", "-m",
        help="Comma-separated model types (xgb, logreg, lgb, rule)"
    ),
    horizons: Optional[str] = typer.Option(
        None,
        "--horizons",
        help="Comma-separated horizons to train (default: all available)"
    ),
    calibrate: bool = typer.Option(
        True,
        "--calibrate/--no-calibrate",
        help="Apply probability calibration"
    ),
    output_dir: str = typer.Option(
        f"{OUTPUT_DIR}/models",
        "--output", "-o",
        help="Output directory for trained models"
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        help="Number of parallel workers for training"
    )
):
    """Train failure detection models"""
    
    logger = logging.getLogger("fresco_fd")
    console.print("[bold blue]FRESCO Failure Detection - Model Training[/bold blue]")
    
    try:
        # Parse parameters
        model_list = models.split(",")
        horizon_list = _parse_horizons(horizons) if horizons else None
        
        console.print(f"[green]Configuration:[/green]")
        console.print(f"  Dataset directory: {dataset_dir}")
        console.print(f"  Models: {model_list}")
        console.print(f"  Horizons: {horizon_list or 'all available'}")
        console.print(f"  Calibration: {calibrate}")
        console.print(f"  Output directory: {output_dir}")
        
        # Load datasets
        console.print("\n[bold]Loading datasets...[/bold]")
        
        sampler = DatasetSampler()
        
        # Discover available datasets
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
            raise typer.Exit(1)
        
        # Find available horizons
        available_horizons = []
        for h_dir in dataset_path.glob("dataset_h*"):
            if h_dir.is_dir():
                horizon = int(h_dir.name.replace("dataset_h", ""))
                available_horizons.append(horizon)
        
        if not available_horizons:
            console.print(f"[red]No dataset splits found in {dataset_dir}[/red]")
            raise typer.Exit(1)
        
        # Use specified horizons or all available
        target_horizons = horizon_list if horizon_list else available_horizons
        target_horizons = [h for h in target_horizons if h in available_horizons]
        
        if not target_horizons:
            console.print("[red]No matching horizons found[/red]")
            raise typer.Exit(1)
        
        console.print(f"Training models for horizons: {target_horizons}")
        
        # Load splits
        splits = sampler.load_dataset_splits(dataset_dir, target_horizons)
        
        if not splits:
            console.print("[red]Failed to load dataset splits[/red]")
            raise typer.Exit(1)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Train models for each horizon
        for horizon in target_horizons:
            if horizon not in splits:
                console.print(f"[yellow]Skipping horizon {horizon} (not available)[/yellow]")
                continue
                
            console.print(f"\n[bold]Training models for {horizon}-minute horizon...[/bold]")
            
            split = splits[horizon]
            horizon_output = output_path / f"h{horizon}"
            horizon_output.mkdir(exist_ok=True)
            
            # Prepare features
            feature_eng = FeatureEngineering()
            feature_eng.fit_preprocessors(split.train_features)
            
            X_train = feature_eng.transform_features(split.train_features)
            X_val = feature_eng.transform_features(split.val_features)
            y_train = split.train_features['label']
            y_val = split.val_features['label']
            
            # Train each model type
            for model_type in model_list:
                console.print(f"  Training {model_type} model...")
                
                try:
                    model = ModelFactory.create_model(model_type)
                    
                    # Train with validation data for tree models
                    if model_type in ['xgb', 'lgb']:
                        model.fit(
                            X_train, y_train,
                            calibration=calibrate,
                            validation_data=(X_val, y_val)
                        )
                    else:
                        model.fit(X_train, y_train, calibration=calibrate)
                    
                    # Save model
                    model_path = horizon_output / f"{model_type}_model.joblib"
                    model.save(str(model_path))
                    
                    # Save feature importance if available
                    if hasattr(model, 'get_feature_importance'):
                        importance = model.get_feature_importance()
                        importance_path = horizon_output / f"{model_type}_importance.csv"
                        importance.to_csv(importance_path, index=False)
                    
                    console.print(f"    ✓ {model_type} model saved to {model_path}")
                    
                except Exception as e:
                    console.print(f"    [red]✗ {model_type} model failed: {e}[/red]")
            
            # Save preprocessors
            feature_eng.save_preprocessors(str(horizon_output / "preprocessors.joblib"))
        
        console.print(f"\n[green]✓ Model training completed successfully[/green]")
        console.print(f"[green]  Models saved to: {output_dir}[/green]")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show pipeline status and validate installation"""
    console.print("[bold blue]FRESCO Failure Detection - Status[/bold blue]")
    
    # Check installation
    console.print("\n[bold]Installation Status:[/bold]")
    
    status_table = Table(show_header=True, header_style="bold magenta")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Details")
    
    # Check core modules
    try:
        from . import config, utils, io_discovery, labeling, features, models
        status_table.add_row("Core Modules", "[green]✓[/green]", "All modules loaded")
    except ImportError as e:
        status_table.add_row("Core Modules", "[red]✗[/red]", f"Import error: {e}")
    
    # Check dependencies
    try:
        import pandas, numpy, sklearn, xgboost, lightgbm
        status_table.add_row("ML Dependencies", "[green]✓[/green]", "All dependencies available")
    except ImportError as e:
        status_table.add_row("ML Dependencies", "[red]✗[/red]", f"Missing: {e}")
    
    # Check data directory
    data_path = Path(DEFAULT_DATA_ROOT)
    if data_path.exists():
        status_table.add_row("Default Data Root", "[green]✓[/green]", f"Found: {data_path}")
    else:
        status_table.add_row("Default Data Root", "[yellow]?[/yellow]", f"Not found: {data_path}")
    
    # Check artifacts directory
    artifacts_path = Path(OUTPUT_DIR)
    if artifacts_path.exists():
        datasets = len(list(artifacts_path.glob("datasets/dataset_h*")))
        models = len(list(artifacts_path.glob("models/h*/**/*.joblib")))
        status_table.add_row("Artifacts", "[green]✓[/green]", f"Datasets: {datasets}, Models: {models}")
    else:
        status_table.add_row("Artifacts", "[yellow]?[/yellow]", "No artifacts directory")
    
    console.print(status_table)
    
    # Configuration summary
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Default horizons: {PREDICTION_HORIZONS} minutes")
    console.print(f"  Supported clusters: {[c.value for c in ClusterType]}")
    console.print(f"  Available models: {ModelFactory.get_available_models()}")
    
    console.print(f"\n[bold]Memory Usage:[/bold]")
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    console.print(f"  RSS: {memory_info.rss / 1024**3:.1f} GB")
    console.print(f"  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")


def _parse_clusters(clusters_str: str) -> List[str]:
    """Parse cluster specification"""
    if clusters_str.lower() == "all":
        return ["all"]
    
    cluster_names = [c.strip() for c in clusters_str.split(",")]
    valid_clusters = [c.value for c in ClusterType]
    
    for cluster in cluster_names:
        if cluster not in valid_clusters:
            raise ValueError(f"Invalid cluster: {cluster}. Valid clusters: {valid_clusters}")
    
    return cluster_names


def _parse_years(years_str: str) -> List[int]:
    """Parse year specification"""
    try:
        years = [int(y.strip()) for y in years_str.split(",")]
        for year in years:
            if not (2010 <= year <= 2030):
                raise ValueError(f"Invalid year: {year}")
        return years
    except ValueError as e:
        raise ValueError(f"Invalid year specification: {years_str}") from e


def _parse_months(months_str: str) -> List[int]:
    """Parse month specification"""
    try:
        months = [int(m.strip()) for m in months_str.split(",")]
        for month in months:
            if not (1 <= month <= 12):
                raise ValueError(f"Invalid month: {month}")
        return months
    except ValueError as e:
        raise ValueError(f"Invalid month specification: {months_str}") from e


def _parse_horizons(horizons_str: str) -> List[int]:
    """Parse horizon specification"""
    try:
        horizons = [int(h.strip()) for h in horizons_str.split(",")]
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError(f"Horizon must be positive: {horizon}")
        return horizons
    except ValueError as e:
        raise ValueError(f"Invalid horizon specification: {horizons_str}") from e


def _display_dataset_summary(splits: dict) -> None:
    """Display dataset summary table"""
    console.print("\n[bold]Dataset Summary:[/bold]")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Horizon (min)", justify="center")
    summary_table.add_column("Train", justify="right")
    summary_table.add_column("Validation", justify="right") 
    summary_table.add_column("Test", justify="right")
    summary_table.add_column("Positive Rate", justify="center")
    
    for horizon, split in splits.items():
        train_pos_rate = split.split_info['train_label_dist'].get(1, 0) / split.train_size
        
        summary_table.add_row(
            str(horizon),
            str(split.train_size),
            str(split.val_size),
            str(split.test_size),
            f"{train_pos_rate:.1%}"
        )
    
    console.print(summary_table)


if __name__ == "__main__":
    app()