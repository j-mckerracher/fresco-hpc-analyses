.PHONY: help install install-dev test lint format type-check clean run-mini docs

# Default target
help:
	@echo "FRESCO Failure Detection Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install the package and dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run tests without coverage"
	@echo "  lint         Run code linting (ruff)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean up temporary files and caches"
	@echo "  run-mini     Run mini smoke test on sample data"
	@echo "  setup        Initial setup and validation"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing targets
test:
	pytest tests/ --cov=fresco_fd --cov-report=term-missing --cov-report=html -v

test-fast:
	pytest tests/ -v --no-cov

# Code quality targets
lint:
	ruff check fresco_fd tests
	ruff format --check fresco_fd tests

format:
	black fresco_fd tests
	ruff format fresco_fd tests

type-check:
	mypy fresco_fd

# Clean up targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf artifacts/cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Utility targets
run-mini:
	@echo "Running mini smoke test..."
	python -m fresco_fd.cli --help
	@echo "Mini test completed. Run with actual data:"
	@echo "python -m fresco_fd.cli prepare --data-root /path/to/data --dry-run"

setup:
	@echo "Setting up FRESCO Failure Detection Pipeline..."
	@echo "1. Installing package..."
	$(MAKE) install
	@echo "2. Creating artifact directories..."
	mkdir -p artifacts/{datasets,models,reports,plots,policies,cache}
	@echo "3. Validating installation..."
	python -c "import fresco_fd; print('✓ Package installed successfully')"
	python -c "from fresco_fd.config import validate_config; validate_config(); print('✓ Configuration validated')"
	@echo "4. Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  make run-mini    # Test CLI"
	@echo "  make test        # Run test suite"

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify everything works"

check-all: lint type-check test
	@echo "All checks passed!"

# Data processing targets (for actual runs)
prepare-data:
	python -m fresco_fd.cli prepare \
		--data-root /home/dynamo/a/jmckerra/projects/fresco-analysis/data \
		--clusters all \
		--horizons 5,15,60 \
		--out ./artifacts/datasets/

train-models:
	python -m fresco_fd.cli train \
		--dataset ./artifacts/datasets/ \
		--models xgb,logreg,baseline \
		--out ./artifacts/models/

evaluate:
	python -m fresco_fd.cli eval \
		--models ./artifacts/models/ \
		--dataset ./artifacts/datasets/ \
		--out ./artifacts/reports/

generate-plots:
	python -m fresco_fd.cli plots \
		--eval ./artifacts/reports/ \
		--out ./artifacts/plots/

# Full pipeline
run-pipeline: prepare-data train-models evaluate generate-plots
	@echo "Full pipeline completed!"
	@echo "Results available in ./artifacts/"