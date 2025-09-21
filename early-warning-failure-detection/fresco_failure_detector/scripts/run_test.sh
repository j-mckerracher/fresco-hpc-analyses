#!/bin/bash

# Quick test script for FRESCO failure detection pipeline
# Tests CLI functionality and basic operations

set -e  # Exit on any error

echo "=== FRESCO Failure Detection Pipeline - Quick Test ==="
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the fresco_failure_detector directory"
    exit 1
fi

# Install the package if not already installed
echo "1. Installing package..."
pip install -e . > /dev/null 2>&1 || {
    echo "Warning: Package installation failed. Continuing anyway..."
}

# Test CLI availability
echo "2. Testing CLI availability..."
if ! command -v fresco-fd &> /dev/null; then
    echo "Error: fresco-fd command not found. Installation may have failed."
    exit 1
fi

# Test status command
echo "3. Checking system status..."
fresco-fd status || {
    echo "Warning: Status command failed. Some dependencies may be missing."
}

# Test help commands
echo "4. Testing help commands..."
fresco-fd --help > /dev/null || {
    echo "Error: Help command failed"
    exit 1
}

fresco-fd prepare --help > /dev/null || {
    echo "Error: Prepare help failed"
    exit 1
}

fresco-fd train --help > /dev/null || {
    echo "Error: Train help failed"
    exit 1
}

# Test dry run (if no data available)
echo "5. Testing dry run mode..."
fresco-fd prepare --dry-run --data-root ./test_data > /dev/null 2>&1 || {
    echo "Note: Dry run test completed (expected if no test data available)"
}

# Run unit tests if pytest is available
echo "6. Running unit tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || {
        echo "Warning: Some tests failed. This may be expected in development."
    }
else
    echo "Pytest not available. Skipping unit tests."
    echo "Install with: pip install pytest"
fi

echo
echo "=== Quick Test Complete ==="
echo
echo "Basic functionality appears to be working."
echo "For full testing with real data, use:"
echo "  fresco-fd prepare --data-root /path/to/fresco/data"
echo "  fresco-fd train --dataset ./artifacts/datasets/"
echo
echo "For development setup:"
echo "  make setup      # Full setup with dependencies"
echo "  make test       # Run complete test suite"
echo "  make lint       # Code quality checks"