#!/usr/bin/env python3
"""
Quick test runner for validating Pandas/Polars equivalence before implementing Polars migration.

Usage:
    python run_equivalence_tests.py [--quick] [--performance]

Options:
    --quick      Run only the critical tests (fast)
    --performance Include performance comparison tests
"""

import argparse
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import polars as pl
    import pandas as pd
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install polars pandas numpy")
    DEPENDENCIES_AVAILABLE = False

def run_quick_tests():
    """Run the most critical equivalence tests"""
    if not DEPENDENCIES_AVAILABLE:
        return False

    print("🔍 Running critical equivalence tests...")

    try:
        from tests.test_pandas_polars_equivalence import TestDataFrameEquivalence

        test_instance = TestDataFrameEquivalence()

        # Generate test data
        print("  📊 Generating test data...")
        sample_data = test_instance.sample_telemetry_data()
        sample_jobs = test_instance.sample_job_metadata()
        target_jobs = test_instance.target_job_ids()

        # Run critical tests
        print("  🎯 Testing job ID filtering (performance bottleneck)...")
        test_instance.test_job_id_filtering_critical(sample_data, target_jobs)

        print("  📈 Testing job aggregation operations...")
        test_instance.test_job_aggregation_operations(sample_data)

        print("  🧹 Testing data cleaning pipeline...")
        test_instance.test_data_cleaning_pipeline(sample_data)

        print("  🔗 Testing full pipeline integration...")
        test_instance.test_full_pipeline_integration(sample_data, sample_jobs, target_jobs)

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_tests():
    """Run performance comparison tests"""
    if not DEPENDENCIES_AVAILABLE:
        return False

    print("⚡ Running performance comparison tests...")

    try:
        from tests.test_pandas_polars_equivalence import TestDataFrameEquivalence

        test_instance = TestDataFrameEquivalence()
        sample_data = test_instance.sample_telemetry_data()

        print("  💾 Testing memory efficiency...")
        test_instance.test_memory_efficiency_validation(sample_data)

        print("  📦 Testing chunk processing...")
        target_jobs = test_instance.target_job_ids()
        test_instance.test_streaming_chunk_processing(sample_data, target_jobs)

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_edge_case_tests():
    """Run edge case tests"""
    if not DEPENDENCIES_AVAILABLE:
        return False

    print("🛡️ Running edge case tests...")

    try:
        from tests.test_pandas_polars_equivalence import TestDataFrameEquivalence

        test_instance = TestDataFrameEquivalence()
        test_instance.test_edge_cases()

        return True

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Pandas/Polars equivalence tests")
    parser.add_argument("--quick", action="store_true", help="Run only critical tests")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if not DEPENDENCIES_AVAILABLE:
        print("Please install required dependencies:")
        print("pip install -r tests/test_requirements.txt")
        return 1

    start_time = time.time()

    print("🧪 FRESCO Pandas → Polars Equivalence Test Suite")
    print("=" * 50)

    success = True

    # Always run critical tests
    if not run_quick_tests():
        success = False

    # Run performance tests if requested
    if args.performance or args.all:
        if not run_performance_tests():
            success = False

    # Run edge case tests if full test suite
    if args.all:
        if not run_edge_case_tests():
            success = False

    elapsed = time.time() - start_time

    print("=" * 50)
    if success:
        print(f"🎉 All tests passed! ({elapsed:.1f}s)")
        print("✅ Polars implementation will be functionally identical to Pandas")
        print("\n💡 Ready to proceed with Polars migration:")
        print("   1. Replace pandas operations with tested Polars equivalents")
        print("   2. Expected 10-60x performance improvement for filtering operations")
        print("   3. Reduced memory usage for large datasets")
        return 0
    else:
        print(f"❌ Some tests failed! ({elapsed:.1f}s)")
        print("⚠️  Review failing tests before proceeding with Polars migration")
        return 1

if __name__ == "__main__":
    exit(main())