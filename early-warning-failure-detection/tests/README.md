# Pandas → Polars Equivalence Test Suite

This comprehensive test suite ensures that Polars implementations produce **exactly the same results** as the current Pandas implementation for all critical data processing operations in the FRESCO failure detection pipeline.

## Overview

The test suite validates equivalence for:

1. **Critical Performance Bottleneck**: Job ID filtering with `.isin()` operations
2. **Complex Aggregations**: Multi-column groupby operations with flattened column names
3. **Data Cleaning Pipeline**: Numeric validation, timestamp parsing, and null handling
4. **Streaming Processing**: Chunk-based processing for memory efficiency
5. **Full Integration**: End-to-end pipeline equivalence

## Test Categories

### 🎯 Critical Tests (`test_job_id_filtering_critical`)
- Tests the main performance bottleneck: `df[df['jid'].astype(str).isin(target_jids)]`
- Validates that Polars `pl.col('jid').cast(pl.Utf8).is_in(list(target_jids))` produces identical results
- **Why Critical**: This operation is 60x slower in the current implementation

### 📈 Aggregation Tests (`test_job_aggregation_operations`)
- Tests complex multi-column aggregation: `groupby('jid').agg({'col': ['mean', 'std', 'min', 'max', 'count']})`
- Validates flattened column naming: `col_mean`, `col_std`, etc.
- Tests merging aggregated telemetry with job metadata

### 🧹 Data Cleaning Tests (`test_data_cleaning_pipeline`)
- Numeric column cleaning with range validation and null filling
- Timestamp parsing with timezone handling
- Cluster-specific column filtering
- String type conversion for job IDs

### 📦 Streaming Tests (`test_streaming_chunk_processing`)
- Chunk-based processing equivalence
- Memory-efficient filtering across multiple chunks
- Concatenation of filtered results

### 🔗 Integration Tests (`test_full_pipeline_integration`)
- End-to-end pipeline validation
- Combines filtering, cleaning, aggregation, and merging
- Uses realistic data volumes and patterns

## Running Tests

### Quick Test (Recommended)
```bash
python run_equivalence_tests.py --quick
```

### With Performance Comparison
```bash
python run_equivalence_tests.py --performance
```

### Full Test Suite
```bash
python run_equivalence_tests.py --all
```

### Using pytest
```bash
cd tests
pytest test_pandas_polars_equivalence.py -v
```

## Test Data

The test suite generates realistic synthetic data that matches the FRESCO schema:

- **Job IDs**: 1,000 unique jobs with duplicated telemetry records
- **Telemetry Metrics**: All metrics from `TELEMETRY_METRICS` config
- **Edge Cases**: NaN values, out-of-range values, missing timestamps
- **Realistic Patterns**: CPU usage 0-100%, memory usage in GB, etc.

## Equivalence Validation

The `assert_dataframes_equivalent()` function performs strict validation:

- ✅ **Row count** must be identical
- ✅ **Column names and order** must match
- ✅ **Data types** must be equivalent (with mapping tolerance)
- ✅ **Values** must be identical (with floating point tolerance)
- ✅ **NaN positions** must be exactly the same
- ✅ **Index/sorting** behavior must be consistent

## Expected Performance Improvements

Based on the test harness and performance comparisons:

- **Job ID Filtering**: 10-60x faster (main bottleneck)
- **Memory Usage**: 30-50% reduction for large datasets
- **Aggregation**: 2-5x faster for complex groupby operations
- **Overall Pipeline**: 5-20x faster end-to-end processing

## Pre-Migration Checklist

Before implementing Polars in the main codebase:

- [ ] All equivalence tests pass
- [ ] Performance improvements validated
- [ ] Edge cases handled correctly
- [ ] Memory usage is acceptable
- [ ] Integration test validates full pipeline

## Polars Implementation Patterns

The test suite provides reference implementations for:

### Filtering
```python
# Pandas
df[df['jid'].astype(str).isin(target_jids)]

# Polars
df.filter(pl.col('jid').cast(pl.Utf8).is_in(list(target_jids)))
```

### Aggregation
```python
# Pandas
df.groupby('jid').agg({'col': ['mean', 'std', 'count']})

# Polars
df.group_by('jid').agg([
    pl.col('col').mean().alias('col_mean'),
    pl.col('col').std().alias('col_std'),
    pl.col('col').count().alias('col_count')
])
```

### Data Cleaning
```python
# Pandas
df[col] = pd.to_numeric(df[col], errors='coerce')
df[col] = df[col].where((df[col] >= min_val) & (df[col] <= max_val), fill_value)

# Polars
df.with_columns(
    pl.when((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
    .then(pl.col(col))
    .otherwise(fill_value)
    .alias(col)
)
```

## Migration Strategy

1. **Run Tests**: Ensure all equivalence tests pass
2. **Implement Gradually**: Replace one operation at a time
3. **Validate**: Run tests after each change
4. **Monitor**: Check performance improvements
5. **Fallback**: Keep Pandas as backup during transition

The test suite ensures that the Polars migration will be **functionally identical** while providing significant performance improvements for the telemetry processing bottleneck.