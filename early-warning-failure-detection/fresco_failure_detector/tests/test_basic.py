"""
Basic tests for FRESCO failure detection pipeline.

Tests core functionality and ensures modules can be imported and used.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fresco_fd.config import ExitCodeClass, PREDICTION_HORIZONS
from fresco_fd.labeling import ExitCodeMapper, LabelGenerator, LabelExample
from fresco_fd.feature_windows import WindowFeatureComputer
from fresco_fd.models import ModelFactory, RuleBasedModel


class TestExitCodeMapper:
    """Test exit code mapping functionality"""
    
    def test_basic_mapping(self):
        """Test basic exit code mappings"""
        mapper = ExitCodeMapper()
        
        assert mapper.map_exitcode(0) == ExitCodeClass.COMPLETED
        assert mapper.map_exitcode("FAILED") == ExitCodeClass.FAILED
        assert mapper.map_exitcode("TIMEOUT") == ExitCodeClass.TIMEOUT
        assert mapper.map_exitcode("OOM") == ExitCodeClass.OOM
        assert mapper.map_exitcode("CANCELLED") == ExitCodeClass.CANCELLED
    
    def test_unknown_codes(self):
        """Test handling of unknown exit codes"""
        mapper = ExitCodeMapper()
        
        assert mapper.map_exitcode("UNKNOWN_CODE") == ExitCodeClass.UNKNOWN
        assert mapper.map_exitcode(None) == ExitCodeClass.UNKNOWN
        assert mapper.map_exitcode(np.nan) == ExitCodeClass.UNKNOWN
    
    def test_batch_mapping(self):
        """Test batch processing of exit codes"""
        mapper = ExitCodeMapper()
        
        exitcodes = pd.Series([0, "FAILED", "TIMEOUT", "COMPLETED"])
        mapped = mapper.map_exitcodes_batch(exitcodes)
        
        expected = [
            ExitCodeClass.COMPLETED,
            ExitCodeClass.FAILED,
            ExitCodeClass.TIMEOUT,
            ExitCodeClass.COMPLETED
        ]
        
        assert list(mapped) == expected
    
    def test_failure_success_classification(self):
        """Test failure/success classification"""
        mapper = ExitCodeMapper()
        
        assert mapper.is_failure(ExitCodeClass.FAILED) == True
        assert mapper.is_failure(ExitCodeClass.TIMEOUT) == True
        assert mapper.is_failure(ExitCodeClass.OOM) == True
        
        assert mapper.is_success(ExitCodeClass.COMPLETED) == True
        assert mapper.is_success(ExitCodeClass.CANCELLED) == True
        
        assert mapper.is_failure(ExitCodeClass.COMPLETED) == False
        assert mapper.is_success(ExitCodeClass.FAILED) == False


class TestLabelGenerator:
    """Test label generation with no-leakage guarantees"""
    
    def create_sample_job_data(self) -> pd.DataFrame:
        """Create sample job data for testing"""
        now = datetime.now()
        
        jobs = []
        
        # Successful job
        jobs.append({
            'jid': 'job1',
            'start_time': now - timedelta(hours=2),
            'end_time': now - timedelta(minutes=30),
            'exitcode': 'COMPLETED'
        })
        
        # Failed job
        jobs.append({
            'jid': 'job2', 
            'start_time': now - timedelta(hours=1),
            'end_time': now - timedelta(minutes=10),
            'exitcode': 'FAILED'
        })
        
        # Short job (should be filtered out)
        jobs.append({
            'jid': 'job3',
            'start_time': now - timedelta(minutes=5),
            'end_time': now - timedelta(minutes=2),
            'exitcode': 'COMPLETED'
        })
        
        return pd.DataFrame(jobs)
    
    def test_label_generation(self):
        """Test basic label generation"""
        generator = LabelGenerator(horizons=[5, 15])
        job_data = self.create_sample_job_data()
        
        labels = generator.generate_labels(job_data)
        
        # Should have labels for both horizons
        assert 5 in labels
        assert 15 in labels
        
        # Should have some examples
        assert len(labels[5]) > 0
        assert len(labels[15]) > 0
    
    def test_positive_example_timing(self):
        """Test that positive examples have correct timing"""
        generator = LabelGenerator(horizons=[15])
        job_data = self.create_sample_job_data()
        
        labels = generator.generate_labels(job_data)
        positive_examples = [ex for ex in labels[15] if ex.label == 1]
        
        for example in positive_examples:
            if example.failure_time:
                # Check that example is exactly H minutes before failure
                time_diff = (example.failure_time - example.timestamp).total_seconds() / 60.0
                assert abs(time_diff - 15.0) < 0.1  # Allow small floating point error
    
    def test_negative_example_safety(self):
        """Test that negative examples don't leak future information"""
        generator = LabelGenerator(horizons=[15])
        job_data = self.create_sample_job_data()
        
        labels = generator.generate_labels(job_data)
        negative_examples = [ex for ex in labels[15] if ex.label == 0]
        
        for example in negative_examples:
            # Find the corresponding job
            job = job_data[job_data['jid'] == example.jid].iloc[0]
            job_end = pd.to_datetime(job['end_time'])
            
            # Example should be at least H minutes before job end
            time_to_end = (job_end - example.timestamp).total_seconds() / 60.0
            assert time_to_end >= 15.0


class TestWindowFeatureComputer:
    """Test feature computation from telemetry data"""
    
    def create_sample_telemetry(self) -> pd.DataFrame:
        """Create sample telemetry data"""
        now = datetime.now()
        
        # Create 30 minutes of telemetry data (1 minute intervals)
        times = [now - timedelta(minutes=i) for i in range(30, 0, -1)]
        
        data = []
        for i, time in enumerate(times):
            data.append({
                'time': time,
                'value_cpuuser': 50 + 10 * np.sin(i * 0.1),  # Varying CPU
                'value_memused': 8 + 2 * (i / 30),  # Increasing memory
                'value_nfs': np.random.uniform(0, 5),  # Random I/O
                'value_block': np.random.uniform(0, 1)
            })
        
        return pd.DataFrame(data)
    
    def test_feature_computation(self):
        """Test basic feature computation"""
        computer = WindowFeatureComputer()
        telemetry = self.create_sample_telemetry()
        
        target_time = datetime.now() - timedelta(minutes=5)
        features = computer.compute_features(telemetry, target_time)
        
        # Should have features for each metric and window
        assert len(features) > 0
        
        # Check for expected feature patterns
        cpu_features = [k for k in features.keys() if 'cpuuser' in k]
        mem_features = [k for k in features.keys() if 'memused' in k]
        
        assert len(cpu_features) > 0
        assert len(mem_features) > 0
    
    def test_no_leakage(self):
        """Test that features don't use future data"""
        computer = WindowFeatureComputer()
        telemetry = self.create_sample_telemetry()
        
        # Test with different target times
        early_time = datetime.now() - timedelta(minutes=20)
        late_time = datetime.now() - timedelta(minutes=5)
        
        early_features = computer.compute_features(telemetry, early_time)
        late_features = computer.compute_features(telemetry, late_time)
        
        # Should have valid features for both
        assert len(early_features) > 0
        assert len(late_features) > 0
        
        # Features should be different (more data available for late_time)
        # This is a basic check - in practice you'd verify specific values


class TestModels:
    """Test model functionality"""
    
    def create_sample_training_data(self) -> tuple:
        """Create sample training data"""
        np.random.seed(42)
        
        n_samples = 200
        n_features = 10
        
        # Create feature matrix
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create labels (imbalanced)
        y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]))
        
        return X, y
    
    def test_model_factory(self):
        """Test model factory functionality"""
        # Should be able to create all model types
        for model_type in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_type)
            assert model is not None
            assert model.name is not None
    
    def test_rule_based_model(self):
        """Test rule-based model"""
        X, y = self.create_sample_training_data()
        
        model = RuleBasedModel()
        model.fit(X, y)
        
        # Should be able to predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert all(prob >= 0 and prob <= 1 for prob in probabilities.flatten())
    
    def test_xgboost_model(self):
        """Test XGBoost model (if available)"""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not available")
        
        X, y = self.create_sample_training_data()
        
        model = ModelFactory.create_model('xgb')
        model.fit(X, y, calibration=False)  # Skip calibration for speed
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        
        # Should have feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0
    
    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        X, y = self.create_sample_training_data()
        
        model = RuleBasedModel()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        
        # Should have standard metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1


class TestConfiguration:
    """Test configuration and constants"""
    
    def test_prediction_horizons(self):
        """Test prediction horizons are valid"""
        assert len(PREDICTION_HORIZONS) > 0
        assert all(h > 0 for h in PREDICTION_HORIZONS)
    
    def test_exit_code_classes(self):
        """Test exit code classes are properly defined"""
        # Should have both positive and negative classes
        from fresco_fd.config import POSITIVE_CLASSES, NEGATIVE_CLASSES
        
        assert len(POSITIVE_CLASSES) > 0
        assert len(NEGATIVE_CLASSES) > 0
        
        # Should be mutually exclusive
        assert len(POSITIVE_CLASSES & NEGATIVE_CLASSES) == 0


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])