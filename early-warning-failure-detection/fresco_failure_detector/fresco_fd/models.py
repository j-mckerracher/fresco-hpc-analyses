"""
Machine learning models for FRESCO failure detection.

Implements baseline rule-based detectors and ML models with calibration.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb

from .config import MODEL_CONFIG, MEMORY_PRESSURE_THRESHOLDS
from .utils import save_artifact, load_artifact, setup_logging


logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class BaseModel(ABC):
    """Abstract base class for failure detection models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        pass
    
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        save_artifact(self, filepath)
        logger.info(f"Saved {self.name} model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from disk"""
        model = load_artifact(filepath)
        logger.info(f"Loaded {model.name} model from {filepath}")
        return model
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if len(self.predict_proba(X).shape) > 1 else self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        
        return metrics


class RuleBasedModel(BaseModel):
    """
    Rule-based baseline failure detector.
    
    Uses interpretable rules based on telemetry thresholds:
    - High memory pressure + low CPU + low I/O
    - Sudden resource changes
    - Cross-metric anomalies
    """
    
    def __init__(
        self,
        memory_pressure_threshold: float = MEMORY_PRESSURE_THRESHOLDS["critical"],
        cpu_idle_threshold: float = 10.0,
        io_stall_threshold: float = 0.01
    ):
        super().__init__("RuleBasedModel")
        self.memory_pressure_threshold = memory_pressure_threshold
        self.cpu_idle_threshold = cpu_idle_threshold
        self.io_stall_threshold = io_stall_threshold
        self.feature_stats = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit rule thresholds based on data distribution"""
        logger.info("Fitting rule-based model thresholds")
        
        # Calculate feature statistics for adaptive thresholds
        feature_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in feature_columns:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'p75': X[col].quantile(0.75),
                'p90': X[col].quantile(0.90),
                'p95': X[col].quantile(0.95)
            }
        
        self.is_fitted = True
        logger.info("Rule-based model fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Apply rule-based predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate rule-based failure probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(X)
        scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            score = self._evaluate_rules(X.iloc[i])
            scores[i] = score
        
        # Convert scores to probabilities
        probabilities = np.clip(scores, 0, 1)
        
        # Return as binary classification probabilities
        return np.column_stack([1 - probabilities, probabilities])
    
    def _evaluate_rules(self, sample: pd.Series) -> float:
        """Evaluate rules for a single sample"""
        score = 0.0
        
        # Rule 1: Memory pressure rule
        memory_score = self._evaluate_memory_pressure_rule(sample)
        score += memory_score * 0.4
        
        # Rule 2: CPU-Memory coupling rule
        cpu_memory_score = self._evaluate_cpu_memory_coupling_rule(sample)
        score += cpu_memory_score * 0.3
        
        # Rule 3: I/O anomaly rule
        io_score = self._evaluate_io_anomaly_rule(sample)
        score += io_score * 0.2
        
        # Rule 4: Cross-metric anomaly rule
        cross_metric_score = self._evaluate_cross_metric_rule(sample)
        score += cross_metric_score * 0.1
        
        return min(score, 1.0)
    
    def _evaluate_memory_pressure_rule(self, sample: pd.Series) -> float:
        """Evaluate memory pressure rule"""
        score = 0.0
        
        # Look for memory pressure indicators
        memory_pressure_cols = [col for col in sample.index if 'memory_pressure' in col.lower()]
        memory_usage_cols = [col for col in sample.index if 'memused' in col and 'mean' in col]
        
        # Direct memory pressure features
        for col in memory_pressure_cols:
            if pd.notna(sample[col]) and sample[col] > self.memory_pressure_threshold:
                score += 0.5
        
        # High memory usage with time over threshold
        for col in memory_usage_cols:
            if pd.notna(sample[col]):
                # Look for corresponding time_over features
                base_name = col.replace('_mean', '')
                time_over_col = base_name + '_time_over_95pct'
                
                if time_over_col in sample.index and pd.notna(sample[time_over_col]):
                    if sample[time_over_col] > 0.5:  # More than 50% time over 95%
                        score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_cpu_memory_coupling_rule(self, sample: pd.Series) -> float:
        """Evaluate CPU-memory coupling anomalies"""
        score = 0.0
        
        # Look for memory high + CPU low pattern
        coupling_cols = [col for col in sample.index if 'mem_high_cpu_low' in col.lower()]
        
        for col in coupling_cols:
            if pd.notna(sample[col]) and sample[col] > 0:
                score += 0.5
        
        # Look for opposite trends in memory and CPU
        cpu_slope_cols = [col for col in sample.index if 'cpuuser' in col and 'slope' in col]
        memory_slope_cols = [col for col in sample.index if 'memused' in col and 'slope' in col]
        
        if cpu_slope_cols and memory_slope_cols:
            cpu_slope = sample[cpu_slope_cols[0]] if pd.notna(sample[cpu_slope_cols[0]]) else 0
            memory_slope = sample[memory_slope_cols[0]] if pd.notna(sample[memory_slope_cols[0]]) else 0
            
            # Memory increasing while CPU decreasing (potential problem)
            if memory_slope > 0 and cpu_slope < 0:
                score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_io_anomaly_rule(self, sample: pd.Series) -> float:
        """Evaluate I/O anomaly patterns"""
        score = 0.0
        
        # Look for I/O stall indicators
        io_stall_cols = [col for col in sample.index if 'io_high_cpu_low' in col.lower()]
        no_io_cols = [col for col in sample.index if 'time_no_io' in col.lower()]
        
        for col in io_stall_cols:
            if pd.notna(sample[col]) and sample[col] > 0:
                score += 0.4
        
        # High percentage of time with no I/O
        for col in no_io_cols:
            if pd.notna(sample[col]) and sample[col] > 0.8:  # >80% time with no I/O
                score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_cross_metric_rule(self, sample: pd.Series) -> float:
        """Evaluate cross-metric anomalies"""
        score = 0.0
        
        # GPU-CPU coupling (if available)
        gpu_cpu_cols = [col for col in sample.index if 'gpu_high_cpu_low' in col.lower()]
        
        for col in gpu_cpu_cols:
            if pd.notna(sample[col]) and sample[col] > 0:
                score += 0.3
        
        # High variability in critical metrics
        variability_indicators = [
            col for col in sample.index 
            if ('cv' in col or 'std' in col) and ('memused' in col or 'cpuuser' in col)
        ]
        
        for col in variability_indicators:
            if pd.notna(sample[col]):
                col_base = col.replace('_cv', '').replace('_std', '')
                if col_base in self.feature_stats:
                    # High variability compared to typical
                    if sample[col] > self.feature_stats[col_base].get('p90', 0):
                        score += 0.1
        
        return min(score, 1.0)


class XGBoostModel(BaseModel):
    """XGBoost-based failure detector with calibration"""
    
    def __init__(self, **model_params):
        super().__init__("XGBoostModel")
        self.model_params = {**MODEL_CONFIG["xgboost"], **model_params}
        self.model = None
        self.calibrator = None
        self.feature_importance = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            calibration: bool = True,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> None:
        """Fit XGBoost model with optional calibration"""
        logger.info("Training XGBoost model")
        
        # Prepare data
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Set up validation
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_clean = X_val.select_dtypes(include=[np.number]).fillna(0)
            eval_set = [(X_clean, y), (X_val_clean, y_val)]
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(**self.model_params)
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = 10
            fit_params['verbose'] = False
        
        self.model.fit(X_clean, y, **fit_params)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calibration
        if calibration:
            logger.info("Calibrating XGBoost model")
            self.calibrator = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            self.calibrator.fit(X_clean, y)
        
        self.is_fitted = True
        logger.info("XGBoost model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict(X_clean)
        else:
            return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict_proba(X_clean)
        else:
            return self.model.predict_proba(X_clean)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance.copy()


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline with calibration"""
    
    def __init__(self, **model_params):
        super().__init__("LogisticRegressionModel")
        self.model_params = {**MODEL_CONFIG["logistic_regression"], **model_params}
        self.model = None
        self.calibrator = None
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            calibration: bool = True, **kwargs) -> None:
        """Fit logistic regression with calibration"""
        logger.info("Training Logistic Regression model")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        self.feature_names = list(X_clean.columns)
        
        # Train model
        self.model = LogisticRegression(**self.model_params)
        self.model.fit(X_clean, y)
        
        # Calibration
        if calibration:
            logger.info("Calibrating Logistic Regression model")
            self.calibrator = CalibratedClassifierCV(
                self.model, method='sigmoid', cv=3
            )
            self.calibrator.fit(X_clean, y)
        
        self.is_fitted = True
        logger.info("Logistic Regression training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict(X_clean)
        else:
            return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict_proba(X_clean)
        else:
            return self.model.predict_proba(X_clean)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get coefficient-based feature importance"""
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        coefficients = np.abs(self.model.coef_[0])
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coefficients
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LightGBMModel(BaseModel):
    """LightGBM-based failure detector"""
    
    def __init__(self, **model_params):
        super().__init__("LightGBMModel")
        self.model_params = {**MODEL_CONFIG["lightgbm"], **model_params}
        self.model = None
        self.calibrator = None
        self.feature_importance = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            calibration: bool = True,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> None:
        """Fit LightGBM model"""
        logger.info("Training LightGBM model")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Set up validation
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_clean = X_val.select_dtypes(include=[np.number]).fillna(0)
            eval_set = [(X_val_clean, y_val)]
        
        # Train model
        self.model = lgb.LGBMClassifier(**self.model_params)
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = 10
            fit_params['verbose'] = 0
        
        self.model.fit(X_clean, y, **fit_params)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calibration
        if calibration:
            logger.info("Calibrating LightGBM model")
            self.calibrator = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            self.calibrator.fit(X_clean, y)
        
        self.is_fitted = True
        logger.info("LightGBM training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict(X_clean)
        else:
            return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.calibrator:
            return self.calibrator.predict_proba(X_clean)
        else:
            return self.model.predict_proba(X_clean)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        return self.feature_importance.copy()


class ModelFactory:
    """Factory for creating failure detection models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create a model instance"""
        
        if model_type.lower() == 'rule':
            return RuleBasedModel(**kwargs)
        elif model_type.lower() in ['xgb', 'xgboost']:
            return XGBoostModel(**kwargs)
        elif model_type.lower() in ['logreg', 'logistic']:
            return LogisticRegressionModel(**kwargs)
        elif model_type.lower() in ['lgb', 'lightgbm']:
            return LightGBMModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types"""
        return ['rule', 'xgboost', 'logistic', 'lightgbm']


class EnsembleModel(BaseModel):
    """Ensemble of multiple failure detection models"""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        super().__init__("EnsembleModel")
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fit all ensemble models"""
        logger.info(f"Training ensemble of {len(self.models)} models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted voting
        weighted_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble probability averaging"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        probabilities = []
        
        for model in self.models:
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(probabilities[0])
        
        for proba, weight in zip(probabilities, self.weights):
            ensemble_proba += proba * weight
        
        return ensemble_proba