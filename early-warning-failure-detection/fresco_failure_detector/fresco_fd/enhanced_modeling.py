#!/usr/bin/env python3
"""
Enhanced modeling for FRESCO failure detection.
Includes XGBoost, feature selection, and temporal modeling approaches.

Usage:
    python enhanced_modeling.py --data-path /path/to/dataset --horizon 5 --mode all
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureSelector:
    """Advanced feature selection for failure detection"""
    
    def __init__(self):
        self.selected_features = {}
        self.selection_results = {}
    
    def statistical_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 100) -> List[str]:
        """Select top k features using statistical tests"""
        print(f"Running statistical feature selection (k={k})...")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected = feature_scores.head(k)['feature'].tolist()
        self.selected_features['statistical'] = selected
        self.selection_results['statistical'] = feature_scores
        
        print(f"Selected {len(selected)} features using statistical selection")
        return selected
    
    def recursive_selection(self, X: pd.DataFrame, y: pd.Series, 
                          estimator=None, cv: int = 3) -> List[str]:
        """Recursive feature elimination with cross-validation"""
        print("Running recursive feature elimination...")
        if estimator is None:
            estimator = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Use RFECV with reduced features for speed
        n_features = min(200, X.shape[1])  # Limit to 200 features for speed
        
        rfe = RFECV(
            estimator=estimator,
            step=10,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Start with top statistical features if available
        if 'statistical' in self.selected_features:
            X_subset = X[self.selected_features['statistical'][:n_features]]
        else:
            X_subset = X.iloc[:, :n_features]
        
        rfe.fit(X_subset, y)
        
        selected = X_subset.columns[rfe.support_].tolist()
        self.selected_features['recursive'] = selected
        self.selection_results['recursive'] = {
            'n_features': rfe.n_features_,
            'ranking': dict(zip(X_subset.columns, rfe.ranking_))
        }
        
        print(f"Selected {len(selected)} features using recursive elimination")
        return selected
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                           threshold: float = 0.001) -> List[str]:
        """Feature selection using tree-based feature importance"""
        print("Running tree-based feature selection...")
        
        # Train XGBoost for feature importance
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected = importance_df[importance_df.importance >= threshold]['feature'].tolist()
        
        self.selected_features['tree_based'] = selected
        self.selection_results['tree_based'] = importance_df
        
        print(f"Selected {len(selected)} features using tree-based importance")
        return selected
    
    def combined_selection(self, X: pd.DataFrame, y: pd.Series, 
                         top_k: int = 100) -> List[str]:
        """Combine multiple selection methods"""
        print("Running combined feature selection...")
        
        # Run all methods
        statistical_features = set(self.statistical_selection(X, y, k=min(150, X.shape[1]//3)))
        tree_features = set(self.tree_based_selection(X, y, threshold=0.001))
        
        # Features that appear in multiple methods get priority
        intersection = statistical_features & tree_features
        union = statistical_features | tree_features
        
        # Prioritize intersection, then add from union
        selected = list(intersection)
        remaining = [f for f in union if f not in selected]
        
        # Add remaining features up to top_k
        if len(selected) < top_k:
            selected.extend(remaining[:top_k - len(selected)])
        
        self.selected_features['combined'] = selected[:top_k]
        
        print(f"Combined selection: {len(intersection)} in intersection, "
              f"{len(selected)} final features")
        return self.selected_features['combined']

class XGBoostEnhanced:
    """Enhanced XGBoost with automatic hyperparameter tuning"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
    
    def train_with_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost with validation set for early stopping"""
        
        print("Training XGBoost with early stopping...")
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.model = model
        return model
    
    def hyperparameter_search(self, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> Dict:
        """Simple grid search for hyperparameters"""
        
        print("Running hyperparameter search...")
        
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'n_estimators': [200, 500]
        }
        
        best_score = 0
        best_params = {}
        
        # Simple grid search (you could use GridSearchCV for more thorough search)
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for n_estimators in param_grid['n_estimators'][:1]:  # Limit for speed
                        
                        model = xgb.XGBClassifier(
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            n_estimators=n_estimators,
                            random_state=42,
                            eval_metric='logloss'
                        )
                        
                        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                        mean_score = scores.mean()
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'subsample': subsample,
                                'n_estimators': n_estimators
                            }
        
        self.best_params = best_params
        print(f"Best params: {best_params}, Best CV AUC: {best_score:.4f}")
        
        return best_params

class TemporalModeling:
    """Temporal modeling approaches for time-series failure prediction"""
    
    def __init__(self):
        self.models = {}
    
    def create_temporal_features(self, df: pd.DataFrame, 
                               job_col: str = 'jid', 
                               time_col: str = 'timestamp') -> pd.DataFrame:
        """Create temporal features from time-series data"""
        
        print("Creating temporal features...")
        
        # Sort by job and time
        df = df.sort_values([job_col, time_col])
        
        # Create lag features for each job
        feature_cols = [col for col in df.columns 
                       if col not in [job_col, time_col, 'label']]
        
        temporal_features = []
        
        for lag in [1, 2, 3]:  # 1, 2, 3 time steps back
            lagged = df.groupby(job_col)[feature_cols].shift(lag)
            lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
            temporal_features.append(lagged)
        
        # Create rolling statistics
        for window in [3, 5]:
            rolling_mean = df.groupby(job_col)[feature_cols].rolling(window).mean().reset_index(0, drop=True)
            rolling_mean.columns = [f"{col}_rolling_mean_{window}" for col in rolling_mean.columns]
            temporal_features.append(rolling_mean)
            
            rolling_std = df.groupby(job_col)[feature_cols].rolling(window).std().reset_index(0, drop=True)
            rolling_std.columns = [f"{col}_rolling_std_{window}" for col in rolling_std.columns]
            temporal_features.append(rolling_std)
        
        # Combine all temporal features
        temporal_df = pd.concat([df] + temporal_features, axis=1)
        
        print(f"Created temporal features: {temporal_df.shape[1] - df.shape[1]} new features")
        return temporal_df
    
    def temporal_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                n_splits: int = 5) -> Dict:
        """Time series cross-validation"""
        
        print("Running temporal cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = {
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            results[name] = {
                'cv_scores': cv_scores,
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std()
            }
            print(f"{name} - Temporal CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results

def comprehensive_modeling_pipeline(data_path: str, horizon: int = 5, 
                                  test_size: float = 0.2, mode: str = 'all'):
    """Run comprehensive modeling pipeline"""
    
    print(f"=== Enhanced Modeling Pipeline ===")
    print(f"Data: {data_path}, Horizon: {horizon}min, Mode: {mode}")
    
    # Load data (reuse loading logic from baseline)
    print("Loading data...")
    try:
        df = pd.read_parquet(data_path)
    except:
        df = pd.read_csv(data_path)
    
    if 'horizon_minutes' in df.columns:
        df = df[df.horizon_minutes == horizon]
    
    # Separate features and labels
    feature_cols = [col for col in df.columns 
                   if not col.startswith(('label', 'jid', 'timestamp', 'failure_time'))]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['label'] if 'label' in df.columns else df[df.columns[-1]]
    
    print(f"Dataset shape: {X.shape}, Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    results = {}
    
    # 1. Feature Selection
    if mode in ['all', 'feature_selection']:
        print("\n=== Feature Selection ===")
        selector = EnhancedFeatureSelector()
        
        # Combined selection
        selected_features = selector.combined_selection(X_train, y_train, top_k=100)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
        results['selected_features'] = selected_features
        results['feature_importance'] = selector.selection_results.get('tree_based')
    else:
        X_train_selected = X_train
        X_test_selected = X_test
    
    # 2. XGBoost Enhancement
    if mode in ['all', 'xgboost']:
        print("\n=== Enhanced XGBoost ===")
        xgb_enhanced = XGBoostEnhanced()
        
        # Split training set for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train with early stopping
        model = xgb_enhanced.train_with_validation(X_tr, y_tr, X_val, y_val)
        
        # Evaluate
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        results['xgboost'] = {
            'model': model,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"XGBoost AUC: {auc:.4f}")
    
    # 3. Temporal Modeling
    if mode in ['all', 'temporal'] and 'timestamp' in df.columns:
        print("\n=== Temporal Modeling ===")
        temporal = TemporalModeling()
        
        # Create temporal features (simplified version)
        # In practice, you'd need proper time series structure
        temporal_results = temporal.temporal_cross_validation(X_train_selected, y_train)
        results['temporal_cv'] = temporal_results
    
    # 4. Model Comparison
    if mode in ['all', 'comparison']:
        print("\n=== Model Comparison ===")
        models = {
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        }
        
        comparison_results = {}
        for name, model in models.items():
            model.fit(X_train_selected, y_train)
            y_proba = model.predict_proba(X_test_selected)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            comparison_results[name] = auc
            
        results['model_comparison'] = comparison_results
        
        print("Model AUC Comparison:")
        for name, auc in comparison_results.items():
            print(f"  {name}: {auc:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced modeling pipeline')
    parser.add_argument('--data-path', required=True, help='Path to dataset')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--mode', choices=['all', 'feature_selection', 'xgboost', 'temporal', 'comparison'],
                       default='all', help='Which components to run')
    
    args = parser.parse_args()
    
    try:
        results = comprehensive_modeling_pipeline(
            args.data_path, args.horizon, args.test_size, args.mode
        )
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Results saved in results dictionary")
        
        # Save key results
        if 'model_comparison' in results:
            print("\nFinal Model Comparison:")
            for model, auc in results['model_comparison'].items():
                print(f"  {model}: {auc:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
