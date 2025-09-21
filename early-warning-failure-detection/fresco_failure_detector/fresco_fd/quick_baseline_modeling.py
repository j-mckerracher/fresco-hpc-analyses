#!/usr/bin/env python3
"""
Quick baseline modeling script for FRESCO failure detection.

Usage:
    python quick_baseline_modeling.py --data-path /path/to/dataset --horizon 5
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(data_path, horizon=5):
    """Load dataset and prepare for modeling"""
    print(f"Loading data from {data_path}...")
    
    # Try different possible file structures
    try:
        # Option 1: Single parquet file
        df = pd.read_parquet(data_path)
    except:
        try:
            # Option 2: Directory with multiple parquet files
            df = pd.read_parquet(data_path, engine='pyarrow')
        except:
            # Option 3: CSV format
            df = pd.read_csv(data_path)
    
    print(f"Loaded dataset: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Filter by horizon if column exists
    if 'horizon_minutes' in df.columns:
        df = df[df.horizon_minutes == horizon]
        print(f"Filtered to {horizon}-minute horizon: {df.shape}")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns 
                   if not col.startswith(('label', 'jid', 'timestamp', 'failure_time'))]
    
    X = df[feature_cols]
    y = df['label'] if 'label' in df.columns else df[df.columns[-1]]  # Assume last column is label
    
    print(f"Features: {len(feature_cols)}, Labels: {y.value_counts().to_dict()}")
    return X, y, feature_cols

def quick_baseline_evaluation(X, y, test_size=0.2, random_state=42):
    """Quickly evaluate multiple baseline models"""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training median for test
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for LogReg, original for RF
        X_train_model = X_train_scaled if 'Logistic' in name else X_train
        X_test_model = X_test_scaled if 'Logistic' in name else X_test
        
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        y_proba = model.predict_proba(X_test_model)[:, 1]
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': model.score(X_test_model, y_test),
            'auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {results[name]['accuracy']:.3f}")
        print(f"  AUC: {results[name]['auc']:.3f}")
        print(f"  Classification Report:\n{results[name]['classification_report']}")
    
    return results, X_test, y_test

def plot_model_comparison(results):
    """Create quick comparison plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUC comparison
    models = list(results.keys())
    aucs = [results[model]['auc'] for model in models]
    
    axes[0].bar(models, aucs)
    axes[0].set_title('Model AUC Comparison')
    axes[0].set_ylabel('AUC Score')
    axes[0].set_ylim(0.5, 1.0)
    
    # Confusion matrices
    for i, (name, result) in enumerate(results.items()):
        if i < 2:  # Only plot first 2 models
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Failure'],
                       yticklabels=['Normal', 'Failure'])
            axes[1].set_title(f'Confusion Matrix - {name}')
    
    plt.tight_layout()
    plt.savefig('quick_baseline_results.png', dpi=150, bbox_inches='tight')
    print("Saved results plot to quick_baseline_results.png")

def get_top_features(model, feature_names, top_k=20):
    """Get top features from trained model"""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_k)

def main():
    parser = argparse.ArgumentParser(description='Quick baseline modeling')
    parser.add_argument('--data-path', required=True, help='Path to dataset')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon in minutes')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    
    args = parser.parse_args()
    
    try:
        # Load data
        X, y, feature_cols = load_and_prepare_data(args.data_path, args.horizon)
        
        # Quick evaluation
        results, X_test, y_test = quick_baseline_evaluation(X, y, args.test_size)
        
        # Plot results
        plot_model_comparison(results)
        
        # Show top features
        for name, result in results.items():
            top_features = get_top_features(result['model'], feature_cols)
            if top_features is not None:
                print(f"\nTop 10 Features - {name}:")
                print(top_features.head(10).to_string(index=False))
        
        print(f"\nBaseline modeling complete! Check quick_baseline_results.png for visualizations.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your dataset has the expected structure (features + label column)")

if __name__ == "__main__":
    main()
