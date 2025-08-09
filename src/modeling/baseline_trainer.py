#!/usr/bin/env python3
"""
Phase 1: Baseline Models without Text-Derived Features

This module implements baseline cryptocurrency price prediction models using only:
- Price data (price_usd)
- Technical indicators (price_ma_7, price_volatility)

Hypothesis: Text data significantly improves prediction performance.
This baseline establishes the performance floor without text features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BaselineModelTrainer:
    def __init__(self, data_path, models_dir):
        self.data_path = data_path
        self.models_dir = models_dir
        self.results = {}
        
        # Create baseline directory
        self.baseline_dir = os.path.join(models_dir, 'baseline_phase1')
        os.makedirs(self.baseline_dir, exist_ok=True)
        
    def load_data(self):
        """Load dataset and prepare baseline features (NO TEXT FEATURES)"""
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Baseline features: Only price and technical indicators
        baseline_features = ['price_usd', 'price_ma_7', 'price_volatility']
        
        # Remove rows with missing baseline features
        df_clean = df.dropna(subset=baseline_features)
        
        print(f"✅ Dataset loaded: {len(df_clean)} samples")
        print(f"🔧 Baseline features: {baseline_features}")
        
        # Features (X) - NO TEXT METRICS
        X = df_clean[baseline_features].copy()
        
        # Targets (y) - Both classification and regression
        targets = {
            'direction_1d': df_clean['direction_1d'].copy(),
            'direction_3d': df_clean['direction_3d'].copy(), 
            'direction_7d': df_clean['direction_7d'].copy(),
            'price_change_1d': df_clean['price_change_1d'].copy(),
            'price_change_3d': df_clean['price_change_3d'].copy(),
            'price_change_7d': df_clean['price_change_7d'].copy()
        }
        
        # Remove samples with missing targets
        valid_indices = df_clean.dropna(subset=list(targets.keys())).index
        X = X.loc[valid_indices]
        targets = {k: v.loc[valid_indices] for k, v in targets.items()}
        
        print(f"📈 Final dataset: {len(X)} samples with {len(baseline_features)} baseline features")
        return X, targets, baseline_features
    
    def train_classification_models(self, X, y, target_name, features):
        """Train classification models for direction prediction"""
        print(f"\n🎯 Training CLASSIFICATION models for {target_name}")
        
        # Time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔄 Training {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            # Final training on full dataset
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Metrics
            accuracy = accuracy_score(y, predictions)
            
            results[model_name] = {
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'train_accuracy': float(accuracy),
                'feature_importance': {k: float(v) for k, v in zip(features, model.feature_importances_)}
            }
            
            # Save model
            model_path = os.path.join(self.baseline_dir, f'{model_name}_{target_name}.joblib')
            joblib.dump(model, model_path)
            
            print(f"    ✅ CV Accuracy: {cv_scores.mean():.3f} ±{cv_scores.std():.3f}")
            print(f"    ✅ Train Accuracy: {accuracy:.3f}")
        
        return results
    
    def train_regression_models(self, X, y, target_name, features):
        """Train regression models for price change prediction"""
        print(f"\n📈 Training REGRESSION models for {target_name}")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔄 Training {model_name}...")
            
            # Cross-validation scores (negative MAE)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            
            # Final training
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            
            results[model_name] = {
                'cv_mae_mean': float(-cv_scores.mean()),
                'cv_mae_std': float(cv_scores.std()),
                'train_mae': float(mae),
                'train_mse': float(mse),
                'train_rmse': float(rmse),
                'train_r2': float(r2),
                'feature_importance': {k: float(v) for k, v in zip(features, model.feature_importances_)}
            }
            
            # Save model
            model_path = os.path.join(self.baseline_dir, f'{model_name}_{target_name}.joblib')
            joblib.dump(model, model_path)
            
            print(f"    ✅ CV MAE: {-cv_scores.mean():.3f} ±{cv_scores.std():.3f}")
            print(f"    ✅ Train R²: {r2:.3f}")
        
        return results
    
    def run_baseline_training(self):
        """Execute complete baseline training pipeline"""
        print("🚀 PHASE 1: BASELINE MODEL TRAINING (NO TEXT FEATURES)")
        print("="*60)
        
        # Load data
        X, targets, features = self.load_data()
        
        # Train classification models
        classification_results = {}
        for target in ['direction_1d', 'direction_3d', 'direction_7d']:
            classification_results[target] = self.train_classification_models(
                X, targets[target], target, features
            )
        
        # Train regression models
        regression_results = {}
        for target in ['price_change_1d', 'price_change_3d', 'price_change_7d']:
            regression_results[target] = self.train_regression_models(
                X, targets[target], target, features
            )
        
        # Compile results
        self.results = {
            'phase': 'baseline_phase1',
            'description': 'Baseline models using only price data and technical indicators (NO TEXT FEATURES)',
            'features_used': features,
            'num_features': len(features),
            'dataset_size': len(X),
            'classification_results': classification_results,
            'regression_results': regression_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(self.baseline_dir, 'baseline_phase1_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("📊 PHASE 1 BASELINE RESULTS SUMMARY")
        print("="*60)
        
        print(f"🔧 Features: {self.results['num_features']} baseline features (NO TEXT)")
        print(f"📊 Dataset: {self.results['dataset_size']} samples")
        
        print("\n🎯 CLASSIFICATION PERFORMANCE (Direction Prediction):")
        for target, models in self.results['classification_results'].items():
            print(f"\n  {target.upper()}:")
            for model, metrics in models.items():
                print(f"    {model}: {metrics['cv_accuracy_mean']:.3f} ±{metrics['cv_accuracy_std']:.3f} CV accuracy")
        
        print("\n📈 REGRESSION PERFORMANCE (Price Change Prediction):")
        for target, models in self.results['regression_results'].items():
            print(f"\n  {target.upper()}:")
            for model, metrics in models.items():
                print(f"    {model}: R²={metrics['train_r2']:.3f}, MAE={metrics['cv_mae_mean']:.3f}")
        
        print(f"\n💾 Models saved to: {self.baseline_dir}")
        print("="*60)

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/home/thej/Desktop/CryptoPulse/data/simplified_ml_dataset.csv"
    MODELS_DIR = "/home/thej/Desktop/CryptoPulse/models"
    
    # Initialize and run
    trainer = BaselineModelTrainer(DATA_PATH, MODELS_DIR)
    results = trainer.run_baseline_training()
    
    print("\n🎯 PHASE 1 COMPLETE: Baseline models trained without text features!")
    print("📈 Next: Train enhanced models WITH text features to prove text data value.")