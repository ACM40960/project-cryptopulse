#!/usr/bin/env python3
"""
Phase 2: Enhanced Models with Text-Derived Features

This module implements enhanced cryptocurrency price prediction models using:
- Price data (price_usd) 
- Technical indicators (price_ma_7, price_volatility)
- Text-derived sentiment features (12 features from social media analysis)

Hypothesis: Adding text sentiment features significantly improves prediction performance.
This phase demonstrates the value of text-based social media analysis.
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

class EnhancedModelTrainer:
    def __init__(self, data_path, models_dir):
        self.data_path = data_path
        self.models_dir = models_dir
        self.results = {}
        
        # Create enhanced directory
        self.enhanced_dir = os.path.join(models_dir, 'enhanced_phase2')
        os.makedirs(self.enhanced_dir, exist_ok=True)
        
    def load_data(self):
        """Load dataset and prepare enhanced features (WITH TEXT FEATURES)"""
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Enhanced features: Price + Technical + Text-derived sentiment
        baseline_features = ['price_usd', 'price_ma_7', 'price_volatility']
        text_features = [
            'content_length_max', 'content_length_mean', 'num_comments_sum',
            'volatility_score_reddit', 'volatility_score_mean', 'relevance_score_max',
            'echo_score_mean', 'engagement_sum', 'echo_score_reddit',
            'echo_score_max', 'volatility_score_max', 'engagement_mean'
        ]
        
        enhanced_features = baseline_features + text_features
        
        # Remove rows with missing enhanced features
        df_clean = df.dropna(subset=enhanced_features)
        
        print(f"✅ Dataset loaded: {len(df_clean)} samples")
        print(f"🔧 Enhanced features: {len(enhanced_features)} total")
        print(f"   📊 Baseline features: {len(baseline_features)} (price + technical)")
        print(f"   📝 Text features: {len(text_features)} (sentiment derived)")
        
        # Features (X) - WITH TEXT METRICS
        X = df_clean[enhanced_features].copy()
        
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
        
        print(f"📈 Final dataset: {len(X)} samples with {len(enhanced_features)} enhanced features")
        return X, targets, enhanced_features, baseline_features, text_features
    
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
            model_path = os.path.join(self.enhanced_dir, f'{model_name}_{target_name}.joblib')
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
            model_path = os.path.join(self.enhanced_dir, f'{model_name}_{target_name}.joblib')
            joblib.dump(model, model_path)
            
            print(f"    ✅ CV MAE: {-cv_scores.mean():.3f} ±{cv_scores.std():.3f}")
            print(f"    ✅ Train R²: {r2:.3f}")
        
        return results
    
    def analyze_feature_importance(self, classification_results, regression_results, baseline_features, text_features):
        """Analyze feature importance to show text feature impact"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Aggregate feature importance across all models
        all_importances = {}
        model_count = 0
        
        # Collect from classification models
        for target, models in classification_results.items():
            for model_name, metrics in models.items():
                model_count += 1
                for feature, importance in metrics['feature_importance'].items():
                    if feature not in all_importances:
                        all_importances[feature] = 0
                    all_importances[feature] += importance
        
        # Collect from regression models
        for target, models in regression_results.items():
            for model_name, metrics in models.items():
                model_count += 1
                for feature, importance in metrics['feature_importance'].items():
                    if feature not in all_importances:
                        all_importances[feature] = 0
                    all_importances[feature] += importance
        
        # Average importance
        avg_importances = {k: v/model_count for k, v in all_importances.items()}
        
        # Separate baseline vs text features
        baseline_importance = sum(avg_importances.get(f, 0) for f in baseline_features)
        text_importance = sum(avg_importances.get(f, 0) for f in text_features)
        
        print(f"📊 Average Feature Importance:")
        print(f"   🔧 Baseline features: {baseline_importance:.3f}")
        print(f"   📝 Text features: {text_importance:.3f}")
        print(f"   📈 Text contribution: {text_importance/(baseline_importance+text_importance)*100:.1f}%")
        
        # Top features
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            feature_type = "📝 TEXT" if feature in text_features else "🔧 BASELINE"
            print(f"   {i+1}. {feature}: {importance:.3f} {feature_type}")
        
        return {
            'baseline_importance': baseline_importance,
            'text_importance': text_importance,
            'text_contribution_percent': text_importance/(baseline_importance+text_importance)*100,
            'top_features': sorted_features[:10]
        }
    
    def run_enhanced_training(self):
        """Execute complete enhanced training pipeline"""
        print("🚀 PHASE 2: ENHANCED MODEL TRAINING (WITH TEXT FEATURES)")
        print("="*65)
        
        # Load data
        X, targets, enhanced_features, baseline_features, text_features = self.load_data()
        
        # Train classification models
        classification_results = {}
        for target in ['direction_1d', 'direction_3d', 'direction_7d']:
            classification_results[target] = self.train_classification_models(
                X, targets[target], target, enhanced_features
            )
        
        # Train regression models
        regression_results = {}
        for target in ['price_change_1d', 'price_change_3d', 'price_change_7d']:
            regression_results[target] = self.train_regression_models(
                X, targets[target], target, enhanced_features
            )
        
        # Analyze feature importance
        feature_analysis = self.analyze_feature_importance(
            classification_results, regression_results, baseline_features, text_features
        )
        
        # Compile results
        self.results = {
            'phase': 'enhanced_phase2',
            'description': 'Enhanced models using price data + technical indicators + text-derived sentiment features',
            'features_used': enhanced_features,
            'baseline_features': baseline_features,
            'text_features': text_features,
            'num_features': len(enhanced_features),
            'num_baseline_features': len(baseline_features),
            'num_text_features': len(text_features),
            'dataset_size': len(X),
            'classification_results': classification_results,
            'regression_results': regression_results,
            'feature_analysis': feature_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(self.enhanced_dir, 'enhanced_phase2_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*65)
        print("📊 PHASE 2 ENHANCED RESULTS SUMMARY")
        print("="*65)
        
        print(f"🔧 Features: {self.results['num_features']} total")
        print(f"   📊 Baseline: {self.results['num_baseline_features']} features")
        print(f"   📝 Text-derived: {self.results['num_text_features']} features")
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
        
        fa = self.results['feature_analysis']
        print(f"\n🔍 TEXT FEATURE IMPACT:")
        print(f"   📈 Text contribution: {fa['text_contribution_percent']:.1f}% of total importance")
        print(f"   🏆 Top feature: {fa['top_features'][0][0]} ({fa['top_features'][0][1]:.3f})")
        
        print(f"\n💾 Models saved to: {self.enhanced_dir}")
        print("="*65)

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/home/thej/Desktop/CryptoPulse/data/simplified_ml_dataset.csv"
    MODELS_DIR = "/home/thej/Desktop/CryptoPulse/models"
    
    # Initialize and run
    trainer = EnhancedModelTrainer(DATA_PATH, MODELS_DIR)
    results = trainer.run_enhanced_training()
    
    print("\n🎯 PHASE 2 COMPLETE: Enhanced models trained with text features!")
    print("📈 Next: Integrate CryptoBERT embeddings for ultimate performance.")