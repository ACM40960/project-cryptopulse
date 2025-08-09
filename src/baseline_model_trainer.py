#!/usr/bin/env python3
# src/baseline_model_trainer.py

"""
Baseline Model Trainer for CryptoPulse
Creates baseline models using only basic price and technical indicators
without sentiment metrics to establish performance comparison baseline.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/baseline_model_training.log'),
        logging.StreamHandler()
    ]
)

class BaselineModelTrainer:
    def __init__(self, dataset_path='data/simplified_ml_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Define ONLY basic price and technical features (NO SENTIMENT)
        # IMPORTANT: Remove same-day price changes to avoid data leakage
        self.baseline_features = [
            'price_usd',
            'price_ma_7',
            'price_volatility'
        ]
        
        # Filter to only rows where baseline features are available
        self.baseline_df = self.df.dropna(subset=self.baseline_features).copy()
        
        self.target_cols = ['direction_1d', 'direction_3d', 'direction_7d']
        self.results = {}
        
        logging.info(f"📊 Baseline Model Trainer initialized")
        logging.info(f"   Dataset: {len(self.baseline_df)} samples (after filtering for baseline features)")
        logging.info(f"   Baseline Features: {len(self.baseline_features)}")
        logging.info(f"   Features: {self.baseline_features}")
        logging.info(f"   Date Range: {self.baseline_df['date'].min().date()} to {self.baseline_df['date'].max().date()}")
    
    def create_lag_features(self, df, feature_cols, lags=[1, 2, 3, 7]):
        """Create lagged features for better baseline performance."""
        logging.info(f"   Creating lag features (lags: {lags})...")
        
        # Sort by date to ensure proper lag calculation
        df_sorted = df.sort_values('date').copy()
        
        for col in feature_cols:
            if col in df_sorted.columns:
                for lag in lags:
                    df_sorted[f"{col}_lag_{lag}"] = df_sorted[col].shift(lag)
        
        # Drop rows with NaN from lagging
        max_lag = max(lags)
        df_lagged = df_sorted.iloc[max_lag:].copy()
        
        # Get new feature columns (original + lagged)
        lag_feature_cols = [col for col in df_lagged.columns 
                           if col in feature_cols or any(f"{fc}_lag_" in col for fc in feature_cols)]
        
        logging.info(f"   Created {len(lag_feature_cols)} features (including lags)")
        logging.info(f"   Samples after lagging: {len(df_lagged)}")
        
        return df_lagged, lag_feature_cols
    
    def prepare_baseline_data(self, target='direction_1d', test_size=0.2, use_lags=True):
        """Prepare baseline data with only price/technical features."""
        logging.info(f"📈 Preparing BASELINE data for {target} prediction...")
        
        # Start with baseline features
        feature_cols = self.baseline_features.copy()
        working_df = self.baseline_df.copy()
        
        # Add lag features if requested
        if use_lags:
            working_df, feature_cols = self.create_lag_features(
                working_df, self.baseline_features, lags=[1, 2, 3, 7]
            )
        
        # Remove rows with missing target values
        valid_data = working_df.dropna(subset=[target]).copy()
        valid_data = valid_data.sort_values('date').reset_index(drop=True)
        
        logging.info(f"   Valid samples: {len(valid_data)}")
        
        # Prepare features and target
        X = valid_data[feature_cols].fillna(0)
        y = valid_data[target].astype(int)
        dates = valid_data['date']
        
        # Time-series split
        split_idx = int(len(valid_data) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        train_dates = dates.iloc[:split_idx]
        test_dates = dates.iloc[split_idx:]
        
        logging.info(f"   Training: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
        logging.info(f"   Testing: {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_test': y_test,
            'X_train_raw': X_train, 'X_test_raw': X_test,
            'train_dates': train_dates, 'test_dates': test_dates,
            'scaler': scaler, 'feature_names': feature_cols,
            'feature_count': len(feature_cols)
        }
    
    def train_baseline_random_forest(self, data, target):
        """Train baseline Random Forest."""
        logging.info("🌳 Training Baseline Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(data['X_train_raw'], data['y_train'])
        
        # Predictions
        train_pred = model.predict(data['X_train_raw'])
        test_pred = model.predict(data['X_test_raw'])
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'feature_importance': feature_importance,
            'model_type': 'Baseline_RandomForest'
        }
    
    def train_baseline_lightgbm(self, data, target):
        """Train baseline LightGBM."""
        logging.info("💡 Training Baseline LightGBM...")
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        X_train = data.get('X_train_scaled', data.get('X_train'))
        X_test = data.get('X_test_scaled', data.get('X_test'))
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=data['y_train'])
        valid_data = lgb.Dataset(X_test, label=data['y_test'], reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # Predictions
        train_pred = (model.predict(X_train) > 0.5).astype(int)
        test_pred = (model.predict(X_test) > 0.5).astype(int)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'feature_importance': feature_importance,
            'model_type': 'Baseline_LightGBM'
        }
    
    def train_baseline_xgboost(self, data, target):
        """Train baseline XGBoost."""
        logging.info("🚀 Training Baseline XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        X_train = data.get('X_train_scaled', data.get('X_train'))
        X_test = data.get('X_test_scaled', data.get('X_test'))
        
        model.fit(
            X_train, data['y_train'],
            eval_set=[(X_test, data['y_test'])],
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'feature_importance': feature_importance,
            'model_type': 'Baseline_XGBoost'
        }
    
    def evaluate_model(self, y_true, y_pred, model_name, dataset_name):
        """Evaluate model performance."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Up/Down accuracy breakdown
        up_mask = y_true == 1
        down_mask = y_true == 0
        
        up_accuracy = (y_pred[up_mask] == y_true[up_mask]).mean() if up_mask.sum() > 0 else 0
        down_accuracy = (y_pred[down_mask] == y_true[down_mask]).mean() if down_mask.sum() > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_samples': len(y_true),
            'up_samples': up_mask.sum(),
            'down_samples': down_mask.sum()
        }
        
        logging.info(f"   {model_name} ({dataset_name}):")
        logging.info(f"      Accuracy: {accuracy:.3f}")
        logging.info(f"      Precision: {precision:.3f}")
        logging.info(f"      Recall: {recall:.3f}")
        logging.info(f"      F1-Score: {f1:.3f}")
        logging.info(f"      Up Days Accuracy: {up_accuracy:.3f}")
        logging.info(f"      Down Days Accuracy: {down_accuracy:.3f}")
        
        return metrics
    
    def cross_validate_baseline(self, data, model_func, target, cv_folds=3):
        """Perform time-series cross-validation for baseline."""
        logging.info(f"   Performing {cv_folds}-fold time-series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        X_train_scaled = data['X_train']
        X_train_raw = data['X_train_raw']
        y_train = data['y_train']
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
            X_fold_train_scaled = X_train_scaled[train_idx]
            X_fold_val_scaled = X_train_scaled[val_idx]
            X_fold_train_raw = X_train_raw.iloc[train_idx]
            X_fold_val_raw = X_train_raw.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_data = {
                'X_train': X_fold_train_scaled,
                'X_test': X_fold_val_scaled,
                'y_train': y_fold_train,
                'y_test': y_fold_val,
                'X_train_raw': X_fold_train_raw,
                'X_test_raw': X_fold_val_raw,
                'feature_names': data['feature_names']
            }
            
            fold_result = model_func(fold_data, target)
            fold_accuracy = accuracy_score(y_fold_val, fold_result['test_pred'])
            cv_scores.append(fold_accuracy)
            
            logging.info(f"      Fold {fold+1}: {fold_accuracy:.3f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logging.info(f"   CV Mean: {cv_mean:.3f} ± {cv_std:.3f}")
        
        return cv_scores, cv_mean, cv_std
    
    def train_all_baseline_models(self, target='direction_1d'):
        """Train all baseline models."""
        logging.info(f"🎯 Training BASELINE models for {target}")
        logging.info("=" * 60)
        logging.info("📈 Using ONLY price and technical indicators (NO SENTIMENT)")
        logging.info("=" * 60)
        
        # Prepare baseline data
        data = self.prepare_baseline_data(target)
        
        # Define baseline models
        models = {
            'Baseline_RandomForest': self.train_baseline_random_forest,
            'Baseline_LightGBM': self.train_baseline_lightgbm,
            'Baseline_XGBoost': self.train_baseline_xgboost
        }
        
        target_results = {}
        
        for model_name, model_func in models.items():
            logging.info(f"\n{model_name}:")
            logging.info("-" * 40)
            
            # Train model
            result = model_func(data, target)
            
            # Evaluate on training set
            train_metrics = self.evaluate_model(
                data['y_train'], result['train_pred'], 
                model_name, 'Training'
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(
                data['y_test'], result['test_pred'], 
                model_name, 'Testing'
            )
            
            # Cross-validation
            cv_scores, cv_mean, cv_std = self.cross_validate_baseline(data, model_func, target)
            
            # Store results
            target_results[model_name] = {
                'model': result['model'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'feature_importance': result['feature_importance'],
                'feature_count': data['feature_count'],
                'predictions': {
                    'train_pred': result['train_pred'],
                    'test_pred': result['test_pred'],
                    'y_train': data['y_train'],
                    'y_test': data['y_test']
                }
            }
            
            # Show top features
            logging.info(f"   Top Features:")
            for idx, row in result['feature_importance'].head(8).iterrows():
                logging.info(f"      {row['feature']}: {row['importance']:.4f}")
        
        self.results[target] = target_results
        return target_results
    
    def compare_baseline_models(self, target='direction_1d'):
        """Compare baseline model performance."""
        if target not in self.results:
            logging.error(f"No results found for {target}. Train models first.")
            return
        
        logging.info(f"\n📊 BASELINE MODEL COMPARISON FOR {target}")
        logging.info("=" * 60)
        logging.info("📈 Models trained with ONLY price/technical features")
        logging.info("=" * 60)
        
        comparison = []
        
        for model_name, results in self.results[target].items():
            comparison.append({
                'Model': model_name,
                'Features': results['feature_count'],
                'Test_Accuracy': results['test_metrics']['accuracy'],
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'Up_Days_Acc': results['test_metrics']['up_accuracy'],
                'Down_Days_Acc': results['test_metrics']['down_accuracy'],
                'F1_Score': results['test_metrics']['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('Test_Accuracy', ascending=False)
        
        logging.info("\nBaseline Model Performance Summary:")
        logging.info(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Identify best baseline model
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Test_Accuracy']
        
        logging.info(f"\n🏆 BEST BASELINE MODEL: {best_model}")
        logging.info(f"   Test Accuracy: {best_accuracy:.3f}")
        logging.info(f"   Features Used: {comparison_df.iloc[0]['Features']}")
        logging.info(f"   CV Score: {comparison_df.iloc[0]['CV_Mean']:.3f} ± {comparison_df.iloc[0]['CV_Std']:.3f}")
        
        return comparison_df, best_model, best_accuracy
    
    def save_baseline_models(self, target='direction_1d', save_dir='models/baseline/'):
        """Save baseline models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if target not in self.results:
            logging.error(f"No results found for {target}")
            return
        
        logging.info(f"💾 Saving baseline models for {target}...")
        
        saved_files = []
        
        for model_name, results in self.results[target].items():
            # Save model
            model_file = f"{save_dir}{model_name}_{target}.joblib"
            joblib.dump(results['model'], model_file)
            saved_files.append(model_file)
            
            # Save feature importance
            importance_file = f"{save_dir}{model_name}_{target}_features.csv"
            results['feature_importance'].to_csv(importance_file, index=False)
            saved_files.append(importance_file)
            
            logging.info(f"   {model_name}: {model_file}")
        
        # Save results summary
        summary_file = f"{save_dir}baseline_results_summary_{target}.json"
        summary = {}
        
        for model_name, results in self.results[target].items():
            summary[model_name] = {
                'test_accuracy': results['test_metrics']['accuracy'],
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'feature_count': results['feature_count'],
                'up_accuracy': results['test_metrics']['up_accuracy'],
                'down_accuracy': results['test_metrics']['down_accuracy'],
                'f1_score': results['test_metrics']['f1_score']
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files.append(summary_file)
        
        logging.info(f"✅ Saved {len(saved_files)} baseline files to {save_dir}")
        return saved_files

def main():
    """Main baseline training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline models for crypto prediction')
    parser.add_argument('--dataset', default='data/simplified_ml_dataset.csv', help='Dataset path')
    parser.add_argument('--target', default='direction_1d', choices=['direction_1d', 'direction_3d', 'direction_7d'], help='Target to predict')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    
    args = parser.parse_args()
    
    # Initialize baseline trainer
    trainer = BaselineModelTrainer(args.dataset)
    
    # Train all baseline models
    results = trainer.train_all_baseline_models(args.target)
    
    # Compare baseline models
    comparison, best_model, best_accuracy = trainer.compare_baseline_models(args.target)
    
    # Save models if requested
    if args.save_models:
        trainer.save_baseline_models(args.target)
    
    logging.info(f"\n🎉 Baseline training complete!")
    logging.info(f"   Best baseline model: {best_model}")
    logging.info(f"   Best baseline accuracy: {best_accuracy:.3f}")
    logging.info(f"   📈 This is our benchmark to beat with sentiment features!")

if __name__ == "__main__":
    main()