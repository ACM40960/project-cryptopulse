#!/usr/bin/env python3

"""
Simple, Robust Model Trainer for CryptoPulse
Trains a Logistic Regression model with a limited, robust feature set
to create a more reliable and interpretable baseline.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_model_training.log'),
        logging.StreamHandler()
    ]
)

class SimpleModelTrainer:
    def __init__(self, dataset_path='data/simplified_ml_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Define a limited, robust feature set (NO content_length)
        self.robust_features = [
            'num_comments_sum',
            'volatility_score_reddit',
            'volatility_score_mean',
            'relevance_score_max',
            'echo_score_mean',
            'engagement_sum',
            'echo_score_reddit'
        ]
        
        self.target_cols = ['direction_1d', 'direction_3d', 'direction_7d']
        self.results = {}
        
        logging.info(f"🧠 Simple Model Trainer initialized")
        logging.info(f"   Dataset: {len(self.df)} samples")
        logging.info(f"   Features: {len(self.robust_features)}")
        logging.info(f"   Features used: {self.robust_features}")

    def prepare_data(self, target='direction_1d', test_size=0.2):
        """Prepare data for training with the robust feature set."""
        logging.info(f"📊 Preparing data for {target} prediction...")
        
        # Remove rows with missing target or feature values
        valid_data = self.df.dropna(subset=[target] + self.robust_features).copy()
        valid_data = valid_data.sort_values('date').reset_index(drop=True)
        
        # Prepare features and target
        X = valid_data[self.robust_features].fillna(0)
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
            'feature_names': self.robust_features,
            'scaler': scaler
        }

    def train_logistic_regression(self, data, target):
        """Train a simple Logistic Regression model."""
        logging.info("📈 Training Logistic Regression...")
        
        model = LogisticRegression(
            random_state=42,
            solver='liblinear',  # Good for smaller datasets
            penalty='l1',        # L1 regularization to encourage sparsity
            C=0.1                # Regularization strength
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        train_pred = model.predict(data['X_train'])
        test_pred = model.predict(data['X_test'])
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': model.coef_[0]
        }).sort_values('importance', key=abs, ascending=False)
        
        return {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'feature_importance': feature_importance,
            'model_type': 'LogisticRegression'
        }

    def evaluate_model(self, y_true, y_pred, model_name, dataset_name):
        """Evaluate model performance."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
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
            'down_accuracy': down_accuracy
        }
        
        logging.info(f"   {model_name} ({dataset_name}):")
        logging.info(f"      Accuracy: {accuracy:.3f}")
        logging.info(f"      F1-Score: {f1:.3f}")
        logging.info(f"      Up Days Accuracy: {up_accuracy:.3f}")
        logging.info(f"      Down Days Accuracy: {down_accuracy:.3f}")
        
        return metrics

    def run_training(self, target='direction_1d'):
        """Run the full training and evaluation pipeline."""
        logging.info(f"🎯 Training simple model for {target}")
        
        data = self.prepare_data(target)
        
        result = self.train_logistic_regression(data, target)
        
        logging.info("\n--- Evaluation ---")
        train_metrics = self.evaluate_model(data['y_train'], result['train_pred'], 'LogisticRegression', 'Training')
        test_metrics = self.evaluate_model(data['y_test'], result['test_pred'], 'LogisticRegression', 'Testing')
        
        logging.info("\n--- Feature Importance (Coefficients) ---")
        logging.info(result['feature_importance'])
        
        self.results[target] = {
            'model': result['model'],
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': result['feature_importance'].to_dict()
        }
        
        return self.results[target]

def main():
    """Main function to run the simple model trainer."""
    trainer = SimpleModelTrainer()
    results = trainer.run_training(target='direction_1d')
    
    logging.info("\n🎉 Simple model training complete!")
    logging.info(f"   Test Accuracy: {results['test_metrics']['accuracy']:.3f}")
    logging.info(f"   This provides a more robust baseline for comparison.")

if __name__ == "__main__":
    main()
