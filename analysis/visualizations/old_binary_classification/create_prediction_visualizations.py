#!/usr/bin/env python3
"""
CryptoPulse Model Prediction Visualizations
Creates comprehensive plots showing how different models predict ETH price movements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_style("darkgrid")
sns.set_palette("husl")

class CryptoPulsePlotter:
    def __init__(self):
        self.data_path = 'data/simplified_ml_dataset.csv'
        self.models_dir = 'models'
        self.output_dir = 'plots'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Get feature columns
        with open('data/simplified_ml_dataset_info.json', 'r') as f:
            info = json.load(f)
            self.feature_cols = info['feature_columns']
            self.target_cols = info['target_columns']
        
        print(f"✅ Loaded {len(self.df)} samples with {len(self.feature_cols)} features")
        
        # Split data (same as training)
        train_size = int(0.8 * len(self.df))
        self.train_data = self.df.iloc[:train_size]
        self.test_data = self.df.iloc[train_size:]
        
        print(f"📈 Train: {len(self.train_data)} samples, Test: {len(self.test_data)} samples")
        
        # Prepare features
        self.X_train = self.train_data[self.feature_cols].fillna(0)
        self.X_test = self.test_data[self.feature_cols].fillna(0)
        self.y_train = self.train_data['direction_1d']
        self.y_test = self.test_data['direction_1d']
        
        # Store models and predictions
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        
    def load_models(self):
        """Load all available trained models"""
        print("\n🤖 Loading trained models...")
        
        # Sentiment-enhanced models
        model_files = [
            ('LightGBM-Enhanced', 'LightGBM_direction_1d.joblib'),
            ('RandomForest-Enhanced', 'RandomForest_direction_1d.joblib'),
            ('XGBoost-Enhanced', 'XGBoost_direction_1d.joblib')
        ]
        
        # Baseline models
        baseline_files = [
            ('LightGBM-Baseline', 'baseline/Baseline_LightGBM_direction_1d.joblib'),
            ('RandomForest-Baseline', 'baseline/Baseline_RandomForest_direction_1d.joblib'),
            ('XGBoost-Baseline', 'baseline/Baseline_XGBoost_direction_1d.joblib')
        ]
        
        all_models = model_files + baseline_files
        
        for name, file_path in all_models:
            full_path = os.path.join(self.models_dir, file_path)
            if os.path.exists(full_path):
                try:
                    model = joblib.load(full_path)
                    self.models[name] = model
                    
                    # Generate predictions
                    pred = model.predict(self.X_test)
                    self.predictions[name] = pred
                    
                    # Generate probabilities if available
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(self.X_test)
                        self.probabilities[name] = prob[:, 1]  # Probability of class 1 (up)
                    
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"⚠️ Model file not found: {full_path}")
        
        print(f"📊 Successfully loaded {len(self.models)} models")
    
    def create_prediction_timeline(self):
        """Create timeline showing actual vs predicted for all models"""
        print("\n📈 Creating prediction timeline...")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('CryptoPulse: Model Predictions vs Reality Over Time', fontsize=16, fontweight='bold')
        
        test_dates = self.test_data['date'].values
        actual_prices = self.test_data['price_usd'].values
        actual_directions = self.y_test.values
        
        # Plot 1: Price timeline with predictions
        ax1 = axes[0]
        ax1.plot(test_dates, actual_prices, 'k-', linewidth=2, label='Actual ETH Price', alpha=0.8)
        
        # Add prediction markers for best model
        if 'LightGBM-Enhanced' in self.predictions:
            pred = self.predictions['LightGBM-Enhanced']
            for i, (date, price, actual, predicted) in enumerate(zip(test_dates, actual_prices, actual_directions, pred)):
                color = 'green' if predicted == actual else 'red'
                marker = '^' if predicted == 1 else 'v'
                ax1.scatter(date, price, c=color, marker=marker, s=60, alpha=0.7)
        
        ax1.set_ylabel('ETH Price (USD)', fontsize=12)
        ax1.set_title('ETH Price with LightGBM-Enhanced Predictions (Green=Correct, Red=Wrong)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Direction predictions heatmap
        ax2 = axes[1]
        
        # Create prediction matrix
        model_names = list(self.predictions.keys())
        pred_matrix = np.array([self.predictions[name] for name in model_names])
        
        # Add actual direction as first row
        combined_matrix = np.vstack([actual_directions, pred_matrix])
        combined_names = ['Actual'] + model_names
        
        # Create heatmap
        im = ax2.imshow(combined_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax2.set_yticks(range(len(combined_names)))
        ax2.set_yticklabels(combined_names)
        ax2.set_xlabel('Test Sample Index')
        ax2.set_title('Prediction Heatmap: Green=Up, Red=Down', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.05)
        cbar.set_label('Direction (0=Down, 1=Up)')
        
        # Plot 3: Accuracy comparison
        ax3 = axes[2]
        
        model_accuracies = []
        model_names_clean = []
        
        for name, pred in self.predictions.items():
            accuracy = np.mean(pred == actual_directions) * 100
            model_accuracies.append(accuracy)
            model_names_clean.append(name.replace('-', '\n'))
        
        bars = ax3.bar(model_names_clean, model_accuracies, alpha=0.7)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_title('Model Accuracy Comparison on Test Set', fontsize=14)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Chance')
        ax3.set_ylim(0, 100)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, model_accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/prediction_timeline.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved prediction timeline to {self.output_dir}/prediction_timeline.png")
        
    def create_confidence_analysis(self):
        """Analyze prediction confidence and uncertainty"""
        print("\n🎯 Creating confidence analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Confidence and Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Prediction probabilities distribution
        ax1 = axes[0, 0]
        
        for name, prob in self.probabilities.items():
            ax1.hist(prob, bins=20, alpha=0.6, label=name, density=True)
        
        ax1.set_xlabel('Prediction Probability (Up Direction)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Prediction Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence vs Accuracy
        ax2 = axes[0, 1]
        
        if 'LightGBM-Enhanced' in self.probabilities:
            prob = self.probabilities['LightGBM-Enhanced']
            pred = self.predictions['LightGBM-Enhanced']
            actual = self.y_test.values
            
            # Calculate confidence (distance from 0.5)
            confidence = np.abs(prob - 0.5)
            correct = (pred == actual)
            
            # Scatter plot
            colors = ['red' if not c else 'green' for c in correct]
            ax2.scatter(confidence, prob, c=colors, alpha=0.7)
            ax2.set_xlabel('Confidence (|prob - 0.5|)')
            ax2.set_ylabel('Prediction Probability')
            ax2.set_title('LightGBM: Confidence vs Predictions\n(Green=Correct, Red=Wrong)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance (if available)
        ax3 = axes[1, 0]
        
        if 'LightGBM-Enhanced' in self.models:
            model = self.models['LightGBM-Enhanced']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = [name.replace('_', '\n') for name in self.feature_cols]
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                ax3.barh(range(len(indices)), importances[indices])
                ax3.set_yticks(range(len(indices)))
                ax3.set_yticklabels([feature_names[i] for i in indices])
                ax3.set_xlabel('Feature Importance')
                ax3.set_title('Top 10 Most Important Features\n(LightGBM-Enhanced)')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis by time
        ax4 = axes[1, 1]
        
        if 'LightGBM-Enhanced' in self.predictions:
            pred = self.predictions['LightGBM-Enhanced']
            actual = self.y_test.values
            test_dates = self.test_data['date'].values
            
            errors = pred != actual
            error_dates = test_dates[errors]
            correct_dates = test_dates[~errors]
            
            # Plot error distribution over time
            ax4.scatter(correct_dates, [1]*len(correct_dates), c='green', alpha=0.6, label='Correct', s=40)
            ax4.scatter(error_dates, [1]*len(error_dates), c='red', alpha=0.8, label='Wrong', s=40)
            ax4.set_ylabel('Predictions')
            ax4.set_xlabel('Date')
            ax4.set_title('Prediction Accuracy Over Time\n(LightGBM-Enhanced)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved confidence analysis to {self.output_dir}/confidence_analysis.png")
    
    def create_model_comparison(self):
        """Create detailed model comparison visualizations"""
        print("\n📊 Creating model comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        actual = self.y_test.values
        
        # Plot 1: Accuracy bar chart
        ax1 = axes[0, 0]
        
        model_names = list(self.predictions.keys())
        accuracies = [np.mean(self.predictions[name] == actual) * 100 for name in model_names]
        
        # Separate enhanced vs baseline
        enhanced_names = [name for name in model_names if 'Enhanced' in name]
        baseline_names = [name for name in model_names if 'Baseline' in name]
        
        enhanced_acc = [np.mean(self.predictions[name] == actual) * 100 for name in enhanced_names]
        baseline_acc = [np.mean(self.predictions[name] == actual) * 100 for name in baseline_names]
        
        x_enh = np.arange(len(enhanced_names))
        x_base = np.arange(len(baseline_names)) + len(enhanced_names) + 0.5
        
        bars1 = ax1.bar(x_enh, enhanced_acc, alpha=0.8, color='skyblue', label='Sentiment-Enhanced')
        bars2 = ax1.bar(x_base, baseline_acc, alpha=0.8, color='lightcoral', label='Baseline')
        
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Enhanced vs Baseline Model Performance')
        ax1.set_xticks(list(x_enh) + list(x_base))
        ax1.set_xticklabels([name.replace('-Enhanced', '').replace('-Baseline', '') for name in enhanced_names + baseline_names], rotation=45)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, acc in zip(bars1, enhanced_acc):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        for bar, acc in zip(bars2, baseline_acc):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Prediction agreement matrix
        ax2 = axes[0, 1]
        
        # Create agreement matrix
        n_models = len(model_names)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    agreement = np.mean(self.predictions[name1] == self.predictions[name2])
                    agreement_matrix[i][j] = agreement
                else:
                    agreement_matrix[i][j] = 1.0
        
        im = ax2.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(n_models))
        ax2.set_yticks(range(n_models))
        ax2.set_xticklabels([name.replace('-', '\n') for name in model_names], rotation=45)
        ax2.set_yticklabels([name.replace('-', '\n') for name in model_names])
        ax2.set_title('Model Prediction Agreement Matrix')
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = ax2.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                               ha="center", va="center", color="red" if agreement_matrix[i, j] < 0.5 else "black")
        
        plt.colorbar(im, ax=ax2, fraction=0.05)
        
        # Plot 3: Ensemble prediction
        ax3 = axes[1, 0]
        
        # Create simple majority vote ensemble
        all_predictions = np.array([self.predictions[name] for name in model_names])
        ensemble_pred = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
        ensemble_acc = np.mean(ensemble_pred == actual) * 100
        
        # Compare ensemble with individual models
        all_acc = accuracies + [ensemble_acc]
        all_names_ens = [name.replace('-', '\n') for name in model_names] + ['Ensemble\n(Majority)']
        
        bars = ax3.bar(range(len(all_acc)), all_acc, alpha=0.7)
        bars[-1].set_color('gold')  # Highlight ensemble
        
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Individual Models vs Ensemble')
        ax3.set_xticks(range(len(all_names_ens)))
        ax3.set_xticklabels(all_names_ens, rotation=45)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # Add values
        for bar, acc in zip(bars, all_acc):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Price movement correlation
        ax4 = axes[1, 1]
        
        if 'LightGBM-Enhanced' in self.predictions:
            pred = self.predictions['LightGBM-Enhanced']
            price_changes = self.test_data['price_change_1d'].values
            
            # Group by prediction and actual
            pred_up_actual_up = price_changes[(pred == 1) & (actual == 1)]
            pred_up_actual_down = price_changes[(pred == 1) & (actual == 0)]
            pred_down_actual_up = price_changes[(pred == 0) & (actual == 1)]
            pred_down_actual_down = price_changes[(pred == 0) & (actual == 0)]
            
            # Box plot
            data_to_plot = [pred_up_actual_up, pred_up_actual_down, pred_down_actual_up, pred_down_actual_down]
            labels = ['Pred↑\nActual↑', 'Pred↑\nActual↓', 'Pred↓\nActual↑', 'Pred↓\nActual↓']
            
            bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightgreen', 'lightcoral', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax4.set_ylabel('Price Change (%)')
            ax4.set_title('Price Change Distribution by Prediction\n(LightGBM-Enhanced)')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved model comparison to {self.output_dir}/model_comparison.png")
    
    def create_performance_summary(self):
        """Create a summary table of all model performances"""
        print("\n📋 Creating performance summary...")
        
        # Calculate metrics for all models
        results = []
        actual = self.y_test.values
        
        for name, pred in self.predictions.items():
            accuracy = np.mean(pred == actual) * 100
            
            # Calculate precision, recall, F1
            tp = np.sum((pred == 1) & (actual == 1))
            fp = np.sum((pred == 1) & (actual == 0))
            fn = np.sum((pred == 0) & (actual == 1))
            tn = np.sum((pred == 0) & (actual == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy:.1f}%",
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}",
                'Correct': tp + tn,
                'Wrong': fp + fn,
                'Total': len(pred)
            })
        
        # Create summary table
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*80)
        print("CRYPTOPULSE MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Save to file
        results_df.to_csv(f'{self.output_dir}/model_performance_summary.csv', index=False)
        print(f"✅ Saved performance summary to {self.output_dir}/model_performance_summary.csv")
        
        return results_df
    
    def run_all_visualizations(self):
        """Run all visualization functions"""
        print("🎨 Starting CryptoPulse Visualization Suite...")
        print("="*60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models found! Please train models first.")
            return
        
        # Create all visualizations
        self.create_prediction_timeline()
        self.create_confidence_analysis()
        self.create_model_comparison()
        self.create_performance_summary()
        
        print("\n" + "="*60)
        print("🎉 All visualizations completed!")
        print(f"📁 Check the '{self.output_dir}' directory for all plots")
        print("="*60)

if __name__ == "__main__":
    plotter = CryptoPulsePlotter()
    plotter.run_all_visualizations()