#!/usr/bin/env python3
"""
CryptoPulse Regression Analysis
Predicting actual price changes (continuous values) instead of just direction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset for regression"""
    print("📊 Loading dataset for regression analysis...")
    
    df = pd.read_csv('../../data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    with open('../../data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    # Split data
    train_size = int(0.8 * len(df))
    test_data = df.iloc[train_size:].copy()
    
    X_train = df.iloc[:train_size][feature_cols].fillna(0)
    X_test = test_data[feature_cols].fillna(0)
    
    # Use price_change_1d as target (continuous regression target)
    y_train = df.iloc[:train_size]['price_change_1d'].fillna(0)
    y_test = test_data['price_change_1d'].fillna(0)
    
    print(f"✅ Loaded {len(test_data)} test samples")
    print(f"📈 Target: price_change_1d (continuous values)")
    print(f"📊 Target range: {y_test.min():.2f}% to {y_test.max():.2f}%")
    
    return test_data, X_train, X_test, y_train, y_test

def train_regression_models(X_train, X_test, y_train):
    """Train regression models to predict actual price changes"""
    print("🤖 Training regression models...")
    
    models = {}
    predictions = {}
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr.predict(X_test)
    
    # SVR (Support Vector Regression)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr
    predictions['SVR'] = svr.predict(X_test)
    
    # Simple baseline - predict average change
    baseline_pred = np.full(len(X_test), y_train.mean())
    predictions['Baseline (Mean)'] = baseline_pred
    
    print(f"✅ Trained {len(models)} regression models + baseline")
    return models, predictions

def evaluate_models(y_test, predictions):
    """Evaluate regression model performance"""
    print("\n📊 REGRESSION MODEL PERFORMANCE:")
    print("="*70)
    
    for model_name, pred in predictions.items():
        # Calculate regression metrics
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mse)
        
        print(f"\n{model_name}:")
        print(f"   RMSE: {rmse:.3f}%")
        print(f"   MAE:  {mae:.3f}%") 
        print(f"   R²:   {r2:.3f}")
        
        # Direction accuracy (for comparison with classification)
        actual_direction = (y_test > 0).astype(int)
        pred_direction = (pred > 0).astype(int)
        direction_acc = np.mean(actual_direction == pred_direction) * 100
        print(f"   Direction Accuracy: {direction_acc:.1f}%")

def create_regression_plot(test_data, y_test, predictions):
    """Create beautiful regression plot with continuous curves"""
    print("\n🎨 Creating regression visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    dates = test_data['date'].values
    actual_changes = y_test.values
    
    # Plot 1: Actual vs Predicted Price Changes (Continuous Lines)
    ax1.plot(dates, actual_changes, 'k-', linewidth=3, label='Actual Price Changes', alpha=0.8)
    
    # Plot model predictions as smooth continuous lines
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax1.plot(dates, pred, color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2, label=model_name, alpha=0.7)
    
    ax1.set_ylabel('Price Change (%)', fontsize=14, fontweight='bold')
    ax1.set_title('CryptoPulse: Actual vs Predicted Price Changes (Regression)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Rotate dates
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Residuals (Prediction Errors) for best model
    best_model = 'Random Forest'  # Assuming RF is best
    if best_model in predictions:
        residuals = actual_changes - predictions[best_model]
        
        ax2.scatter(dates, residuals, color='red', alpha=0.6, s=50)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax2.set_ylabel('Prediction Error (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax2.set_title(f'{best_model}: Prediction Errors (Residuals)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add error statistics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ax2.text(0.02, 0.98, f'RMSE: {rmse:.3f}%\nMAE: {mae:.3f}%', 
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/regression_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Regression analysis plot saved!")

def create_scatter_comparison(y_test, predictions):
    """Create scatter plots comparing predicted vs actual"""
    print("🎨 Creating scatter comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Predicted vs Actual Price Changes (Scatter Plots)', fontsize=16, fontweight='bold')
    
    model_names = list(predictions.keys())
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        if i >= 4:  # Only plot first 4 models
            break
            
        ax = axes[i//2, i%2]
        
        # Scatter plot
        ax.scatter(y_test, pred, alpha=0.6, s=50)
        
        # Perfect prediction line (y=x)
        min_val = min(y_test.min(), pred.min())
        max_val = max(y_test.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_test, pred)
        
        ax.set_xlabel('Actual Price Change (%)', fontweight='bold')
        ax.set_ylabel('Predicted Price Change (%)', fontweight='bold')
        ax.set_title(f'{model_name} (R² = {r2:.3f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/scatter_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Scatter comparison plot saved!")

def detailed_analysis(test_data, y_test, predictions):
    """Detailed analysis of regression results"""
    print("\n🔍 DETAILED REGRESSION ANALYSIS:")
    print("="*70)
    
    dates = test_data['date'].values
    actual_changes = y_test.values
    
    # Overall statistics
    print(f"📊 ACTUAL PRICE CHANGES STATISTICS:")
    print(f"   Mean: {np.mean(actual_changes):.3f}%")
    print(f"   Std:  {np.std(actual_changes):.3f}%")
    print(f"   Range: {np.min(actual_changes):.3f}% to {np.max(actual_changes):.3f}%")
    print(f"   Positive changes: {np.sum(actual_changes > 0)}/{len(actual_changes)} ({np.mean(actual_changes > 0)*100:.1f}%)")
    
    # Best and worst predictions for Random Forest
    if 'Random Forest' in predictions:
        rf_pred = predictions['Random Forest']
        errors = np.abs(actual_changes - rf_pred)
        
        print(f"\n🎯 RANDOM FOREST DETAILED RESULTS:")
        
        # Best predictions (smallest errors)
        best_indices = np.argsort(errors)[:5]
        print(f"✅ BEST PREDICTIONS (smallest errors):")
        for idx in best_indices:
            date_str = pd.to_datetime(dates[idx]).strftime('%Y-%m-%d')
            actual = actual_changes[idx]
            predicted = rf_pred[idx]
            error = errors[idx]
            print(f"   {date_str}: Actual {actual:+.2f}%, Predicted {predicted:+.2f}%, Error {error:.2f}%")
        
        # Worst predictions (largest errors)
        worst_indices = np.argsort(errors)[-5:]
        print(f"\n❌ WORST PREDICTIONS (largest errors):")
        for idx in worst_indices:
            date_str = pd.to_datetime(dates[idx]).strftime('%Y-%m-%d')
            actual = actual_changes[idx]
            predicted = rf_pred[idx]
            error = errors[idx]
            print(f"   {date_str}: Actual {actual:+.2f}%, Predicted {predicted:+.2f}%, Error {error:.2f}%")

def main():
    """Main function"""
    print("🎯 CryptoPulse Regression Analysis")
    print("="*50)
    print("Converting from binary classification to continuous regression...")
    print("Target: Predicting actual price change percentages")
    
    # Load data
    test_data, X_train, X_test, y_train, y_test = load_data()
    
    # Train regression models
    models, predictions = train_regression_models(X_train, X_test, y_train)
    
    # Evaluate models
    evaluate_models(y_test, predictions)
    
    # Create visualizations
    create_regression_plot(test_data, y_test, predictions)
    create_scatter_comparison(y_test, predictions)
    
    # Detailed analysis
    detailed_analysis(test_data, y_test, predictions)
    
    print("\n" + "="*70)
    print("🎉 REGRESSION ANALYSIS COMPLETE!")
    print("📁 Check plots:")
    print("   • regression_analysis.png - Continuous prediction curves")
    print("   • scatter_comparison.png - Predicted vs actual scatter plots")
    print("💡 Now you have smooth, meaningful curves instead of step functions!")
    print("="*70)

if __name__ == "__main__":
    main()