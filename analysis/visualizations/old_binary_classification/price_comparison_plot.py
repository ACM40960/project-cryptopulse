#!/usr/bin/env python3
"""
Price Comparison Plot - Show actual ETH prices with prediction overlays
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    print("📊 Loading dataset...")
    
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
    y_train = df.iloc[:train_size]['direction_1d']
    y_test = test_data['direction_1d']
    
    print(f"✅ Loaded {len(test_data)} test samples")
    return test_data, X_train, X_test, y_train, y_test

def get_predictions(X_train, X_test, y_train):
    """Get predictions from different models"""
    print("🤖 Getting model predictions...")
    
    predictions = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions['Random Forest'] = rf.predict(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    predictions['Logistic Regression'] = lr.predict(X_test)
    
    # SVM
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    predictions['SVM'] = svm.predict(X_test)
    
    print(f"✅ Generated predictions from {len(predictions)} models")
    return predictions

def create_price_plot(test_data, y_test, predictions):
    """Create plot showing actual ETH prices with prediction indicators"""
    print("🎨 Creating price comparison plot...")
    
    dates = test_data['date'].values
    prices = test_data['price_usd'].values
    actual_directions = y_test.values
    
    # Create subplots for direction indicators
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: ETH Price
    ax1.plot(dates, prices, 'k-', linewidth=3, label='ETH Price (USD)', alpha=0.8)
    ax1.set_ylabel('ETH Price (USD)', fontsize=14, fontweight='bold')
    ax1.set_title('CryptoPulse: ETH Price vs Model Predictions', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Bottom plot: Direction predictions as step plots
    ax2.step(dates, actual_directions, 'k-', linewidth=3, label='Actual Direction', 
             alpha=0.8, where='post')
    
    colors = ['red', 'blue', 'green']
    linestyles = ['--', ':', '-.']
    
    i = 0
    for model_name, pred in predictions.items():
        ax2.step(dates, pred, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, label=model_name, alpha=0.7, where='post')
        i += 1
    
    ax2.set_ylabel('Direction\n(0=Down, 1=Up)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Format dates
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/price_comparison_plot.png', dpi=300, bbox_inches='tight')
    
    print("✅ Price comparison plot saved!")
    
    # Print accuracies
    print("\n📊 Model Accuracies:")
    for model_name, pred in predictions.items():
        acc = np.mean(pred == actual_directions) * 100
        print(f"   {model_name}: {acc:.1f}%")
    
    # Print price statistics
    print(f"\n📈 Price Range: ${prices.min():.0f} - ${prices.max():.0f}")
    print(f"📊 Price Volatility: {((prices.max() - prices.min()) / prices.mean() * 100):.1f}%")

def main():
    """Main function"""
    print("🎯 Price Comparison Plot Creator")
    print("="*40)
    
    # Load data
    test_data, X_train, X_test, y_train, y_test = load_data()
    
    # Get predictions
    predictions = get_predictions(X_train, X_test, y_train)
    
    # Create price plot
    create_price_plot(test_data, y_test, predictions)
    
    print("\n🎉 Done! Check plots/price_comparison_plot.png")
    print("📊 This shows:")
    print("   • Top: Actual ETH price movement") 
    print("   • Bottom: Direction predictions (histogram style)")

if __name__ == "__main__":
    main()