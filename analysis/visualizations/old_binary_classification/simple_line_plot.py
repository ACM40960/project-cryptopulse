#!/usr/bin/env python3
"""
Simple Line Plot - Actual vs Model Predictions
One clean plot with actual line and model prediction lines
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

def create_simple_plot(test_data, y_test, predictions):
    """Create simple histogram-style step plot"""
    print("🎨 Creating histogram-style step plot...")
    
    plt.figure(figsize=(16, 8))
    
    dates = test_data['date'].values
    actual = y_test.values
    
    # Plot actual data as solid black step plot (histogram style)
    plt.step(dates, actual, 'k-', linewidth=3, label='Actual', alpha=0.8, where='post')
    
    # Plot model predictions as step plots with different colors
    colors = ['red', 'blue', 'green']
    linestyles = ['--', ':', '-.']
    
    i = 0
    for model_name, pred in predictions.items():
        plt.step(dates, pred, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, label=model_name, alpha=0.7, where='post')
        i += 1
    
    # Simple formatting
    plt.ylabel('Direction (0=Down, 1=Up)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.title('CryptoPulse: Actual vs Model Predictions (Histogram Style)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    # Rotate dates
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/simple_step_plot.png', dpi=300, bbox_inches='tight')
    
    print("✅ Histogram-style step plot saved!")
    
    # Print accuracies
    print("\n📊 Model Accuracies:")
    for model_name, pred in predictions.items():
        acc = np.mean(pred == actual) * 100
        print(f"   {model_name}: {acc:.1f}%")

def main():
    """Main function"""
    print("🎯 Simple Line Plot Creator")
    print("="*40)
    
    # Load data
    test_data, X_train, X_test, y_train, y_test = load_data()
    
    # Get predictions
    predictions = get_predictions(X_train, X_test, y_train)
    
    # Create simple plot
    create_simple_plot(test_data, y_test, predictions)
    
    print("\n🎉 Done! Check plots/simple_line_plot.png")

if __name__ == "__main__":
    main()