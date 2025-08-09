#!/usr/bin/env python3
"""
Visualize Model Predictions - Focus on Working Models
Shows how models predict vs actual results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_style("whitegrid")

def create_simple_model_predictions():
    """Create simple model predictions for visualization"""
    print("🤖 Creating model predictions for visualization...")
    
    # Load data
    df = pd.read_csv('data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    with open('data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['direction_1d']
    
    # Split data (same as original)
    train_size = int(0.8 * len(df))
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    test_data = df.iloc[train_size:]
    
    # Train simple models for visualization
    models = {}
    predictions = {}
    probabilities = {}
    
    # Random Forest (reliable)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf_pred
    probabilities['Random Forest'] = rf_prob[:, 1]
    
    # Baseline model (always predict majority class)
    majority_class = y_train.mode()[0]
    baseline_pred = np.full(len(y_test), majority_class)
    predictions['Baseline (Majority)'] = baseline_pred
    probabilities['Baseline (Majority)'] = np.full(len(y_test), 0.6 if majority_class == 1 else 0.4)
    
    # Random model
    np.random.seed(42)
    random_pred = np.random.choice([0, 1], size=len(y_test))
    predictions['Random'] = random_pred
    probabilities['Random'] = np.random.random(len(y_test))
    
    return test_data, y_test, predictions, probabilities, models

def create_prediction_comparison_plot(test_data, y_test, predictions, probabilities):
    """Create comprehensive prediction comparison plots"""
    print("📊 Creating prediction comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CryptoPulse: Model Prediction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Prediction accuracy comparison
    ax1 = axes[0, 0]
    model_names = list(predictions.keys())
    accuracies = [accuracy_score(y_test, pred) * 100 for pred in predictions.values()]
    
    bars = ax1.bar(model_names, accuracies, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgray'])
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    ax1.set_ylim(0, 100)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction timeline
    ax2 = axes[0, 1]
    dates = test_data['date'].values
    actual_prices = test_data['price_usd'].values
    
    # Plot price line
    ax2.plot(dates, actual_prices, 'k-', linewidth=2, alpha=0.7, label='ETH Price')
    
    # Add prediction markers for Random Forest
    rf_pred = predictions['Random Forest']
    actual = y_test.values
    
    for i, (date, price, pred, act) in enumerate(zip(dates, actual_prices, rf_pred, actual)):
        if pred == act:
            color, marker = ('green', 'o')
        else:
            color, marker = ('red', 'x')
        ax2.scatter(date, price, c=color, marker=marker, s=60, alpha=0.8)
    
    ax2.set_ylabel('ETH Price (USD)', fontweight='bold')
    ax2.set_title('Random Forest Predictions\n(Green=Correct, Red=Wrong)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrices
    ax3 = axes[1, 0]
    
    # Focus on Random Forest
    rf_pred = predictions['Random Forest']
    cm = confusion_matrix(y_test, rf_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    ax3.set_title('Random Forest Confusion Matrix', fontweight='bold')
    
    # Plot 4: Prediction confidence
    ax4 = axes[1, 1]
    
    rf_prob = probabilities['Random Forest']
    rf_pred = predictions['Random Forest']
    actual = y_test.values
    
    # Create confidence vs accuracy plot
    confidence = np.abs(rf_prob - 0.5)  # Distance from 0.5
    correct = (rf_pred == actual)
    
    colors = ['red' if not c else 'green' for c in correct]
    ax4.scatter(confidence, rf_prob, c=colors, alpha=0.7, s=60)
    
    ax4.set_xlabel('Confidence (|prob - 0.5|)', fontweight='bold')
    ax4.set_ylabel('Prediction Probability', fontweight='bold')
    ax4.set_title('Random Forest: Confidence vs Accuracy\n(Green=Correct, Red=Wrong)', fontweight='bold')
    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved model predictions comparison")

def create_feature_importance_plot(models):
    """Create feature importance visualization"""
    print("🔍 Creating feature importance plot...")
    
    # Load feature names
    with open('data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if 'Random Forest' in models:
        rf = models['Random Forest']
        importances = rf.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Create the plot
        feature_names = [feature_cols[i].replace('_', '\n') for i in indices]
        
        bars = ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance', fontweight='bold')
        ax.set_title('Feature Importance Analysis\n(Random Forest Model)', fontsize=14, fontweight='bold')
        
        # Color bars by importance
        max_importance = max(importances)
        for bar, importance in zip(bars, importances[indices]):
            bar.set_color(plt.cm.viridis(importance / max_importance))
        
        ax.grid(True, alpha=0.3)
        
        # Add importance values
        for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
            ax.text(importance + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved feature importance plot")

def create_prediction_details_plot(test_data, y_test, predictions, probabilities):
    """Create detailed prediction analysis"""
    print("📋 Creating prediction details...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Prediction Analysis', fontsize=16, fontweight='bold')
    
    rf_pred = predictions['Random Forest']
    rf_prob = probabilities['Random Forest']
    actual = y_test.values
    price_changes = test_data['price_change_1d'].values
    
    # Plot 1: Correct vs Incorrect predictions by price change magnitude
    ax1 = axes[0, 0]
    
    correct_mask = (rf_pred == actual)
    correct_changes = price_changes[correct_mask]
    incorrect_changes = price_changes[~correct_mask]
    
    ax1.hist(np.abs(correct_changes), bins=10, alpha=0.7, label='Correct Predictions', color='green', density=True)
    ax1.hist(np.abs(incorrect_changes), bins=10, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
    
    ax1.set_xlabel('Absolute Price Change (%)', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Prediction Accuracy vs Price Change Magnitude', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: High confidence vs Low confidence predictions
    ax2 = axes[0, 1]
    
    confidence = np.abs(rf_prob - 0.5)
    high_conf_mask = confidence > 0.2
    low_conf_mask = confidence <= 0.2
    
    high_conf_acc = accuracy_score(actual[high_conf_mask], rf_pred[high_conf_mask]) * 100 if np.sum(high_conf_mask) > 0 else 0
    low_conf_acc = accuracy_score(actual[low_conf_mask], rf_pred[low_conf_mask]) * 100 if np.sum(low_conf_mask) > 0 else 0
    
    bars = ax2.bar(['High Confidence\n(>0.2)', 'Low Confidence\n(≤0.2)'], 
                   [high_conf_acc, low_conf_acc], 
                   color=['darkgreen', 'darkred'], alpha=0.7)
    
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Accuracy by Confidence Level', fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
    
    # Add sample counts and accuracy values
    for bar, acc, mask in zip(bars, [high_conf_acc, low_conf_acc], [high_conf_mask, low_conf_mask]):
        count = np.sum(mask)
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction probability distribution
    ax3 = axes[1, 0]
    
    correct_probs = rf_prob[correct_mask]
    incorrect_probs = rf_prob[~correct_mask] 
    
    ax3.hist(correct_probs, bins=15, alpha=0.7, label='Correct Predictions', color='green', density=True)
    ax3.hist(incorrect_probs, bins=15, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
    
    ax3.set_xlabel('Prediction Probability', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Probability Distribution by Accuracy', fontweight='bold')
    ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model performance summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    rf_accuracy = accuracy_score(actual, rf_pred)
    rf_precision = precision_score(actual, rf_pred)
    rf_recall = recall_score(actual, rf_pred)
    rf_f1 = f1_score(actual, rf_pred)
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Accuracy', f'{rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)'],
        ['Precision', f'{rf_precision:.3f}'],
        ['Recall', f'{rf_recall:.3f}'],
        ['F1-Score', f'{rf_f1:.3f}'],
        ['', ''],
        ['Test Samples', f'{len(actual)}'],
        ['Correct Predictions', f'{np.sum(rf_pred == actual)}'],
        ['Wrong Predictions', f'{np.sum(rf_pred != actual)}'],
        ['', ''],
        ['High Confidence (>0.2)', f'{np.sum(high_conf_mask)} samples'],
        ['Low Confidence (≤0.2)', f'{np.sum(low_conf_mask)} samples']
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='left', loc='center', colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(2):
            cell = table[i, j]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2')
    
    ax4.set_title('Random Forest Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/prediction_details.png', dpi=300, bbox_inches='tight')
    print("✅ Saved prediction details")

def main():
    """Main function"""
    print("🎯 CryptoPulse Model Prediction Visualizer")
    print("="*50)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Create model predictions
    test_data, y_test, predictions, probabilities, models = create_simple_model_predictions()
    
    # Create visualizations
    create_prediction_comparison_plot(test_data, y_test, predictions, probabilities)
    create_feature_importance_plot(models)
    create_prediction_details_plot(test_data, y_test, predictions, probabilities)
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for name, pred in predictions.items():
        acc = accuracy_score(y_test, pred) * 100
        print(f"  {name}: {acc:.1f}% accuracy")
    
    print(f"\n🎯 Test set size: {len(y_test)} samples")
    print(f"📈 Actual up days: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    print(f"📉 Actual down days: {len(y_test) - np.sum(y_test)} ({(1-np.mean(y_test))*100:.1f}%)")
    
    print("\n" + "="*50)
    print("🎉 Model prediction visualizations completed!")
    print("📁 New plots saved:")
    print("   • model_predictions_comparison.png")
    print("   • feature_importance.png")
    print("   • prediction_details.png")
    print("="*50)

if __name__ == "__main__":
    main()