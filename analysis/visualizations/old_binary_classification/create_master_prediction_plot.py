#!/usr/bin/env python3
"""
CryptoPulse Master Prediction Plot
One comprehensive plot showing all model predictions vs actual values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load dataset and prepare for visualization"""
    print("📊 Loading dataset...")
    
    # Load main dataset
    df = pd.read_csv('../../data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Load feature info
    with open('../../data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    # Split data same as training
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Prepare features
    X_train = train_data[feature_cols].fillna(0)
    X_test = test_data[feature_cols].fillna(0)
    y_train = train_data['direction_1d']
    y_test = test_data['direction_1d']
    
    print(f"✅ Data loaded: {len(test_data)} test samples")
    
    return test_data, X_train, X_test, y_train, y_test, feature_cols

def create_model_predictions(X_train, X_test, y_train):
    """Create predictions from multiple models"""
    print("🤖 Training models and generating predictions...")
    
    predictions = {}
    models = {}
    
    # Model 1: Random Forest (Best performer)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    predictions['Random Forest'] = rf.predict(X_test)
    models['Random Forest'] = rf
    
    # Model 2: Logistic Regression (Simple baseline)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    predictions['Logistic Regression'] = lr.predict(X_test)
    models['Logistic Regression'] = lr
    
    # Model 3: SVM (Different approach)
    svm = SVC(random_state=42, probability=True)
    svm.fit(X_train, y_train)
    predictions['SVM'] = svm.predict(X_test)
    models['SVM'] = svm
    
    # Model 4: Ensemble (Majority vote)
    ensemble_pred = []
    for i in range(len(X_test)):
        votes = [predictions['Random Forest'][i], 
                predictions['Logistic Regression'][i], 
                predictions['SVM'][i]]
        ensemble_pred.append(1 if sum(votes) >= 2 else 0)
    predictions['Ensemble'] = np.array(ensemble_pred)
    
    # Model 5: Baseline (Always predict majority class)
    majority_class = y_train.mode()[0]
    predictions['Baseline'] = np.full(len(X_test), majority_class)
    
    print(f"✅ Generated predictions from {len(predictions)} models")
    return predictions, models

def create_master_plot(test_data, y_test, predictions):
    """Create the master prediction plot"""
    print("🎨 Creating master prediction plot...")
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle('CryptoPulse: Master Model Prediction Analysis', fontsize=20, fontweight='bold')
    
    # Prepare data
    dates = test_data['date'].values
    prices = test_data['price_usd'].values
    actual = y_test.values
    
    # Define colors for each model
    colors = {
        'Actual': 'black',
        'Random Forest': '#2E8B57',      # Sea Green
        'Logistic Regression': '#4169E1', # Royal Blue  
        'SVM': '#FF6347',                # Tomato
        'Ensemble': '#9932CC',           # Dark Orchid
        'Baseline': '#808080'            # Gray
    }
    
    # Plot 1: Price timeline with prediction accuracy
    ax1.plot(dates, prices, 'k-', linewidth=3, alpha=0.8, label='ETH Price', zorder=1)
    
    # Add prediction accuracy markers for each model
    marker_size = 120
    alpha = 0.8
    
    for model_name, pred in predictions.items():
        if model_name == 'Baseline':  # Skip baseline in timeline
            continue
            
        x_positions = []
        y_positions = []
        marker_colors = []
        
        for i, (date, price, prediction, actual_val) in enumerate(zip(dates, prices, pred, actual)):
            x_positions.append(date)
            y_positions.append(price)
            
            # Color based on correctness
            if prediction == actual_val:
                marker_colors.append(colors[model_name])
            else:
                marker_colors.append('red')
        
        ax1.scatter(x_positions, y_positions, c=marker_colors, s=marker_size, 
                   alpha=alpha, label=f'{model_name}', marker='o', edgecolors='white', linewidth=2)
    
    ax1.set_ylabel('ETH Price (USD)', fontsize=14, fontweight='bold')
    ax1.set_title('Timeline: Model Predictions vs Actual Price Movements\n(Colored dots = Correct predictions, Red dots = Wrong predictions)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Prediction comparison matrix
    ax2_main = ax2
    
    # Create prediction matrix for visualization
    model_names = list(predictions.keys())
    n_samples = len(actual)
    n_models = len(model_names)
    
    # Create matrix: rows = models, columns = test samples
    prediction_matrix = np.zeros((n_models + 1, n_samples))  # +1 for actual
    
    # First row is actual values
    prediction_matrix[0, :] = actual
    
    # Following rows are model predictions
    for i, model_name in enumerate(model_names):
        prediction_matrix[i + 1, :] = predictions[model_name]
    
    # Create the heatmap
    im = ax2_main.imshow(prediction_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax2_main.set_yticks(range(n_models + 1))
    ax2_main.set_yticklabels(['ACTUAL'] + model_names, fontsize=12, fontweight='bold')
    ax2_main.set_xlabel('Test Sample Index (Chronological Order)', fontsize=14, fontweight='bold')
    ax2_main.set_title('Prediction Matrix: Green=Up (1), Red=Down (0)\nFirst row shows actual direction, following rows show model predictions', 
                       fontsize=14, fontweight='bold')
    
    # Add sample dates on x-axis (every few samples)
    sample_indices = range(0, n_samples, max(1, n_samples // 8))
    ax2_main.set_xticks(sample_indices)
    ax2_main.set_xticklabels([pd.to_datetime(dates[i]).strftime('%m/%d') for i in sample_indices], rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2_main, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Direction: 0=Down, 1=Up', fontsize=12, fontweight='bold')
    
    # Add accuracy text for each model
    text_y_pos = n_models + 1.5
    for i, model_name in enumerate(model_names):
        pred = predictions[model_name]
        accuracy = np.mean(pred == actual) * 100
        
        # Position text
        ax2_main.text(n_samples + 1, i + 1, f'{accuracy:.1f}%', 
                     fontsize=12, fontweight='bold', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=colors.get(model_name, 'lightblue'), alpha=0.7))
    
    # Add "Accuracy" header
    ax2_main.text(n_samples + 1, 0, 'Actual', fontsize=12, fontweight='bold', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # Extend plot to accommodate accuracy text
    ax2_main.set_xlim(-0.5, n_samples + 5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/master_prediction_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Master plot saved as plots/master_prediction_plot.png")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for model_name, pred in predictions.items():
        accuracy = np.mean(pred == actual) * 100
        correct = np.sum(pred == actual)
        total = len(actual)
        print(f"{model_name:20}: {accuracy:5.1f}% ({correct:2d}/{total:2d} correct)")
    
    print("="*60)
    print(f"📈 Test period: {pd.to_datetime(dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}")
    print(f"📊 Total test samples: {len(actual)}")
    print(f"📈 Actual up days: {np.sum(actual)} ({np.mean(actual)*100:.1f}%)")
    print(f"📉 Actual down days: {len(actual) - np.sum(actual)} ({(1-np.mean(actual))*100:.1f}%)")
    print("="*60)

def main():
    """Main function"""
    print("🎯 CryptoPulse Master Prediction Visualizer")
    print("="*60)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    test_data, X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
    
    # Generate predictions
    predictions, models = create_model_predictions(X_train, X_test, y_train)
    
    # Create master plot
    create_master_plot(test_data, y_test, predictions)
    
    print("\n🎉 Master prediction plot completed!")
    print("📁 Check plots/master_prediction_plot.png")
    print("🎯 This single plot shows:")
    print("   • Timeline with price and prediction accuracy")
    print("   • Prediction matrix comparing all models")
    print("   • Accuracy percentages for each model")
    print("   • Chronological test sample progression")

if __name__ == "__main__":
    main()