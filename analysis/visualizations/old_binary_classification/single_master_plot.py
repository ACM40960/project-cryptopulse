#!/usr/bin/env python3
"""
Single Master Plot - All Models vs Actual
One comprehensive plot showing actual values and predictions from all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    print("📊 Loading dataset...")
    
    # Navigate to correct path from visualizations directory
    df = pd.read_csv('../../data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Load features
    with open('../../data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    # Split same as original training
    train_size = int(0.8 * len(df))
    test_data = df.iloc[train_size:].copy()
    
    # Prepare features
    X_train = df.iloc[:train_size][feature_cols].fillna(0)
    X_test = test_data[feature_cols].fillna(0)
    y_train = df.iloc[:train_size]['direction_1d']
    y_test = test_data['direction_1d']
    
    print(f"✅ Loaded {len(test_data)} test samples from {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    
    return test_data, X_train, X_test, y_train, y_test

def train_models_and_predict(X_train, X_test, y_train):
    """Train multiple models and get predictions"""
    print("🤖 Training models...")
    
    models = {}
    predictions = {}
    
    # Random Forest - Best performer
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # Logistic Regression - Simple baseline  
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = lr.predict(X_test)
    
    # SVM - Different approach
    svm = SVC(random_state=42, kernel='rbf')
    svm.fit(X_train, y_train)
    models['SVM'] = svm  
    predictions['SVM'] = svm.predict(X_test)
    
    # Majority baseline
    majority_class = y_train.mode()[0]
    predictions['Baseline (Majority)'] = np.full(len(X_test), majority_class)
    
    print(f"✅ Trained {len(models)} models + baseline")
    return predictions

def create_single_master_plot(test_data, y_test, predictions):
    """Create THE single plot showing everything"""
    print("🎨 Creating single master plot...")
    
    # Set up the plot - SINGLE LARGE PLOT
    plt.figure(figsize=(20, 12))
    
    # Prepare data
    dates = test_data['date'].values
    prices = test_data['price_usd'].values
    actual_directions = y_test.values
    
    # Define distinct colors for each model + actual
    colors = {
        'ACTUAL': '#000000',           # Black - most important
        'Random Forest': '#228B22',    # Forest Green - best model
        'Logistic Regression': '#4169E1',  # Royal Blue  
        'SVM': '#FF4500',             # Orange Red
        'Baseline (Majority)': '#808080'  # Gray - worst model
    }
    
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(20, 12))
    
    # Plot ETH price as line (background)
    ax1.plot(dates, prices, color='lightgray', linewidth=2, alpha=0.6, 
             label='ETH Price (USD)', zorder=1)
    ax1.set_ylabel('ETH Price (USD)', fontsize=16, fontweight='bold', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    # Create second y-axis for predictions
    ax2 = ax1.twinx()
    
    # Plot ACTUAL values first (most important)
    y_positions_actual = np.where(actual_directions == 1, 1.1, -0.1)  # Up=1.1, Down=-0.1
    ax2.scatter(dates, y_positions_actual, 
               color=colors['ACTUAL'], s=200, alpha=0.9, 
               marker='s', edgecolors='white', linewidth=2,
               label='ACTUAL', zorder=10)
    
    # Plot each model's predictions
    model_names = [name for name in predictions.keys()]
    y_offsets = [0.9, 0.7, 0.5, 0.3]  # Different heights for each model
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        # Calculate positions: Up predictions go above 0.5, Down below 0.5
        y_positions = np.where(pred == 1, y_offsets[i], -y_offsets[i])
        
        # Different markers for each model
        markers = ['o', '^', 'v', 'D']
        marker = markers[i % len(markers)]
        
        # Color correct/incorrect predictions differently
        scatter_colors = []
        for j, (prediction, actual) in enumerate(zip(pred, actual_directions)):
            if prediction == actual:
                scatter_colors.append(colors[model_name])  # Correct = model color
            else:
                scatter_colors.append('red')  # Incorrect = red
        
        ax2.scatter(dates, y_positions, 
                   c=scatter_colors, s=150, alpha=0.8,
                   marker=marker, edgecolors='white', linewidth=1,
                   label=f'{model_name}', zorder=5)
    
    # Customize the prediction axis
    ax2.set_ylabel('Predictions (Up = Positive, Down = Negative)', 
                   fontsize=16, fontweight='bold')
    ax2.set_ylim(-1.3, 1.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticklabels(['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up'])
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add title and legend
    plt.title('CryptoPulse: Single Master View - All Model Predictions vs Actual\n' +
              'Black squares = ACTUAL direction | Colored shapes = Model predictions\n' +
              'Model color = Correct prediction | Red = Wrong prediction',
              fontsize=18, fontweight='bold', pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add accuracy text box
    accuracy_text = "MODEL ACCURACY:\n"
    for model_name, pred in predictions.items():
        acc = np.mean(pred == actual_directions) * 100
        accuracy_text += f"{model_name}: {acc:.1f}%\n"
    
    # Add text box with accuracies
    ax2.text(0.02, 0.98, accuracy_text, transform=ax2.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('plots/single_master_plot.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Single master plot saved!")
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 SINGLE PLOT SUMMARY - EVERYTHING YOU NEED TO KNOW")
    print("="*80)
    print("📊 What the plot shows:")
    print("   • Gray line: ETH price over time")
    print("   • Black squares: ACTUAL price direction (up/down)")
    print("   • Colored shapes: Model predictions")
    print("   • Model color = Correct prediction")
    print("   • Red = Wrong prediction")
    print("")
    print("📈 Model Performance:")
    for model_name, pred in predictions.items():
        acc = np.mean(pred == actual_directions) * 100
        correct = np.sum(pred == actual_directions)
        total = len(actual_directions)
        print(f"   {model_name:20}: {acc:5.1f}% ({correct:2d}/{total} correct)")
    print("="*80)

def main():
    """Main execution"""
    print("🎯 CryptoPulse: Single Master Plot Creator")
    print("="*60)
    
    # Load data
    test_data, X_train, X_test, y_train, y_test = load_data()
    
    # Get predictions from all models
    predictions = train_models_and_predict(X_train, X_test, y_train)
    
    # Create THE single plot
    create_single_master_plot(test_data, y_test, predictions)
    
    print("\n🎉 DONE! Check plots/single_master_plot.png")
    print("📋 This ONE plot shows you everything:")
    print("   ✅ Actual ETH price timeline")
    print("   ✅ Actual price directions") 
    print("   ✅ All model predictions")
    print("   ✅ Which predictions were correct/wrong")
    print("   ✅ Model accuracy percentages")

if __name__ == "__main__":
    main()