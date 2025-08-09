#!/usr/bin/env python3
"""
Comprehensive Regression Model Suite
Train multiple advanced regression models for price change prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    print("📊 Loading dataset for comprehensive regression analysis...")
    
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
    y_train = df.iloc[:train_size]['price_change_1d'].fillna(0)
    y_test = test_data['price_change_1d'].fillna(0)
    
    # Scale features for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Loaded {len(test_data)} test samples")
    print(f"📊 Features: {len(feature_cols)} sentiment/content features")
    print(f"📈 Target range: {y_test.min():.2f}% to {y_test.max():.2f}%")
    
    return test_data, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_comprehensive_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train):
    """Train comprehensive suite of regression models"""
    print("🤖 Training comprehensive regression model suite...")
    
    models = {}
    predictions = {}
    
    # 1. Tree-based models (don't need scaling)
    print("   🌳 Training tree-based models...")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, min_samples_split=5)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf.predict(X_test)
    
    # XGBoost
    try:
        xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42, max_depth=6, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        predictions['XGBoost'] = xgb_model.predict(X_test)
    except:
        print("   ⚠️ XGBoost not available, skipping...")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6, learning_rate=0.1)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = gb.predict(X_test)
    
    # Extra Trees
    et = ExtraTreesRegressor(n_estimators=200, random_state=42, max_depth=10)
    et.fit(X_train, y_train)
    models['Extra Trees'] = et
    predictions['Extra Trees'] = et.predict(X_test)
    
    # 2. Linear models (need scaling)
    print("   📏 Training linear models...")
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr.predict(X_test_scaled)
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge'] = ridge
    predictions['Ridge'] = ridge.predict(X_test_scaled)
    
    # Lasso Regression
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=2000)
    lasso.fit(X_train_scaled, y_train)
    models['Lasso'] = lasso
    predictions['Lasso'] = lasso.predict(X_test_scaled)
    
    # Elastic Net
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
    elastic.fit(X_train_scaled, y_train)
    models['Elastic Net'] = elastic
    predictions['Elastic Net'] = elastic.predict(X_test_scaled)
    
    # 3. Support Vector Regression
    print("   🎯 Training SVR models...")
    
    # SVR with RBF kernel
    svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_rbf.fit(X_train_scaled, y_train)
    models['SVR (RBF)'] = svr_rbf
    predictions['SVR (RBF)'] = svr_rbf.predict(X_test_scaled)
    
    # SVR with linear kernel
    svr_linear = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr_linear.fit(X_train_scaled, y_train)
    models['SVR (Linear)'] = svr_linear
    predictions['SVR (Linear)'] = svr_linear.predict(X_test_scaled)
    
    # 4. Neural Network
    print("   🧠 Training neural network...")
    
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500, 
                       alpha=0.01, learning_rate_init=0.001)
    mlp.fit(X_train_scaled, y_train)
    models['Neural Network'] = mlp
    predictions['Neural Network'] = mlp.predict(X_test_scaled)
    
    # 5. Other models
    print("   🔄 Training additional models...")
    
    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=5)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    predictions['Decision Tree'] = dt.predict(X_test)
    
    # K-Nearest Neighbors
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models['KNN'] = knn
    predictions['KNN'] = knn.predict(X_test_scaled)
    
    # 6. Baselines
    baseline_mean = np.full(len(X_test), y_train.mean())
    predictions['Baseline (Mean)'] = baseline_mean
    
    baseline_median = np.full(len(X_test), y_train.median())
    predictions['Baseline (Median)'] = baseline_median
    
    print(f"✅ Trained {len(models)} models + 2 baselines")
    return models, predictions

def evaluate_comprehensive_models(y_test, predictions):
    """Comprehensive evaluation of all models"""
    print("\n📊 COMPREHENSIVE MODEL EVALUATION:")
    print("="*90)
    
    results = []
    
    for model_name, pred in predictions.items():
        # Regression metrics
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        actual_direction = (y_test > 0).astype(int)
        pred_direction = (pred > 0).astype(int)
        direction_acc = np.mean(actual_direction == pred_direction) * 100
        
        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Direction_Acc': direction_acc
        })
    
    # Sort by RMSE (lower is better)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Dir_Acc':<8}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} {row['RMSE']:<8.3f} {row['MAE']:<8.3f} {row['R²']:<8.3f} {row['Direction_Acc']:<8.1f}%")
    
    return results_df

def create_comprehensive_visualization(test_data, y_test, predictions, results_df):
    """Create comprehensive visualization of all models"""
    print("\n🎨 Creating comprehensive visualization...")
    
    # Get top 5 models by RMSE
    top_models = results_df.head(5)['Model'].tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    
    dates = test_data['date'].values
    actual_changes = y_test.values
    
    # Plot 1: Top 5 models vs actual
    ax1 = axes[0]
    ax1.plot(dates, actual_changes, 'k-', linewidth=4, label='Actual', alpha=0.9)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, model_name in enumerate(top_models):
        if model_name in predictions:
            ax1.plot(dates, predictions[model_name], color=colors[i], 
                    linestyle=linestyles[i], linewidth=2, 
                    label=f'{model_name}', alpha=0.8)
    
    ax1.set_ylabel('Price Change (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Top 5 Regression Models: Predicted vs Actual Price Changes', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Model performance comparison (bar chart)
    ax2 = axes[1]
    
    # Get all models sorted by RMSE
    model_names = results_df['Model'].tolist()
    rmse_values = results_df['RMSE'].tolist()
    
    bars = ax2.bar(range(len(model_names)), rmse_values, alpha=0.7)
    ax2.set_ylabel('RMSE (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Model Performance Comparison (Lower RMSE = Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Color bars - green for best, red for worst
    for i, bar in enumerate(bars):
        if i == 0:  # Best model
            bar.set_color('green')
        elif i >= len(bars) - 2:  # Worst models (baselines)
            bar.set_color('red')
        else:
            bar.set_color('skyblue')
    
    # Add RMSE values on bars
    for i, (bar, rmse) in enumerate(zip(bars, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{rmse:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: R² comparison (scatter plot)
    ax3 = axes[2]
    
    r2_values = results_df['R²'].tolist()
    direction_acc = results_df['Direction_Acc'].tolist()
    
    # Color points by performance
    colors_scatter = []
    for r2, rmse in zip(r2_values, rmse_values):
        if r2 > 0.1 and rmse < 3.0:  # Good performance
            colors_scatter.append('green')
        elif r2 > -0.5 and rmse < 4.0:  # Moderate performance
            colors_scatter.append('orange')
        else:  # Poor performance
            colors_scatter.append('red')
    
    scatter = ax3.scatter(r2_values, direction_acc, c=colors_scatter, s=100, alpha=0.7)
    
    # Add model names as labels
    for i, model_name in enumerate(model_names):
        ax3.annotate(model_name, (r2_values[i], direction_acc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('R² Score', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Direction Accuracy (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Model Performance: R² vs Direction Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    
    plt.tight_layout()
    plt.savefig('plots/comprehensive_regression_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive visualization saved!")

def main():
    """Main function"""
    print("🎯 Comprehensive Regression Model Suite")
    print("="*60)
    print("Training 14+ regression models for ETH price change prediction")
    
    # Load data
    test_data, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_data()
    
    # Train all models
    models, predictions = train_comprehensive_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train)
    
    # Evaluate all models
    results_df = evaluate_comprehensive_models(y_test, predictions)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(test_data, y_test, predictions, results_df)
    
    # Save results
    results_df.to_csv('plots/comprehensive_model_results.csv', index=False)
    
    # Print summary
    best_model = results_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   RMSE: {best_model['RMSE']:.3f}%")
    print(f"   MAE: {best_model['MAE']:.3f}%")
    print(f"   R²: {best_model['R²']:.3f}")
    print(f"   Direction Accuracy: {best_model['Direction_Acc']:.1f}%")
    
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 Files created:")
    print("   • comprehensive_regression_analysis.png")
    print("   • comprehensive_model_results.csv")
    print("💡 Now you have 14+ models compared with smooth regression curves!")
    print("="*60)

if __name__ == "__main__":
    main()