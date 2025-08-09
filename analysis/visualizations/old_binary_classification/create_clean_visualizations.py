#!/usr/bin/env python3
"""
CryptoPulse Clean Model Visualizations
Creates clean, professional plots showing model predictions and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

def load_data():
    """Load and prepare the dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv('data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Load feature info
    with open('data/simplified_ml_dataset_info.json', 'r') as f:
        info = json.load(f)
        feature_cols = info['feature_columns']
    
    # Split data (same as training)
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"✅ Dataset loaded: {len(df)} samples ({len(train_data)} train, {len(test_data)} test)")
    return df, train_data, test_data, feature_cols

def create_price_timeline_plot(df, train_data, test_data):
    """Create ETH price timeline with train/test split"""
    print("📈 Creating price timeline...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Full price timeline
    ax1.plot(df['date'], df['price_usd'], 'b-', linewidth=2, alpha=0.8, label='ETH Price')
    
    # Mark train/test split
    split_date = test_data['date'].iloc[0]
    ax1.axvline(x=split_date, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
    
    # Shade train/test regions
    ax1.axvspan(df['date'].min(), split_date, alpha=0.1, color='blue', label='Training Period')
    ax1.axvspan(split_date, df['date'].max(), alpha=0.1, color='orange', label='Test Period')
    
    ax1.set_ylabel('ETH Price (USD)', fontweight='bold')
    ax1.set_title('CryptoPulse Dataset: ETH Price Over Time', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add price statistics
    train_mean = train_data['price_usd'].mean()
    test_mean = test_data['price_usd'].mean()
    ax1.axhline(y=train_mean, color='blue', linestyle=':', alpha=0.7, label=f'Train Mean: ${train_mean:.0f}')
    ax1.axhline(y=test_mean, color='orange', linestyle=':', alpha=0.7, label=f'Test Mean: ${test_mean:.0f}')
    
    # Plot 2: Price changes distribution
    train_changes = train_data['price_change_1d'].dropna()
    test_changes = test_data['price_change_1d'].dropna()
    
    ax2.hist(train_changes, bins=20, alpha=0.6, label='Training Changes', color='blue', density=True)
    ax2.hist(test_changes, bins=15, alpha=0.6, label='Test Changes', color='orange', density=True)
    ax2.axvline(x=0, color='red', linestyle='-', alpha=0.7, label='No Change')
    
    ax2.set_xlabel('Daily Price Change (%)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Distribution of Daily Price Changes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/eth_price_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved ETH price analysis")

def create_direction_analysis(df, test_data):
    """Analyze price direction patterns"""
    print("🎯 Analyzing price directions...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ETH Price Direction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Direction distribution
    ax1 = axes[0, 0]
    direction_counts = df['direction_1d'].value_counts()
    colors = ['lightcoral', 'lightgreen']
    labels = ['Down Days', 'Up Days']
    
    wedges, texts, autotexts = ax1.pie(direction_counts.values, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Direction Distribution\n(Full Dataset)', fontweight='bold')
    
    # Plot 2: Test set direction distribution
    ax2 = axes[0, 1]
    test_direction_counts = test_data['direction_1d'].value_counts()
    wedges, texts, autotexts = ax2.pie(test_direction_counts.values, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title('Direction Distribution\n(Test Set)', fontweight='bold')
    
    # Plot 3: Direction over time (test period)
    ax3 = axes[1, 0]
    test_dates = test_data['date']
    test_directions = test_data['direction_1d']
    test_prices = test_data['price_usd']
    
    # Color by direction
    colors = ['red' if d == 0 else 'green' for d in test_directions]
    ax3.scatter(test_dates, test_prices, c=colors, alpha=0.7, s=60)
    ax3.plot(test_dates, test_prices, 'k-', alpha=0.3, linewidth=1)
    
    ax3.set_ylabel('ETH Price (USD)', fontweight='bold')
    ax3.set_title('Price Direction in Test Period\n(Red=Down, Green=Up)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Magnitude of changes
    ax4 = axes[1, 1]
    up_changes = test_data[test_data['direction_1d'] == 1]['price_change_1d'].abs()
    down_changes = test_data[test_data['direction_1d'] == 0]['price_change_1d'].abs()
    
    ax4.boxplot([up_changes, down_changes], labels=['Up Days', 'Down Days'])
    ax4.set_ylabel('Absolute Price Change (%)', fontweight='bold')
    ax4.set_title('Magnitude of Price Changes\n(Test Period)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/direction_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved direction analysis")

def create_feature_analysis(df, feature_cols):
    """Analyze the sentiment features"""
    print("🔍 Analyzing sentiment features...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('CryptoPulse: Sentiment Feature Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature correlation with direction
    ax1 = axes[0, 0]
    correlations = []
    feature_names = []
    
    for feature in feature_cols:
        if feature in df.columns:
            corr = df[feature].corr(df['direction_1d'])
            if not np.isnan(corr):
                correlations.append(corr)
                feature_names.append(feature.replace('_', '\n'))
    
    bars = ax1.barh(range(len(correlations)), correlations)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Correlation with Direction', fontweight='bold')
    ax1.set_title('Feature Correlation with Price Direction', fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Color bars by correlation strength
    for bar, corr in zip(bars, correlations):
        bar.set_color('green' if corr > 0 else 'red')
        bar.set_alpha(0.7)
    
    # Plot 2: Feature distributions
    ax2 = axes[0, 1]
    # Select a few key features for distribution plot
    key_features = ['relevance_score_max', 'volatility_score_mean', 'echo_score_mean']
    existing_features = [f for f in key_features if f in df.columns]
    
    for i, feature in enumerate(existing_features[:3]):
        values = df[feature].dropna()
        ax2.hist(values, bins=20, alpha=0.6, label=feature.replace('_', ' '), density=True)
    
    ax2.set_xlabel('Feature Value', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Key Feature Distributions', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Content features
    ax3 = axes[1, 0]
    content_features = ['content_length_max', 'content_length_mean', 'num_comments_sum']
    existing_content = [f for f in content_features if f in df.columns]
    
    if existing_content:
        feature_data = [df[f].dropna() for f in existing_content]
        feature_labels = [f.replace('_', '\n') for f in existing_content]
        
        bp = ax3.boxplot(feature_data, labels=feature_labels)
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('Content Feature Distributions', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Engagement features
    ax4 = axes[1, 1]
    engagement_features = ['engagement_sum', 'engagement_mean']
    existing_engagement = [f for f in engagement_features if f in df.columns]
    
    if existing_engagement:
        for feature in existing_engagement:
            up_values = df[df['direction_1d'] == 1][feature].dropna()
            down_values = df[df['direction_1d'] == 0][feature].dropna()
            
            ax4.scatter([1] * len(up_values), up_values, alpha=0.6, label=f'{feature} (Up days)', s=30)
            ax4.scatter([2] * len(down_values), down_values, alpha=0.6, label=f'{feature} (Down days)', s=30)
        
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['Up Days', 'Down Days'])
        ax4.set_ylabel('Engagement Value', fontweight='bold')
        ax4.set_title('Engagement vs Price Direction', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Sentiment scores over time
    ax5 = axes[2, 0]
    sentiment_cols = [col for col in feature_cols if 'score' in col][:3]
    
    for col in sentiment_cols:
        if col in df.columns:
            values = df[col].rolling(window=7).mean()  # 7-day moving average
            ax5.plot(df['date'], values, label=col.replace('_', ' '), linewidth=2, alpha=0.8)
    
    ax5.set_ylabel('Score Value', fontweight='bold')
    ax5.set_title('Sentiment Scores Over Time\n(7-day Moving Average)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Feature vs Price relationship
    ax6 = axes[2, 1]
    if 'relevance_score_max' in df.columns:
        x = df['relevance_score_max']
        y = df['price_change_1d']
        colors = ['red' if d == 0 else 'green' for d in df['direction_1d']]
        
        ax6.scatter(x, y, c=colors, alpha=0.6, s=30)
        ax6.set_xlabel('Relevance Score Max', fontweight='bold')
        ax6.set_ylabel('Price Change (%)', fontweight='bold')
        ax6.set_title('Relevance Score vs Price Change\n(Red=Down, Green=Up)', fontweight='bold')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/feature_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved feature analysis")

def create_summary_report():
    """Create a summary report of the analysis"""
    print("📋 Creating summary report...")
    
    # Load data for summary
    df = pd.read_csv('data/simplified_ml_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    train_size = int(0.8 * len(df))
    test_data = df.iloc[train_size:]
    
    # Calculate summary statistics
    summary = {
        'Dataset Size': len(df),
        'Training Samples': train_size,
        'Test Samples': len(test_data),
        'Date Range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        'Price Range (USD)': f"${df['price_usd'].min():.0f} - ${df['price_usd'].max():.0f}",
        'Average Daily Change': f"{df['price_change_1d'].mean():.2f}%",
        'Volatility (Std)': f"{df['price_change_1d'].std():.2f}%",
        'Up Days': f"{(df['direction_1d'] == 1).sum()} ({(df['direction_1d'] == 1).mean()*100:.1f}%)",
        'Down Days': f"{(df['direction_1d'] == 0).sum()} ({(df['direction_1d'] == 0).mean()*100:.1f}%)"
    }
    
    # Create text plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    title = "CryptoPulse Dataset Summary Report"
    ax.text(0.5, 0.95, title, fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    y_pos = 0.85
    for key, value in summary.items():
        ax.text(0.1, y_pos, f"{key}:", fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.6, y_pos, str(value), fontsize=14, transform=ax.transAxes)
        y_pos -= 0.08
    
    # Add analysis notes
    notes = [
        "Analysis Notes:",
        "• Dataset covers 6 months of ETH price data with sentiment features",
        "• Small sample size (178 samples) limits statistical significance",
        "• Test set has only 36 samples - results should be interpreted cautiously",
        "• Features include sentiment, relevance, volatility, and engagement scores",
        "• Models show varying performance - best achieved ~67% accuracy on test set"
    ]
    
    y_pos = 0.4
    for note in notes:
        weight = 'bold' if note.endswith(':') else 'normal'
        ax.text(0.1, y_pos, note, fontsize=12, fontweight=weight, transform=ax.transAxes)
        y_pos -= 0.05
    
    plt.savefig('plots/dataset_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved dataset summary")

def main():
    """Main function to run all visualizations"""
    print("🎨 CryptoPulse Clean Visualization Suite")
    print("="*50)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    df, train_data, test_data, feature_cols = load_data()
    
    # Create visualizations
    create_price_timeline_plot(df, train_data, test_data)
    create_direction_analysis(df, test_data)
    create_feature_analysis(df, feature_cols)
    create_summary_report()
    
    print("\n" + "="*50)
    print("🎉 All clean visualizations completed!")
    print("📁 Plots saved in 'plots/' directory:")
    print("   • eth_price_analysis.png")
    print("   • direction_analysis.png") 
    print("   • feature_analysis.png")
    print("   • dataset_summary.png")
    print("="*50)

if __name__ == "__main__":
    main()