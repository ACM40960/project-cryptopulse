#!/usr/bin/env python3
"""
Generate Updated Model Comparison Plot for CryptoPulse Poster

This script creates a comprehensive model comparison plot reflecting the current
project status and key finding about overfitting vs. robust modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Plotting Style Configuration ---

# Consistent color palette
COLORS = {
    'primary': '#003f5c',
    'secondary': '#58508d',
    'accent': '#bc5090',
    'highlight': '#ff6361',
    'neutral': '#ffa600',
    'up': '#4CAF50',
    'down': '#F44336',
    'background': '#F5F5F5',
    'complex': '#ff6361',
    'simple': '#4CAF50',
    'baseline': '#58508d'
}

# Matplotlib style
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelcolor': COLORS['primary'],
    'axes.titlecolor': COLORS['primary'],
    'xtick.color': COLORS['secondary'],
    'ytick.color': COLORS['secondary'],
    'axes.edgecolor': COLORS['primary'],
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#E0E0E0',
    'figure.autolayout': True
})

def generate_model_comparison_plot(save_path):
    """Generates and saves the updated model comparison plot."""
    print("📊 Generating updated model comparison plot...")
    
    # Current model results based on the project documentation
    model_data = {
        'Model': ['LightGBM\n(Complex)', 'Random Forest\n(Sentiment-Enhanced)', 'XGBoost\n(Sentiment-Enhanced)', 
                  'Logistic Regression\n(Simple)', 'Baseline\n(Majority Class)'],
        'Accuracy': [0.75, 0.528, 0.50, 0.333, 0.517],
        'Type': ['Complex', 'Complex', 'Complex', 'Simple', 'Baseline'],
        'Up_Accuracy': [1.0, 0.583, 0.417, 0.167, 0.517],
        'Down_Accuracy': [0.25, 0.417, 0.667, 0.70, 0.483],
        'F1_Score': [0.695, 0.538, 0.508, 0.303, 0.350],
        'Key_Feature': ['content_length', 'sentiment_mix', 'sentiment_mix', 'robust_features', 'none']
    }
    
    df = pd.DataFrame(model_data)
    
    # --- Create the plot ---
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout with more spacing
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    
    # Main title
    fig.suptitle('CryptoPulse: Model Performance Comparison', 
                 fontsize=20, fontweight='bold', color=COLORS['primary'], y=0.98)
    
    # Plot 1: Overall Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [COLORS['complex'] if t == 'Complex' else COLORS['simple'] if t == 'Simple' else COLORS['baseline'] 
              for t in df['Type']]
    bars = ax1.bar(range(len(df)), df['Accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Overall Test Accuracy\n(Small Dataset: 36 Test Samples)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    
    # Add percentage labels on bars
    for i, (bar, acc) in enumerate(zip(bars, df['Accuracy'])):
        ax1.annotate(f'{acc*100:.1f}%', 
                     (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Directional Accuracy Breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df['Up_Accuracy'], width, label='Up Days', color=COLORS['up'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, df['Down_Accuracy'], width, label='Down Days', color=COLORS['down'], alpha=0.8)
    
    ax2.set_title('Directional Accuracy Breakdown\n(Up vs Down Days)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model Reliability Analysis
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create reliability score (inverse of accuracy variance between up/down)
    reliability = 1 - abs(df['Up_Accuracy'] - df['Down_Accuracy'])
    
    scatter_colors = [COLORS['complex'] if t == 'Complex' else COLORS['simple'] if t == 'Simple' else COLORS['baseline'] 
                     for t in df['Type']]
    
    scatter = ax3.scatter(df['Accuracy'], reliability, c=scatter_colors, s=200, alpha=0.8, edgecolors='black')
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax3.annotate(model.replace('\n', ' '), (df['Accuracy'].iloc[i], reliability.iloc[i]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))
    
    ax3.set_title('Model Reliability: Accuracy vs. Balanced Performance\n(Closer to top-right = better)', 
                  fontsize=16, fontweight='bold')
    ax3.set_xlabel('Overall Test Accuracy', fontsize=14)
    ax3.set_ylabel('Reliability Score\n(1 - |Up_Acc - Down_Acc|)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend for model types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['complex'], 
                  markersize=12, label='Complex Models (Prone to Overfitting)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['simple'], 
                  markersize=12, label='Simple Models (More Robust)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'], 
                  markersize=12, label='Baseline Models')
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    
    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Model comparison plot saved to {save_path}")

if __name__ == "__main__":
    # Correct the working directory to be the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.chdir(project_root)
    
    SAVE_PATH = 'analysis/visualizations/plots/model_comparison.png'
    
    generate_model_comparison_plot(SAVE_PATH)
    print("🎉 Model comparison plot generation complete!")