#!/usr/bin/env python3
"""
Create Poster-Ready Plots for CryptoPulse

This script generates high-quality, focused visualizations suitable for a
scientific poster, addressing the key findings of the CryptoPulse project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

# Add src directory to path to import the trainer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ml_model_trainer import CryptoPulseMLTrainer

# --- Plotting Style Configuration ---

# Consistent color palette
COLORS = {
    'primary': '#003f5c',
    'secondary': '#58508d',
    'accent': '#bc5090',
    'highlight': '#ff6361',
    'neutral': '#ffa600',
    'up': '#4CAF50', # Green
    'down': '#F44336', # Red
    'background': '#F5F5F5'
}

# Matplotlib style
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelcolor': COLORS['primary'],
    'axes.titlecolor': COLORS['primary'],
    'xtick.color': COLORS['secondary'],
    'ytick.color': COLORS['secondary'],
    'axes.edgecolor': COLORS['primary'],
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#E0E0E0'
})

# --- Plotting Functions ---

def create_confusion_matrix_plot(y_true, y_pred, model_name, save_path):
    """Creates and saves a high-quality confusion matrix plot."""
    print(f"🎨 Creating confusion matrix for {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=.5, linecolor='gray', cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    
    ax.set_title(f'{model_name}: Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Down', 'Up'], fontsize=12)
    ax.set_yticklabels(['Down', 'Up'], fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix to {save_path}")

def create_directional_accuracy_plot(model_comparison_df, save_path):
    """Creates a bar chart comparing up vs. down day accuracy for each model."""
    print("📊 Creating directional accuracy comparison plot...")
    
    df = model_comparison_df.copy()
    df = df.set_index('Model')
    
    # Select only the directional accuracies
    dir_acc_df = df[['Up_Days_Acc', 'Down_Days_Acc']] * 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dir_acc_df.plot(kind='bar', ax=ax, color=[COLORS['up'], COLORS['down']], width=0.8)
    
    ax.set_title('Model Performance: Directional Accuracy', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.legend(['Up Day Accuracy', 'Down Day Accuracy'], fontsize=12, loc='upper right')
    ax.axhline(50, color='black', linestyle='--', linewidth=1, label='Random Chance')
    
    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9),
                      textcoords='offset points', fontsize=10, fontweight='bold',
                      color=COLORS['primary'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved directional accuracy plot to {save_path}")

# --- Main Execution ---

def main():
    """Main function to generate all poster plots."""
    print("🚀 Starting CryptoPulse Poster Plot Generation")
    print("="*50)
    
    # Define output directory
    output_dir = 'plots/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Train models to get results
    print("🧠 Training models to get fresh results...")
    trainer = CryptoPulseMLTrainer()
    results = trainer.train_all_models(target='direction_1d')
    comparison_df, best_model_name = trainer.compare_models(target='direction_1d')
    
    print(f"🏆 Best model identified: {best_model_name}")
    
    # 2. Get predictions for the best model
    best_model_results = results[best_model_name]
    y_true = best_model_results['predictions']['y_test']
    y_pred = best_model_results['predictions']['test_pred']
    
    # 3. Create and save the plots
    cm_path = os.path.join(output_dir, f'{best_model_name}_confusion_matrix.png')
    create_confusion_matrix_plot(y_true, y_pred, best_model_name, cm_path)
    
    dir_acc_path = os.path.join(output_dir, 'model_directional_accuracy_comparison.png')
    create_directional_accuracy_plot(comparison_df, dir_acc_path)
    
    print("\n" + "="*50)
    print("🎉 Poster plot generation complete!")
    print("📁 New plots saved in 'plots/' directory:")
    print(f"   - {os.path.basename(cm_path)}")
    print(f"   - {os.path.basename(dir_acc_path)}")
    print("="*50)

if __name__ == "__main__":
    # Correct the working directory to be the project root
    # so that the script can find the data and src folders.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.chdir(project_root)
    main()
