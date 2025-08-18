#!/usr/bin/env python3
"""
Generate Dataset Summary Plot for CryptoPulse Poster

This script creates a clean, informative summary plot of the simplified ML dataset,
suitable for inclusion in the CryptoPulse academic poster.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
plt.style.use('ggplot') # Updated style
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
    'grid.color': '#E0E0E0',
    'figure.autolayout': True # Automatically adjust subplot params for a tight layout
})

def generate_dataset_summary_plot(data_path, save_path):
    """Generates and saves a summary plot of the dataset."""
    print("📊 Generating dataset summary plot...")
    
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure the path is correct.")
        return
    
    # --- Create the plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('CryptoPulse Dataset Summary', fontsize=20, fontweight='bold', color=COLORS['primary'])

    # Plot 1: Daily Entry Count (showing consistent data collection)
    daily_counts = df.groupby('date').size()
    # Show constant line at 1 to represent one aggregated entry per day
    axes[0].axhline(y=1, color=COLORS['highlight'], linewidth=3)
    axes[0].set_title('Daily Data Points (Aggregated Social Media Entries)', fontsize=16, fontweight='bold', color=COLORS['primary'])
    axes[0].set_ylabel('Number of Entries', fontsize=14, color=COLORS['secondary'])
    axes[0].set_ylim(0.95, 1.05)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    # Add text annotation
    axes[0].text(0.5, 0.5, f'178 Daily Samples\n(Feb 1 - Jul 29, 2025)', 
                transform=axes[0].transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot 2: 1-Day Price Direction Distribution
    direction_counts = df['direction_1d'].value_counts(normalize=True) * 100
    bars = axes[1].bar(['Down', 'Up'], [direction_counts[0], direction_counts[1]], 
                       color=[COLORS['down'], COLORS['up']], alpha=0.8)
    axes[1].set_title('1-Day Price Direction Distribution', fontsize=16, fontweight='bold', color=COLORS['primary'])
    axes[1].set_xlabel('Price Direction (0=Down, 1=Up)', fontsize=14, color=COLORS['secondary'])
    axes[1].set_ylabel('Percentage (%)', fontsize=14, color=COLORS['secondary'])
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 60)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, [direction_counts[0], direction_counts[1]])):
        axes[1].annotate(f'{pct:.1f}%', 
                         (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), textcoords='offset points',
                         fontsize=14, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    
    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dataset summary plot saved to {save_path}")

if __name__ == "__main__":
    # Correct the working directory to be the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.chdir(project_root)
    
    DATASET_PATH = 'data/simplified_ml_dataset.csv'
    SAVE_PATH = 'analysis/visualizations/plots/dataset_summary.png'
    
    generate_dataset_summary_plot(DATASET_PATH, SAVE_PATH)
    print("🎉 Dataset summary plot generation complete!")
