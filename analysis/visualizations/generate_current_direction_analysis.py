#!/usr/bin/env python3
"""
Generate Updated Direction Analysis Plot for CryptoPulse Poster

This script creates an updated direction analysis plot reflecting the current
project's understanding of price movements and dataset characteristics.
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
    'background': '#F5F5F5'
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

def generate_direction_analysis_plot(data_path, save_path):
    """Generates and saves the updated direction analysis plot."""
    print("📊 Generating updated direction analysis plot...")
    
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure the path is correct.")
        return
    
    # --- Create the plot ---
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('ETH Price Direction Analysis\nCryptoPulse Dataset (Feb-Jul 2025)', 
                 fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.95)
    
    # Calculate statistics
    total_samples = len(df)
    up_pct = (df['direction_1d'] == 1).sum() / total_samples * 100
    down_pct = (df['direction_1d'] == 0).sum() / total_samples * 100
    
    # Simulate test set split (last 36 samples as commonly used in the project)
    test_samples = 36
    test_df = df.tail(test_samples)
    test_up_pct = (test_df['direction_1d'] == 1).sum() / test_samples * 100
    test_down_pct = (test_df['direction_1d'] == 0).sum() / test_samples * 100
    
    # Plot 1: Overall Direction Distribution (Full Dataset)
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [down_pct, up_pct]
    colors = [COLORS['down'], COLORS['up']]
    wedges, texts, autotexts = ax1.pie(sizes, labels=['Down Days', 'Up Days'], 
                                      colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Overall Direction Distribution\n(Full Dataset)', fontsize=14, fontweight='bold')
    
    # Plot 2: Test Set Direction Distribution  
    ax2 = fig.add_subplot(gs[0, 1])
    test_sizes = [test_down_pct, test_up_pct]
    wedges2, texts2, autotexts2 = ax2.pie(test_sizes, labels=['Down Days', 'Up Days'], 
                                         colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Direction Distribution\n(Test Set)', fontsize=14, fontweight='bold')
    
    # Plot 3: Price Timeline with Direction Colors
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get price data and dates
    prices = df['price_usd'].values
    dates = df['date'].values
    directions = df['direction_1d'].values
    
    # Create scatter plot with colors based on direction
    up_mask = directions == 1
    down_mask = directions == 0
    
    ax3.plot(dates, prices, color='lightgray', alpha=0.7, linewidth=1)
    ax3.scatter(dates[up_mask], prices[up_mask], color=COLORS['up'], 
               label=f'Up Days ({up_pct:.1f}%)', s=30, alpha=0.8)
    ax3.scatter(dates[down_mask], prices[down_mask], color=COLORS['down'], 
               label=f'Down Days ({down_pct:.1f}%)', s=30, alpha=0.8)
    
    ax3.set_title('Price Direction in Test Period\n(Red=Down, Green=Up)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('ETH Price (USD)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price Change Magnitude Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate absolute price changes
    up_changes = abs(df[df['direction_1d'] == 1]['price_change_1d'])
    down_changes = abs(df[df['direction_1d'] == 0]['price_change_1d'])
    
    # Create box plot
    box_data = [up_changes, down_changes]
    box_plot = ax4.boxplot(box_data, labels=['Up Days', 'Down Days'], 
                          patch_artist=True, notch=True)
    
    # Color the boxes
    box_plot['boxes'][0].set_facecolor(COLORS['up'])
    box_plot['boxes'][0].set_alpha(0.7)
    box_plot['boxes'][1].set_facecolor(COLORS['down']) 
    box_plot['boxes'][1].set_alpha(0.7)
    
    ax4.set_title('Magnitude of Price Changes\n(Test Period)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Absolute Price Change (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Dataset Statistics:
Total Samples: {total_samples}
Period: Feb 1 - Jul 29, 2025
Test Set: {test_samples} samples
Balanced: {abs(up_pct - 50) < 5}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Direction analysis plot saved to {save_path}")

if __name__ == "__main__":
    # Correct the working directory to be the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.chdir(project_root)
    
    DATASET_PATH = 'data/simplified_ml_dataset.csv'
    SAVE_PATH = 'analysis/visualizations/plots/direction_analysis.png'
    
    generate_direction_analysis_plot(DATASET_PATH, SAVE_PATH)
    print("🎉 Direction analysis plot generation complete!")