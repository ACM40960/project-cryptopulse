#!/usr/bin/env python3
"""
Generate Dataset Summary Table for CryptoPulse Poster

This script creates a clean, informative table summarizing the dataset characteristics,
more suitable for poster presentation than a simple plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Plotting Style Configuration ---
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def generate_dataset_table(data_path, save_path):
    """Generates and saves a simple, compact dataset summary table."""
    print("📊 Generating dataset summary table...")
    
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure the path is correct.")
        return
    
    # Calculate key statistics
    total_samples = len(df)
    date_range_start = df['date'].min().strftime('%b %d, %Y')
    date_range_end = df['date'].max().strftime('%b %d, %Y')
    
    # Direction analysis
    up_days = (df['direction_1d'] == 1).sum()
    down_days = (df['direction_1d'] == 0).sum()
    up_pct = (up_days / total_samples) * 100
    down_pct = (down_days / total_samples) * 100
    
    # Price statistics
    price_mean = df['price_usd'].mean()
    price_min = df['price_usd'].min()
    price_max = df['price_usd'].max()
    
    # Feature count
    feature_cols = [col for col in df.columns if col not in ['date', 'price_usd', 'price_change_1d', 'price_change_3d', 'price_change_7d', 'direction_1d', 'direction_3d', 'direction_7d', 'price_ma_7', 'price_volatility']]
    num_features = len(feature_cols)
    
    # Create simplified table data
    table_data = [
        ['Metric', 'Value'],
        ['Total Samples', f'{total_samples}'],
        ['Date Range', f'{date_range_start} - {date_range_end}'],
        ['Up Days', f'{up_days} ({up_pct:.1f}%)'],
        ['Down Days', f'{down_days} ({down_pct:.1f}%)'],
        ['Avg ETH Price', f'${price_mean:,.0f}'],
        ['Price Range', f'${price_min:,.0f} - ${price_max:,.0f}'],
        ['Features', f'{num_features} sentiment-based'],
        ['Test Set Size', '36 samples'],
        ['Limitation', 'Small dataset (overfitting risk)']
    ]
    
    # Create figure with absolutely no margins - table fills entire image
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create table that fills entire figure
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='left', loc='center', 
                     bbox=[0, 0, 1, 1])  # Table fills entire axes
    
    # Normal readable styling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.1)  # Normal spacing for readability
    
    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#f0f0f0')
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_height(0.08)
    
    # Highlight the limitation row
    limitation_row = len(table_data) - 2
    for j in range(len(table_data[0])):
        table[(limitation_row, j)].set_facecolor('#ffe6e6')
        if j == 1:  # Value column
            table[(limitation_row, j)].set_text_props(weight='bold')
    
    # Set normal row heights for readability
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            table[(i-1, j)].set_height(0.07)
    
    # Adjust column widths
    cellDict = table.get_celld()
    for i in range(len(table_data)):
        cellDict[(i, 0)].set_width(0.4)
        cellDict[(i, 1)].set_width(0.6)
    
    # No title - table fills entire image
    
    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Dataset summary table saved to {save_path}")

if __name__ == "__main__":
    # Correct the working directory to be the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    os.chdir(project_root)
    
    DATASET_PATH = 'data/simplified_ml_dataset.csv'
    SAVE_PATH = 'analysis/visualizations/plots/dataset_summary.png'
    
    generate_dataset_table(DATASET_PATH, SAVE_PATH)
    print("🎉 Dataset summary table generation complete!")