#!/usr/bin/env python3
# src/ml_dataset_creator.py

"""
ML Dataset Creator for CryptoPulse
Creates ML-ready dataset by aligning text metrics with ETH price movements.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Add src to path
sys.path.append(os.path.dirname(__file__))
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_dataset_creation.log'),
        logging.StreamHandler()
    ]
)

class MLDatasetCreator:
    def __init__(self):
        self.db = CryptoPulseDB()
        logging.info("🤖 ML Dataset Creator initialized")
    
    def analyze_data_coverage(self):
        """Analyze the coverage of text metrics and price data."""
        conn = sqlite3.connect(self.db.db_path)
        
        try:
            # Modern text metrics coverage
            metrics_query = """
                SELECT 
                    DATE(datetime(created_utc, 'unixepoch')) as date,
                    'reddit' as source,
                    created_utc,
                    sentiment_score,
                    relevance_score,
                    volatility_score,
                    echo_score,
                    content_depth_score
                FROM reddit_posts r
                JOIN modern_text_metrics m ON r.id = m.id
                WHERE datetime(r.created_utc, 'unixepoch') >= '2025-02-01' 
                  AND datetime(r.created_utc, 'unixepoch') <= '2025-07-31'
                
                UNION ALL
                
                SELECT 
                    DATE(datetime(created_at, 'unixepoch')) as date,
                    'twitter' as source,
                    created_at as created_utc,
                    sentiment_score,
                    relevance_score,
                    volatility_score,
                    echo_score,
                    content_depth_score
                FROM twitter_posts t
                JOIN modern_text_metrics m ON t.id = m.id
                WHERE datetime(t.created_at, 'unixepoch') >= '2025-02-01' 
                  AND datetime(t.created_at, 'unixepoch') <= '2025-07-31'
                
                UNION ALL
                
                SELECT 
                    DATE(datetime(published_at, 'unixepoch')) as date,
                    'news' as source,
                    published_at as created_utc,
                    sentiment_score,
                    relevance_score,
                    volatility_score,
                    echo_score,
                    content_depth_score
                FROM news_articles n
                JOIN modern_text_metrics m ON n.id = m.id
                WHERE datetime(n.published_at, 'unixepoch') >= '2025-02-01' 
                  AND datetime(n.published_at, 'unixepoch') <= '2025-07-31'
                
                ORDER BY created_utc
            """
            
            metrics_df = pd.read_sql_query(metrics_query, conn)
            
            # ETH price data coverage
            price_query = """
                SELECT 
                    DATE(datetime(timestamp, 'unixepoch')) as date,
                    datetime(timestamp, 'unixepoch') as datetime,
                    timestamp,
                    price_usd,
                    volume_24h,
                    market_cap
                FROM eth_prices 
                WHERE datetime(timestamp, 'unixepoch') >= '2025-02-01' 
                  AND datetime(timestamp, 'unixepoch') <= '2025-07-31'
                ORDER BY timestamp
            """
            
            price_df = pd.read_sql_query(price_query, conn)
            
            logging.info(f"📊 Data Coverage Analysis:")
            logging.info(f"   Text entries with metrics: {len(metrics_df):,}")
            logging.info(f"   ETH price points: {len(price_df):,}")
            logging.info(f"   Text date range: {metrics_df['date'].min()} to {metrics_df['date'].max()}")
            logging.info(f"   Price date range: {price_df['date'].min()} to {price_df['date'].max()}")
            logging.info(f"   Unique text days: {metrics_df['date'].nunique()}")
            logging.info(f"   Unique price days: {price_df['date'].nunique()}")
            
            return metrics_df, price_df
            
        finally:
            conn.close()
    
    def create_daily_aggregates(self, metrics_df):
        """Create daily aggregated features from text metrics."""
        logging.info("📈 Creating daily sentiment aggregates...")
        
        # Convert to datetime for easier manipulation
        metrics_df['datetime'] = pd.to_datetime(metrics_df['created_utc'], unit='s')
        metrics_df['date'] = metrics_df['datetime'].dt.date
        
        # Create daily aggregates
        daily_agg = metrics_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'relevance_score': ['mean', 'std', 'max'],
            'volatility_score': ['mean', 'std', 'max', 'sum'],
            'echo_score': ['mean', 'std', 'max'],
            'content_depth_score': ['mean', 'std', 'max']
        }).round(6)
        
        # Flatten column names
        daily_agg.columns = [f"{col[0]}_{col[1]}" for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()
        
        # Add source-specific aggregates
        source_agg = metrics_df.groupby(['date', 'source']).agg({
            'sentiment_score': 'mean',
            'relevance_score': 'mean',
            'volatility_score': 'mean',
            'echo_score': 'mean',
            'content_depth_score': 'mean'
        }).round(6)
        
        # Pivot to get source columns
        source_pivot = source_agg.reset_index().pivot(
            index='date', 
            columns='source', 
            values=['sentiment_score', 'relevance_score', 'volatility_score', 
                    'echo_score', 'content_depth_score']
        )
        
        # Flatten source column names
        source_pivot.columns = [f"{col[0]}_{col[1]}" for col in source_pivot.columns]
        source_pivot = source_pivot.reset_index()
        
        # Merge aggregates
        daily_features = daily_agg.merge(source_pivot, on='date', how='left')
        daily_features = daily_features.fillna(0)
        
        logging.info(f"✅ Created daily aggregates for {len(daily_features)} days")
        logging.info(f"   Features per day: {len(daily_features.columns) - 1}")
        
        return daily_features
    
    def create_price_labels(self, price_df, prediction_horizons=[1, 3, 7]):
        """Create price movement labels for different prediction horizons."""
        logging.info("💰 Creating price movement labels...")
        
        # Convert to datetime and sort
        price_df = price_df.copy()
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])
        price_df = price_df.sort_values('datetime').reset_index(drop=True)
        
        # Get daily prices (using closing price of each day)
        daily_prices = price_df.groupby('date').agg({
            'price_usd': 'last',  # Last price of the day
            'volume_24h': 'last',
            'market_cap': 'last',
            'datetime': 'last'
        }).reset_index()
        
        daily_prices = daily_prices.sort_values('date').reset_index(drop=True)
        
        # Create price labels for different horizons
        for horizon in prediction_horizons:
            # Price change percentage
            daily_prices[f'price_change_{horizon}d'] = (
                daily_prices['price_usd'].shift(-horizon) / daily_prices['price_usd'] - 1
            ) * 100
            
            # Binary direction (up/down)
            daily_prices[f'price_direction_{horizon}d'] = (
                daily_prices[f'price_change_{horizon}d'] > 0
            ).astype(int)
            
            # Volatility categories
            volatility_abs = daily_prices[f'price_change_{horizon}d'].abs()
            daily_prices[f'price_volatility_{horizon}d'] = pd.cut(
                volatility_abs,
                bins=[-np.inf, 2, 5, 10, np.inf],
                labels=[0, 1, 2, 3]  # Low, Medium, High, Extreme
            )
            # Handle NaN values before converting to int
            daily_prices[f'price_volatility_{horizon}d'] = daily_prices[f'price_volatility_{horizon}d'].cat.add_categories([-1]).fillna(-1).astype(int)
        
        # Add technical indicators
        daily_prices['price_ma_7'] = daily_prices['price_usd'].rolling(7).mean()
        daily_prices['price_ma_30'] = daily_prices['price_usd'].rolling(30).mean()
        daily_prices['price_volatility_7d'] = daily_prices['price_usd'].pct_change().rolling(7).std() * 100
        daily_prices['price_rsi_14'] = self.calculate_rsi(daily_prices['price_usd'], 14)
        
        logging.info(f"✅ Created price labels for {len(daily_prices)} days")
        logging.info(f"   Prediction horizons: {prediction_horizons} days")
        
        return daily_prices
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.round(2)
    
    def create_ml_dataset(self, daily_features, daily_prices, min_data_points=7):
        """Create final ML dataset by merging features and labels."""
        logging.info("🔗 Creating final ML dataset...")
        
        # Convert date columns to same type for merging
        daily_features['date'] = pd.to_datetime(daily_features['date']).dt.date
        daily_prices['date'] = pd.to_datetime(daily_prices['date']).dt.date
        
        logging.info(f"   Features data: {len(daily_features)} days ({daily_features['date'].min()} to {daily_features['date'].max()})")
        logging.info(f"   Price data: {len(daily_prices)} days ({daily_prices['date'].min()} to {daily_prices['date'].max()})")
        
        # Merge features with price labels
        ml_dataset = daily_features.merge(daily_prices, on='date', how='inner')
        logging.info(f"   After merge: {len(ml_dataset)} samples")
        
        # Remove rows with insufficient future data (only remove if ALL are NaN)
        before_drop = len(ml_dataset)
        ml_dataset = ml_dataset.dropna(subset=['price_change_1d'], how='all')
        after_drop = len(ml_dataset)
        logging.info(f"   After removing NaN targets: {after_drop} samples (dropped {before_drop - after_drop})")
        
        # Sort by date
        ml_dataset = ml_dataset.sort_values('date').reset_index(drop=True)
        
        # Feature engineering - add rolling features
        feature_cols = [col for col in ml_dataset.columns if col.endswith(('_mean', '_std', '_max', '_sum', '_count'))]
        
        for col in feature_cols:
            # 3-day rolling averages
            ml_dataset[f'{col}_3d_avg'] = ml_dataset[col].rolling(3).mean()
            # 7-day rolling averages  
            ml_dataset[f'{col}_7d_avg'] = ml_dataset[col].rolling(7).mean()
        
        # Remove rows with insufficient historical data
        ml_dataset = ml_dataset.iloc[min_data_points:].reset_index(drop=True)
        
        logging.info(f"✅ Final ML dataset created:")
        logging.info(f"   Total samples: {len(ml_dataset)}")
        logging.info(f"   Features: {len([col for col in ml_dataset.columns if col not in ['date', 'datetime', 'price_usd', 'volume_24h', 'market_cap']])}")
        logging.info(f"   Date range: {ml_dataset['date'].min()} to {ml_dataset['date'].max()}")
        
        return ml_dataset
    
    def analyze_dataset_quality(self, ml_dataset):
        """Analyze the quality and characteristics of the ML dataset."""
        logging.info("🔍 Analyzing dataset quality...")
        
        # Basic statistics
        total_samples = len(ml_dataset)
        feature_cols = [col for col in ml_dataset.columns if col not in [
            'date', 'datetime', 'price_usd', 'volume_24h', 'market_cap',
            'price_change_1d', 'price_change_3d', 'price_change_7d',
            'price_direction_1d', 'price_direction_3d', 'price_direction_7d',
            'price_volatility_1d', 'price_volatility_3d', 'price_volatility_7d'
        ]]
        
        target_cols = ['price_direction_1d', 'price_direction_3d', 'price_direction_7d']
        
        # Missing values
        missing_data = ml_dataset[feature_cols].isnull().sum()
        missing_pct = (missing_data / total_samples * 100).round(2)
        
        # Target distribution
        target_distribution = {}
        for target in target_cols:
            target_distribution[target] = ml_dataset[target].value_counts(normalize=True).round(3)
        
        # Feature correlations with targets
        correlations = {}
        for target in target_cols:
            correlations[target] = ml_dataset[feature_cols].corrwith(ml_dataset[target]).abs().sort_values(ascending=False).head(10)
        
        quality_report = {
            'total_samples': total_samples,
            'feature_count': len(feature_cols),
            'date_range': f"{ml_dataset['date'].min()} to {ml_dataset['date'].max()}",
            'missing_data': missing_data[missing_data > 0].to_dict(),
            'target_distribution': target_distribution,
            'top_correlations': {k: v.to_dict() for k, v in correlations.items()}
        }
        
        logging.info(f"📊 Dataset Quality Report:")
        logging.info(f"   Total samples: {total_samples}")
        logging.info(f"   Feature count: {len(feature_cols)}")
        logging.info(f"   Missing data features: {len(missing_data[missing_data > 0])}")
        
        for target, dist in target_distribution.items():
            logging.info(f"   {target} distribution: Up={dist.get(1, 0):.1%}, Down={dist.get(0, 0):.1%}")
        
        return quality_report
    
    def save_ml_dataset(self, ml_dataset, filepath='data/ml_dataset.csv'):
        """Save the ML dataset to CSV and database."""
        logging.info(f"💾 Saving ML dataset...")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        ml_dataset.to_csv(filepath, index=False)
        logging.info(f"✅ Saved to CSV: {filepath}")
        
        # Save to database
        conn = sqlite3.connect(self.db.db_path)
        try:
            ml_dataset.to_sql('ml_dataset', conn, if_exists='replace', index=False)
            logging.info(f"✅ Saved to database table: ml_dataset")
        finally:
            conn.close()
        
        # Save feature info
        feature_info = {
            'total_samples': len(ml_dataset),
            'feature_columns': [col for col in ml_dataset.columns if col not in [
                'date', 'datetime', 'price_usd', 'volume_24h', 'market_cap'
            ]],
            'target_columns': [
                'price_change_1d', 'price_change_3d', 'price_change_7d',
                'price_direction_1d', 'price_direction_3d', 'price_direction_7d',
                'price_volatility_1d', 'price_volatility_3d', 'price_volatility_7d'
            ],
            'date_range': {
                'start': str(ml_dataset['date'].min()),
                'end': str(ml_dataset['date'].max())
            },
            'created_at': datetime.now().isoformat()
        }
        
        info_filepath = filepath.replace('.csv', '_info.json')
        with open(info_filepath, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logging.info(f"✅ Saved feature info: {info_filepath}")
        
        return filepath, info_filepath

def main():
    """Main function to create ML dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ML dataset from text metrics and price data')
    parser.add_argument('--output', default='data/ml_dataset.csv', help='Output CSV file path')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze data coverage')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 7], help='Prediction horizons in days')
    
    args = parser.parse_args()
    
    creator = MLDatasetCreator()
    
    # Analyze data coverage
    metrics_df, price_df = creator.analyze_data_coverage()
    
    if args.analyze_only:
        return
    
    # Create daily aggregates
    daily_features = creator.create_daily_aggregates(metrics_df)
    
    # Create price labels
    daily_prices = creator.create_price_labels(price_df, args.horizons)
    
    # Create ML dataset
    ml_dataset = creator.create_ml_dataset(daily_features, daily_prices)
    
    # Analyze quality
    quality_report = creator.analyze_dataset_quality(ml_dataset)
    
    # Save dataset
    csv_path, info_path = creator.save_ml_dataset(ml_dataset, args.output)
    
    logging.info(f"🎉 ML dataset creation complete!")
    logging.info(f"   Dataset: {csv_path}")
    logging.info(f"   Info: {info_path}")
    logging.info(f"   Ready for ML model training!")

if __name__ == "__main__":
    main()