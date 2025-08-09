#!/usr/bin/env python3
# src/simplified_ml_dataset.py

"""
Simplified ML Dataset Creator for CryptoPulse
Creates a focused, high-quality dataset with 10-15 core features.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simplified_ml_dataset.log'),
        logging.StreamHandler()
    ]
)

class SimplifiedMLDataset:
    def __init__(self, db_path='db/cryptopulse.db'):
        self.db_path = db_path
        logging.info("🎯 Simplified ML Dataset Creator initialized")
    
    def extract_core_features(self):
        """Extract only the core working features from our data."""
        logging.info("📊 Extracting core working features...")
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get text metrics with source info - focusing on working metrics only
            query = """
                SELECT 
                    DATE(datetime(r.created_utc, 'unixepoch')) as date,
                    'reddit' as source,
                    m.relevance_score,
                    m.volatility_score,
                    m.echo_score,
                    m.content_depth_score,
                    r.score as engagement,
                    r.num_comments,
                    LENGTH(r.content) as content_length
                FROM reddit_posts r
                JOIN modern_text_metrics m ON r.id = m.id
                WHERE datetime(r.created_utc, 'unixepoch') >= '2025-02-01' 
                  AND datetime(r.created_utc, 'unixepoch') <= '2025-07-31'
                
                UNION ALL
                
                SELECT 
                    DATE(datetime(t.created_at, 'unixepoch')) as date,
                    'twitter' as source,
                    m.relevance_score,
                    m.volatility_score,
                    m.echo_score,
                    m.content_depth_score,
                    t.likes as engagement,
                    t.retweets as num_comments,
                    LENGTH(t.content) as content_length
                FROM twitter_posts t
                JOIN modern_text_metrics m ON t.id = m.id
                WHERE datetime(t.created_at, 'unixepoch') >= '2025-02-01' 
                  AND datetime(t.created_at, 'unixepoch') <= '2025-07-31'
                
                UNION ALL
                
                SELECT 
                    DATE(datetime(n.published_at, 'unixepoch')) as date,
                    'news' as source,
                    m.relevance_score,
                    m.volatility_score,
                    m.echo_score,
                    m.content_depth_score,
                    0 as engagement,
                    0 as num_comments,
                    LENGTH(n.content) as content_length
                FROM news_articles n
                JOIN modern_text_metrics m ON n.id = m.id
                WHERE datetime(n.published_at, 'unixepoch') >= '2025-02-01' 
                  AND datetime(n.published_at, 'unixepoch') <= '2025-07-31'
                
                ORDER BY date, source
            """
            
            text_df = pd.read_sql_query(query, conn)
            
            # Get price data
            price_query = """
                SELECT 
                    DATE(datetime(timestamp, 'unixepoch')) as date,
                    price_usd,
                    volume_24h,
                    market_cap
                FROM eth_prices 
                WHERE datetime(timestamp, 'unixepoch') >= '2025-02-01' 
                  AND datetime(timestamp, 'unixepoch') <= '2025-07-31'
                ORDER BY timestamp
            """
            
            price_df = pd.read_sql_query(price_query, conn)
            
            logging.info(f"✅ Extracted {len(text_df)} text entries and {len(price_df)} price points")
            
            return text_df, price_df
            
        finally:
            conn.close()
    
    def create_daily_features(self, text_df):
        """Create focused daily features from text data."""
        logging.info("🔧 Creating focused daily features...")
        
        # Convert date for grouping
        text_df['date'] = pd.to_datetime(text_df['date'])
        
        # Create daily aggregates - focused on working metrics
        daily_features = text_df.groupby('date').agg({
            # Core sentiment metrics (working ones)
            'relevance_score': ['mean', 'max', 'std'],
            'volatility_score': ['mean', 'max', 'sum'],  # Sum for daily volatility signal
            'echo_score': ['mean', 'max'],
            'content_depth_score': ['mean', 'max'],
            
            # Volume and engagement
            'engagement': ['sum', 'mean'],
            'num_comments': ['sum', 'mean'],
            'content_length': ['mean', 'max'],
            
            # Count of entries per day
            'source': 'count'
        }).round(4)
        
        # Flatten column names
        daily_features.columns = ['_'.join(col).strip() for col in daily_features.columns]
        daily_features = daily_features.reset_index()
        
        # Rename count column
        daily_features = daily_features.rename(columns={'source_count': 'daily_entry_count'})
        
        # Add source-specific features
        source_features = text_df.groupby(['date', 'source']).agg({
            'relevance_score': 'mean',
            'volatility_score': 'mean',
            'echo_score': 'mean',
            'content_depth_score': 'mean'
        }).round(4)
        
        # Pivot source features
        source_pivot = source_features.reset_index().pivot(
            index='date', columns='source', 
            values=['relevance_score', 'volatility_score', 'echo_score', 'content_depth_score']
        )
        source_pivot.columns = ['_'.join(col).strip() for col in source_pivot.columns]
        source_pivot = source_pivot.reset_index().fillna(0)
        
        # Merge source features
        daily_features = daily_features.merge(source_pivot, on='date', how='left')
        daily_features = daily_features.fillna(0)
        
        logging.info(f"✅ Created {len(daily_features)} daily feature vectors with {len(daily_features.columns)-1} features")
        
        return daily_features
    
    def create_price_targets(self, price_df, horizons=[1, 3, 7]):
        """Create simplified price targets."""
        logging.info("💰 Creating price targets...")
        
        # Convert and sort
        price_df = price_df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # Get daily closing prices
        daily_prices = price_df.groupby('date').agg({
            'price_usd': 'last',
            'volume_24h': 'last',
            'market_cap': 'last'
        }).reset_index().sort_values('date').reset_index(drop=True)
        
        # Create price change targets
        for h in horizons:
            daily_prices[f'price_change_{h}d'] = (
                daily_prices['price_usd'].shift(-h) / daily_prices['price_usd'] - 1
            ) * 100
            
            daily_prices[f'direction_{h}d'] = (
                daily_prices[f'price_change_{h}d'] > 0
            ).astype(int)
        
        # Add simple technical indicators
        daily_prices['price_ma_7'] = daily_prices['price_usd'].rolling(7).mean()
        daily_prices['price_volatility'] = daily_prices['price_usd'].pct_change().rolling(7).std() * 100
        daily_prices['volume_ma_7'] = daily_prices['volume_24h'].rolling(7).mean()
        
        logging.info(f"✅ Created price targets for {len(daily_prices)} days")
        
        return daily_prices
    
    def select_best_features(self, features_df, target_col='direction_1d', max_features=12):
        """Select the best features using statistical methods."""
        logging.info(f"🎯 Selecting top {max_features} features...")
        
        # Get feature columns (exclude date and targets)
        feature_cols = [col for col in features_df.columns if col not in [
            'date', 'price_usd', 'volume_24h', 'market_cap', 'price_ma_7', 
            'price_volatility', 'volume_ma_7', 'price_change_1d', 'price_change_3d', 
            'price_change_7d', 'direction_1d', 'direction_3d', 'direction_7d'
        ]]
        
        # Prepare data
        X = features_df[feature_cols].fillna(0)
        y = features_df[target_col].fillna(0)
        
        # Remove any remaining invalid data
        valid_idx = ~(np.isnan(y) | np.isinf(X).any(axis=1))
        X = X[valid_idx]
        y = y[valid_idx]
        
        logging.info(f"   Using {len(X)} samples for feature selection")
        
        # Method 1: F-statistics
        f_selector = SelectKBest(f_classif, k=min(max_features, len(feature_cols)))
        f_selector.fit(X, y)
        f_scores = pd.DataFrame({
            'feature': feature_cols,
            'f_score': f_selector.scores_,
            'f_pvalue': f_selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        # Method 2: Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Combine scores (weighted average)
        f_scores_norm = (f_scores['f_score'] - f_scores['f_score'].min()) / (f_scores['f_score'].max() - f_scores['f_score'].min())
        mi_scores_norm = (mi_df['mi_score'] - mi_df['mi_score'].min()) / (mi_df['mi_score'].max() - mi_df['mi_score'].min())
        
        combined_scores = pd.DataFrame({
            'feature': feature_cols,
            'combined_score': 0.7 * f_scores_norm + 0.3 * mi_scores_norm
        }).sort_values('combined_score', ascending=False)
        
        # Select top features
        selected_features = combined_scores.head(max_features)['feature'].tolist()
        
        logging.info(f"✅ Selected top {len(selected_features)} features:")
        for i, feature in enumerate(selected_features[:10]):  # Show top 10
            score = combined_scores.loc[combined_scores['feature']==feature, 'combined_score'].iloc[0]
            logging.info(f"   {i+1}. {feature} (score: {score:.3f})")
        
        return selected_features, combined_scores
    
    def create_simplified_dataset(self, max_features=12):
        """Create the complete simplified dataset."""
        logging.info("🚀 Creating simplified ML dataset...")
        
        # Extract core data
        text_df, price_df = self.extract_core_features()
        
        # Create features
        daily_features = self.create_daily_features(text_df)
        daily_prices = self.create_price_targets(price_df)
        
        # Merge features and prices
        ml_dataset = daily_features.merge(daily_prices, on='date', how='inner')
        
        # Remove rows without valid targets
        ml_dataset = ml_dataset.dropna(subset=['direction_1d'])
        
        logging.info(f"   Merged dataset: {len(ml_dataset)} samples")
        
        # Select best features
        selected_features, feature_scores = self.select_best_features(ml_dataset, max_features=max_features)
        
        # Create final dataset with selected features
        final_columns = ['date'] + selected_features + [
            'price_usd', 'price_change_1d', 'price_change_3d', 'price_change_7d',
            'direction_1d', 'direction_3d', 'direction_7d', 'price_ma_7', 'price_volatility'
        ]
        
        final_dataset = ml_dataset[final_columns].copy()
        
        # Sort by date
        final_dataset = final_dataset.sort_values('date').reset_index(drop=True)
        
        logging.info(f"✅ Final simplified dataset:")
        logging.info(f"   Samples: {len(final_dataset)}")
        logging.info(f"   Features: {len(selected_features)}")
        logging.info(f"   Date range: {final_dataset['date'].min().date()} to {final_dataset['date'].max().date()}")
        
        return final_dataset, feature_scores
    
    def validate_simplified_dataset(self, dataset):
        """Validate the simplified dataset quality."""
        logging.info("🔍 Validating simplified dataset...")
        
        # Basic checks
        logging.info(f"Dataset shape: {dataset.shape}")
        
        # Feature columns
        feature_cols = [col for col in dataset.columns if col not in [
            'date', 'price_usd', 'price_change_1d', 'price_change_3d', 'price_change_7d',
            'direction_1d', 'direction_3d', 'direction_7d', 'price_ma_7', 'price_volatility'
        ]]
        
        # Check for missing values
        missing_data = dataset[feature_cols].isnull().sum()
        if missing_data.sum() > 0:
            logging.warning(f"⚠️  Missing values found: {missing_data[missing_data > 0].to_dict()}")
        else:
            logging.info("✅ No missing values in features")
        
        # Check target distribution
        for target in ['direction_1d', 'direction_3d', 'direction_7d']:
            if target in dataset.columns:
                dist = dataset[target].value_counts(normalize=True)
                up_pct = dist.get(1, 0) * 100
                down_pct = dist.get(0, 0) * 100
                logging.info(f"   {target}: Up={up_pct:.1f}%, Down={down_pct:.1f}%")
        
        # Feature-to-sample ratio
        ratio = len(dataset) / len(feature_cols)
        logging.info(f"Feature-to-sample ratio: {ratio:.1f}")
        
        if ratio >= 10:
            logging.info("✅ Good feature-to-sample ratio")
        elif ratio >= 5:
            logging.info("⚠️  Moderate feature-to-sample ratio")
        else:
            logging.warning("❌ Poor feature-to-sample ratio")
        
        # Check feature correlations
        corr_matrix = dataset[feature_cols].corr()
        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
        high_corr_count = high_corr.sum().sum() // 2
        
        if high_corr_count > 0:
            logging.warning(f"⚠️  {high_corr_count} high correlations (>0.8) between features")
        else:
            logging.info("✅ No high correlations between features")
        
        return {
            'samples': len(dataset),
            'features': len(feature_cols),
            'ratio': ratio,
            'missing_values': missing_data.sum(),
            'high_correlations': high_corr_count
        }
    
    def save_dataset(self, dataset, feature_scores, filepath='data/simplified_ml_dataset.csv'):
        """Save the simplified dataset."""
        logging.info("💾 Saving simplified dataset...")
        
        # Save main dataset
        dataset.to_csv(filepath, index=False)
        logging.info(f"✅ Saved dataset: {filepath}")
        
        # Save feature importance scores
        scores_path = filepath.replace('.csv', '_feature_scores.csv')
        feature_scores.to_csv(scores_path, index=False)
        logging.info(f"✅ Saved feature scores: {scores_path}")
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'samples': len(dataset),
                'features': len([col for col in dataset.columns if col not in [
                    'date', 'price_usd', 'price_change_1d', 'price_change_3d', 'price_change_7d',
                    'direction_1d', 'direction_3d', 'direction_7d', 'price_ma_7', 'price_volatility'
                ]]),
                'date_range': {
                    'start': str(dataset['date'].min().date()),
                    'end': str(dataset['date'].max().date())
                }
            },
            'feature_columns': [col for col in dataset.columns if col not in [
                'date', 'price_usd', 'price_change_1d', 'price_change_3d', 'price_change_7d',
                'direction_1d', 'direction_3d', 'direction_7d', 'price_ma_7', 'price_volatility'
            ]],
            'target_columns': ['direction_1d', 'direction_3d', 'direction_7d'],
            'price_columns': ['price_change_1d', 'price_change_3d', 'price_change_7d'],
            'technical_indicators': ['price_ma_7', 'price_volatility'],
            'created_at': datetime.now().isoformat()
        }
        
        info_path = filepath.replace('.csv', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"✅ Saved metadata: {info_path}")
        
        return filepath, scores_path, info_path

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create simplified ML dataset')
    parser.add_argument('--output', default='data/simplified_ml_dataset.csv', help='Output path')
    parser.add_argument('--max-features', type=int, default=12, help='Maximum number of features')
    
    args = parser.parse_args()
    
    # Create simplified dataset
    creator = SimplifiedMLDataset()
    
    dataset, feature_scores = creator.create_simplified_dataset(max_features=args.max_features)
    
    # Validate
    validation_results = creator.validate_simplified_dataset(dataset)
    
    # Save
    dataset_path, scores_path, info_path = creator.save_dataset(dataset, feature_scores, args.output)
    
    logging.info(f"\n🎉 Simplified ML dataset created successfully!")
    logging.info(f"   Dataset: {dataset_path}")
    logging.info(f"   Scores: {scores_path}")
    logging.info(f"   Info: {info_path}")
    logging.info(f"   Ready for ML training with proper feature-to-sample ratio!")

if __name__ == "__main__":
    main()