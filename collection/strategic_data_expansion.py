#!/usr/bin/env python3
"""
Strategic Data Expansion for Enhanced ML Modeling

Current Situation Analysis:
- Reddit: 10,081 posts (2015-12-08 to 2025-07-29) ✅ Good coverage
- Twitter: 1,731 posts (limited coverage) ⚠️ NEEDS EXPANSION  
- News: 4,147 articles (no date info visible) ⚠️ NEEDS VALIDATION
- ML Dataset: Only 178 samples for training ❌ INSUFFICIENT

Target: Expand to 1000+ quality samples for robust modeling

Strategy:
1. Define optimal timeframe (2022-2025 for crypto maturity)
2. Intensive Twitter collection from ALL crypto influencers
3. Backfill Reddit data gaps in target period
4. Validate and expand news coverage
5. Ensure balanced daily coverage for feature engineering
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategic_data_expansion.log'),
        logging.StreamHandler()
    ]
)

class StrategicDataExpansion:
    def __init__(self, db_path="db/cryptopulse.db"):
        self.db_path = db_path
        self.target_start = datetime(2022, 1, 1)  # Crypto maturity period
        self.target_end = datetime(2025, 8, 1)    # Current
        
    def analyze_current_coverage(self):
        """Analyze current data coverage and identify gaps"""
        print("🔍 ANALYZING CURRENT DATA COVERAGE")
        print("="*50)
        
        conn = sqlite3.connect(self.db_path)
        
        # Reddit analysis
        reddit_query = """
        SELECT 
            COUNT(*) as total_posts,
            COUNT(CASE WHEN datetime(created_utc, 'unixepoch') >= '2022-01-01' THEN 1 END) as posts_2022_plus,
            MIN(datetime(created_utc, 'unixepoch')) as earliest,
            MAX(datetime(created_utc, 'unixepoch')) as latest,
            COUNT(DISTINCT date(datetime(created_utc, 'unixepoch'))) as unique_days
        FROM reddit_posts 
        WHERE created_utc IS NOT NULL
        """
        
        reddit_stats = pd.read_sql_query(reddit_query, conn).iloc[0]
        
        # Twitter analysis  
        twitter_query = """
        SELECT 
            COUNT(*) as total_posts,
            COUNT(CASE WHEN created_at >= '2022-01-01' THEN 1 END) as posts_2022_plus,
            MIN(created_at) as earliest,
            MAX(created_at) as latest,
            COUNT(DISTINCT date(created_at)) as unique_days
        FROM twitter_posts 
        WHERE created_at IS NOT NULL
        """
        
        twitter_stats = pd.read_sql_query(twitter_query, conn).iloc[0]
        
        # News analysis
        news_query = """
        SELECT 
            COUNT(*) as total_articles,
            COUNT(CASE WHEN published_at >= '2022-01-01' THEN 1 END) as articles_2022_plus,
            MIN(published_at) as earliest,
            MAX(published_at) as latest,
            COUNT(DISTINCT date(published_at)) as unique_days
        FROM news_articles 
        WHERE published_at IS NOT NULL
        """
        
        news_stats = pd.read_sql_query(news_query, conn).iloc[0]
        
        # Daily coverage analysis for 2022-2025
        daily_coverage_query = """
        SELECT 
            date(datetime(created_utc, 'unixepoch')) as date,
            COUNT(*) as reddit_posts,
            0 as twitter_posts,
            0 as news_articles
        FROM reddit_posts 
        WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01'
        GROUP BY date(datetime(created_utc, 'unixepoch'))
        
        UNION ALL
        
        SELECT 
            date(created_at) as date,
            0 as reddit_posts,
            COUNT(*) as twitter_posts,
            0 as news_articles
        FROM twitter_posts 
        WHERE created_at >= '2022-01-01'
        GROUP BY date(created_at)
        
        UNION ALL
        
        SELECT 
            date(published_at) as date,
            0 as reddit_posts,
            0 as twitter_posts,
            COUNT(*) as news_articles
        FROM news_articles 
        WHERE published_at >= '2022-01-01'
        GROUP BY date(published_at)
        """
        
        daily_data = pd.read_sql_query(daily_coverage_query, conn)
        daily_summary = daily_data.groupby('date').sum()
        
        conn.close()
        
        # Print analysis
        print("📊 CURRENT DATA INVENTORY:")
        print(f"   🟩 Reddit: {reddit_stats['total_posts']:,} posts ({reddit_stats['posts_2022_plus']:,} since 2022)")
        print(f"      📅 Coverage: {reddit_stats['earliest']} to {reddit_stats['latest']}")
        print(f"      📈 Unique days: {reddit_stats['unique_days']:,}")
        
        print(f"   🟨 Twitter: {twitter_stats['total_posts']:,} posts ({twitter_stats['posts_2022_plus']:,} since 2022)")
        print(f"      📅 Coverage: {twitter_stats['earliest']} to {twitter_stats['latest']}")
        print(f"      📈 Unique days: {twitter_stats['unique_days']:,}")
        
        print(f"   🟦 News: {news_stats['total_articles']:,} articles ({news_stats['articles_2022_plus']:,} since 2022)")
        print(f"      📅 Coverage: {news_stats['earliest']} to {news_stats['latest']}")
        print(f"      📈 Unique days: {news_stats['unique_days']:,}")
        
        # Calculate target period coverage
        target_days = (self.target_end - self.target_start).days
        covered_days = len(daily_summary)
        coverage_percent = (covered_days / target_days) * 100
        
        print(f"\\n📈 TARGET PERIOD ANALYSIS (2022-2025):")
        print(f"   🎯 Target days: {target_days:,}")
        print(f"   ✅ Days with data: {covered_days:,}")
        print(f"   📊 Coverage: {coverage_percent:.1f}%")
        
        # Identify gaps
        gaps = self.identify_data_gaps(daily_summary)
        
        return {
            'reddit_stats': reddit_stats,
            'twitter_stats': twitter_stats, 
            'news_stats': news_stats,
            'daily_coverage': daily_summary,
            'coverage_percent': coverage_percent,
            'gaps': gaps
        }
    
    def identify_data_gaps(self, daily_summary):
        """Identify significant data gaps that need filling"""
        print("\\n🔍 IDENTIFYING DATA GAPS:")
        
        # Convert index to datetime
        daily_summary.index = pd.to_datetime(daily_summary.index)
        
        # Create complete date range
        date_range = pd.date_range(start=self.target_start, end=self.target_end, freq='D')
        full_coverage = pd.DataFrame(index=date_range)
        
        # Merge with actual data
        merged = full_coverage.join(daily_summary, how='left').fillna(0)
        
        # Find gaps (days with no data from any source)
        no_data_days = merged[(merged['reddit_posts'] == 0) & 
                             (merged['twitter_posts'] == 0) & 
                             (merged['news_articles'] == 0)]
        
        # Find low data days (less than 5 total posts)
        low_data_days = merged[merged.sum(axis=1) < 5]
        
        print(f"   ❌ Days with NO data: {len(no_data_days):,}")
        print(f"   ⚠️ Days with LOW data (<5 posts): {len(low_data_days):,}")
        
        if len(no_data_days) > 0:
            print(f"   📅 Latest no-data gap: {no_data_days.index[-10:].strftime('%Y-%m-%d').tolist()}")
        
        return {
            'no_data_days': no_data_days.index.tolist(),
            'low_data_days': low_data_days.index.tolist(),
            'total_gaps': len(no_data_days) + len(low_data_days)
        }
    
    def design_collection_strategy(self, analysis):
        """Design comprehensive data collection strategy"""
        print("\\n🚀 DESIGNING COLLECTION STRATEGY")
        print("="*50)
        
        strategy = {
            'priority_actions': [],
            'twitter_strategy': {},
            'reddit_strategy': {},
            'news_strategy': {},
            'timeline': {}
        }
        
        # Twitter strategy (highest impact)
        if analysis['twitter_stats']['posts_2022_plus'] < 10000:
            strategy['priority_actions'].append("CRITICAL: Massive Twitter expansion needed")
            strategy['twitter_strategy'] = {
                'target_posts': 25000,
                'current_posts': analysis['twitter_stats']['posts_2022_plus'],
                'gap': 25000 - analysis['twitter_stats']['posts_2022_plus'],
                'method': 'Comprehensive influencer feed scraping',
                'timeframe': '2022-01-01 to 2025-08-01',
                'influencer_categories': [
                    'Ethereum Founders & Core Devs',
                    'DeFi Protocol Leaders', 
                    'Crypto VCs & Investors',
                    'Trading & Analysis Experts',
                    'Crypto Media & Journalists',
                    'CEX & Exchange Leaders',
                    'NFT & Web3 Influencers'
                ]
            }
        
        # Reddit strategy  
        reddit_target_days = len(analysis['gaps']['no_data_days']) + len(analysis['gaps']['low_data_days'])
        if reddit_target_days > 100:
            strategy['reddit_strategy'] = {
                'target_days': reddit_target_days,
                'method': 'Historical backfill for gap days',
                'subreddits': ['ethereum', 'ethtrader', 'ethfinance', 'cryptocurrency', 'cryptomarkets'],
                'posts_per_day_target': 20
            }
        
        # News strategy
        if analysis['news_stats']['articles_2022_plus'] < 5000:
            strategy['news_strategy'] = {
                'target_articles': 8000,
                'current_articles': analysis['news_stats']['articles_2022_plus'],
                'gap': 8000 - analysis['news_stats']['articles_2022_plus'],
                'method': 'RSS feed historical collection',
                'sources': ['coindesk', 'cointelegraph', 'decrypt', 'theblock', 'blockworks']
            }
        
        # Timeline
        strategy['timeline'] = {
            'phase1': 'Twitter influencer expansion (1-2 days)',
            'phase2': 'Reddit gap filling (1 day)', 
            'phase3': 'News historical collection (1 day)',
            'phase4': 'Data validation and ML dataset regeneration (0.5 days)'
        }
        
        print("🎯 STRATEGIC PRIORITIES:")
        for action in strategy['priority_actions']:
            print(f"   ⚡ {action}")
        
        print("\\n📱 TWITTER EXPANSION PLAN:")
        tw_strat = strategy['twitter_strategy']
        print(f"   📊 Target: {tw_strat['target_posts']:,} posts (gap: {tw_strat['gap']:,})")
        print(f"   🎭 Method: {tw_strat['method']}")
        print(f"   📅 Period: {tw_strat['timeframe']}")
        
        print("\\n🔶 REDDIT BACKFILL PLAN:")
        if 'reddit_strategy' in strategy:
            rd_strat = strategy['reddit_strategy']
            print(f"   📊 Target days to fill: {rd_strat['target_days']:,}")
            print(f"   🎭 Method: {rd_strat['method']}")
        else:
            print("   ✅ Reddit coverage adequate")
        
        print("\\n📰 NEWS EXPANSION PLAN:")
        if 'news_strategy' in strategy:
            news_strat = strategy['news_strategy']
            print(f"   📊 Target: {news_strat['target_articles']:,} articles (gap: {news_strat['gap']:,})")
            print(f"   🎭 Method: {news_strat['method']}")
        else:
            print("   ✅ News coverage adequate")
        
        return strategy
    
    def estimate_ml_improvement(self, strategy):
        """Estimate ML modeling improvement from expanded data"""
        print("\\n🤖 EXPECTED ML MODELING IMPROVEMENTS")
        print("="*50)
        
        current_samples = 178  # From our current ML dataset
        
        # Estimate new samples from expanded data
        twitter_boost = strategy['twitter_strategy']['gap'] // 100  # ~1 sample per 100 tweets
        reddit_boost = strategy.get('reddit_strategy', {}).get('target_days', 0) * 0.8  # ~0.8 samples per filled day
        news_boost = strategy.get('news_strategy', {}).get('gap', 0) // 50  # ~1 sample per 50 articles
        
        estimated_new_samples = twitter_boost + reddit_boost + news_boost
        total_estimated_samples = current_samples + estimated_new_samples
        
        print(f"📊 SAMPLE SIZE PROJECTION:")
        print(f"   📈 Current ML samples: {current_samples}")
        print(f"   🐦 Twitter contribution: +{twitter_boost}")
        print(f"   🔶 Reddit contribution: +{reddit_boost}")  
        print(f"   📰 News contribution: +{news_boost}")
        print(f"   🎯 Estimated total: {total_estimated_samples} samples")
        print(f"   📈 Improvement: {(estimated_new_samples/current_samples)*100:.0f}% increase")
        
        print(f"\\n🎯 EXPECTED BENEFITS:")
        print(f"   ✅ Better cross-validation reliability")
        print(f"   ✅ Reduced overfitting risk") 
        print(f"   ✅ More robust feature importance")
        print(f"   ✅ Improved generalization to new data")
        print(f"   ✅ Statistical significance in improvements")

def main():
    print("🚀 STRATEGIC DATA EXPANSION ANALYSIS")
    print("="*60)
    
    expander = StrategicDataExpansion()
    
    # Analyze current situation
    analysis = expander.analyze_current_coverage()
    
    # Design collection strategy
    strategy = expander.design_collection_strategy(analysis)
    
    # Estimate improvements
    expander.estimate_ml_improvement(strategy)
    
    print("\\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("1. 🐦 Implement comprehensive Twitter influencer collection")
    print("2. 🔶 Execute Reddit historical backfill for gap periods") 
    print("3. 📰 Expand news coverage with RSS historical data")
    print("4. 🤖 Regenerate ML dataset with expanded data")
    print("5. 📊 Re-run hypothesis validation with robust dataset")
    print("="*60)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    main()