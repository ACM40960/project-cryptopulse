#!/usr/bin/env python3
# data_distribution_analysis.py

"""
Comprehensive data distribution analysis and gap identification
for CryptoPulse dataset improvement.
"""
import sys
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append('src')
from database import CryptoPulseDB

def analyze_data_distribution():
    """Analyze current data distribution and identify gaps."""
    print("📊 CryptoPulse Data Distribution Analysis")
    print("=" * 60)
    
    db = CryptoPulseDB()
    conn = sqlite3.connect(db.db_path)
    
    # Get monthly distribution for all data types
    queries = {
        'reddit': """
            SELECT strftime('%Y-%m', datetime(created_utc, 'unixepoch')) as month,
                   COUNT(*) as count
            FROM reddit_posts GROUP BY month ORDER BY month
        """,
        'twitter': """
            SELECT strftime('%Y-%m', datetime(created_at, 'unixepoch')) as month,
                   COUNT(*) as count  
            FROM twitter_posts GROUP BY month ORDER BY month
        """,
        'news': """
            SELECT strftime('%Y-%m', datetime(published_at, 'unixepoch')) as month,
                   COUNT(*) as count
            FROM news_articles GROUP BY month ORDER BY month
        """,
        'prices': """
            SELECT strftime('%Y-%m', datetime(timestamp, 'unixepoch')) as month,
                   COUNT(*) as count
            FROM eth_prices GROUP BY month ORDER BY month
        """
    }
    
    data_by_type = {}
    for data_type, query in queries.items():
        df = pd.read_sql_query(query, conn)
        data_by_type[data_type] = df
        print(f"\n{data_type.upper()} DATA:")
        print(f"Total months with data: {len(df)}")
        print(f"Date range: {df['month'].min()} to {df['month'].max()}")
        print(f"Total entries: {df['count'].sum():,}")
        
        # Find gaps (months with 0 entries)
        if len(df) > 0:
            start_date = pd.to_datetime(df['month'].min())
            end_date = pd.to_datetime(df['month'].max())
            
            # Generate all months in range
            all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
            all_months_str = [m.strftime('%Y-%m') for m in all_months]
            existing_months = set(df['month'].tolist())
            
            gaps = [m for m in all_months_str if m not in existing_months]
            if gaps:
                print(f"Months with no data: {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
                print(f"Total gap months: {len(gaps)}")
    
    conn.close()
    
    # Identify priority periods for collection
    priority_periods = identify_priority_periods(data_by_type)
    print(f"\n🎯 PRIORITY COLLECTION PERIODS:")
    print("=" * 40)
    
    for period, reasons in priority_periods.items():
        print(f"{period}: {', '.join(reasons)}")
    
    return data_by_type, priority_periods

def identify_priority_periods(data_by_type):
    """Identify periods that need more data for better distribution."""
    priority_periods = defaultdict(list)
    
    # Historical periods with low Reddit activity (2018-2020)
    reddit_df = data_by_type['reddit']
    
    for _, row in reddit_df.iterrows():
        month = row['month']
        count = row['count']
        year = int(month.split('-')[0])
        
        # Historical periods that need boosting
        if year >= 2018 and year <= 2020 and count < 20:
            priority_periods[month].append("Low Reddit activity")
        
        # Key crypto events periods
        key_periods = {
            '2017-12': "ETH ATH period", 
            '2018-01': "Crypto winter start",
            '2020-03': "COVID crash",
            '2020-12': "DeFi summer peak",
            '2021-05': "ETH 2.0 hype",
            '2021-11': "ETH ATH period",
            '2022-06': "Terra Luna collapse",
            '2022-11': "FTX collapse"
        }
        
        if month in key_periods:
            priority_periods[month].append(key_periods[month])
    
    # Periods with missing Twitter data (everything before 2025-07)
    twitter_months = set(data_by_type['twitter']['month'].tolist()) if len(data_by_type['twitter']) > 0 else set()
    reddit_months = set(data_by_type['reddit']['month'].tolist())
    
    for month in reddit_months:
        if month not in twitter_months and month >= '2020-01':
            priority_periods[month].append("Missing Twitter data")
    
    # Periods with sparse news coverage
    news_df = data_by_type['news']
    news_months = set(news_df['month'].tolist()) if len(news_df) > 0 else set()
    
    for month in reddit_months:
        if month not in news_months and month >= '2018-01':
            priority_periods[month].append("Missing news coverage")
    
    return dict(priority_periods)

def create_collection_strategy(priority_periods):
    """Create targeted collection strategy for identified gaps."""
    print(f"\n🚀 COLLECTION STRATEGY:")
    print("=" * 40)
    
    strategies = {
        "Reddit Historical Boost": {
            "method": "Enhanced subreddit mining + time-based search",
            "targets": [month for month, reasons in priority_periods.items() 
                       if "Low Reddit activity" in reasons],
            "expected_gain": "500-1000 posts per target month"
        },
        "Twitter Historical Mining": {
            "method": "Academic datasets + Web scraping + API alternatives", 
            "targets": [month for month, reasons in priority_periods.items()
                       if "Missing Twitter data" in reasons],
            "expected_gain": "200-500 tweets per target month"
        },
        "News Archive Expansion": {
            "method": "Wayback Machine + RSS archives + More sources",
            "targets": [month for month, reasons in priority_periods.items()
                       if "Missing news coverage" in reasons], 
            "expected_gain": "50-100 articles per target month"
        },
        "Key Events Deep Dive": {
            "method": "Event-specific collection from multiple platforms",
            "targets": [month for month, reasons in priority_periods.items()
                       if any("period" in r or "collapse" in r or "crash" in r or "hype" in r 
                             for r in reasons)],
            "expected_gain": "1000+ entries per key event period"
        }
    }
    
    for strategy_name, details in strategies.items():
        if details["targets"]:
            print(f"\n{strategy_name}:")
            print(f"  Method: {details['method']}")
            print(f"  Targets: {len(details['targets'])} months")
            print(f"  Expected: {details['expected_gain']}")
            print(f"  Priority months: {details['targets'][:5]}{'...' if len(details['targets']) > 5 else ''}")
    
    return strategies

def main():
    """Main analysis function."""
    data_by_type, priority_periods = analyze_data_distribution()
    strategies = create_collection_strategy(priority_periods)
    
    print(f"\n📈 SUMMARY:")
    print(f"Total priority periods identified: {len(priority_periods)}")
    print(f"Collection strategies available: {len(strategies)}")
    print(f"\n💡 Next steps: Implement targeted collection for priority periods")

if __name__ == "__main__":
    main()