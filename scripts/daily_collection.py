#!/usr/bin/env python3
# scripts/daily_collection.py

"""
Daily data collection script for CryptoPulse.
Runs all scrapers and price collection in sequence.
Designed to be run via cron job.
"""
import sys
import os
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from reddit_scraper import RedditScraper
from twitter_scraper import TwitterScraper  
from news_scraper import NewsScraper
from price_collector import PriceCollector

def setup_logging():
    """Setup logging for daily collection."""
    log_file = f"logs/daily_collection_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def should_run_twitter():
    """
    Smart Twitter collection schedule to avoid blocks.
    Only runs Twitter 3 times per day at optimal times.
    """
    current_hour = datetime.now().hour
    
    # Run Twitter only at these hours (3 times per day)
    twitter_hours = [6, 14, 22]  # 6 AM, 2 PM, 10 PM (spread across day)
    
    return current_hour in twitter_hours

def run_daily_collection():
    """Run data collection tasks with smart Twitter blocking avoidance."""
    setup_logging()
    logging.info("=== Starting CryptoPulse Data Collection ===")
    
    current_hour = datetime.now().hour
    
    total_collected = {
        'reddit': 0,
        'twitter': 0, 
        'news': 0,
        'prices': 0
    }
    
    try:
        # 1. Collect Reddit posts (ALWAYS - safe for hourly collection)
        logging.info("--- Collecting Reddit Posts ---")
        reddit_scraper = RedditScraper()
        total_collected['reddit'] = reddit_scraper.scrape_all_subreddits(days_back=3)  # Reduced window for hourly
        
        # 2. Collect Twitter posts (SMART SCHEDULE - only 3x per day)
        if should_run_twitter():
            logging.info(f"--- Collecting Twitter Posts (Hour {current_hour}) ---")
            twitter_scraper = TwitterScraper()
            # Use conservative profile-only mode to avoid detection
            total_collected['twitter'] = twitter_scraper.scrape_crypto_profiles_only(max_per_profile=50)
        else:
            logging.info(f"--- Skipping Twitter (Hour {current_hour}) - Block Avoidance ---")
            total_collected['twitter'] = 0
        
        # 3. Collect News articles (ALWAYS - safe for hourly collection)
        logging.info("--- Collecting News Articles ---")
        news_scraper = NewsScraper()
        total_collected['news'] = news_scraper.scrape_all_sources(max_articles_per_source=15)  # Reduced for hourly
        
        # 4. Collect ETH prices (handled separately by cron)
        logging.info("--- Skipping Prices (handled by separate cron job) ---")
        total_collected['prices'] = 0
        
        # Summary
        logging.info("=== Collection Complete ===")
        logging.info(f"Reddit posts: {total_collected['reddit']}")
        logging.info(f"Twitter posts: {total_collected['twitter']}")  
        logging.info(f"News articles: {total_collected['news']}")
        logging.info(f"Total new data points: {sum(total_collected.values())}")
        
        return total_collected
        
    except Exception as e:
        logging.error(f"Error in collection: {e}")
        return total_collected

if __name__ == "__main__":
    results = run_daily_collection()
    print(f"Daily collection complete. New data points: {sum(results.values())}")