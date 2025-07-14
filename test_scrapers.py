"""Test script to validate scrapers before overnight run."""
import sys
import os
sys.path.append('src')

from reddit_scraper import RedditScraper
from twitter_scraper import TwitterScraper
from database import CryptoPulseDB

def test_reddit():
    print("Testing Reddit scraper...")
    scraper = RedditScraper()
    # Test with small dataset first
    count = scraper.scrape_subreddit_historical('ethereum', days_back=7, posts_per_day=10)
    print(f"Reddit test: {count} posts collected")
    return count > 0

def test_twitter():
    print("Testing Twitter scraper...")
    scraper = TwitterScraper()
    count = scraper.scrape_all_queries(max_tweets_per_query=5)
    print(f"Twitter test: {count} tweets collected")
    return count > 0

def check_database():
    print("Checking database...")
    db = CryptoPulseDB()
    # Add basic database check here
    print("Database initialized successfully")

if __name__ == "__main__":
    print("=== CryptoPulse Scraper Tests ===")
    check_database()
    
    reddit_ok = test_reddit()
    twitter_ok = test_twitter()
    
    print(f"Reddit scraper: {'✅' if reddit_ok else '❌'}")
    print(f"Twitter scraper: {'✅' if twitter_ok else '❌'}")
    
    if reddit_ok and twitter_ok:
        print("All tests passed! Ready for overnight data collection.")
    else:
        print("Some tests failed. Check logs for details.")
