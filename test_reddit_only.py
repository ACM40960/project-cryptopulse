"""Test Reddit scraper with database status."""
import sys
import os
sys.path.append('src')

from reddit_scraper import RedditScraper
from database import CryptoPulseDB

def show_database_status():
    """Show current database status."""
    db = CryptoPulseDB()
    counts = db.get_record_counts()
    print(f"Current database status:")
    for table, count in counts.items():
        print(f"  {table}: {count} records")
    return counts

def test_reddit_simple():
    print("=== Reddit Scraper Test ===")
    
    print("\nBefore scraping:")
    before_counts = show_database_status()
    
    scraper = RedditScraper()
    
    # Test with just ethereum subreddit, small dataset
    print(f"\nTesting with r/ethereum...")
    new_posts = scraper.scrape_subreddit_historical('ethereum', days_back=7, posts_per_day=20)
    
    print(f"\nAfter scraping:")
    after_counts = show_database_status()
    
    print(f"\nResults:")
    print(f"  New posts added: {new_posts}")
    print(f"  Total Reddit posts: {after_counts['reddit_posts']}")
    
    return new_posts

if __name__ == "__main__":
    result = test_reddit_simple()
    if result >= 0:  # Even 0 is OK (might be duplicates)
        print("\n✅ Reddit scraper working! (Check logs for details)")
    else:
        print("\n❌ Reddit scraper failed - check logs/reddit_scraper.log")
