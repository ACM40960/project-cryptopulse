# src/reddit_scraper.py

"""Reddit scraper using PRAW for crypto-related subreddits, with duplicate skipping."""
import os
import time
import logging
from datetime import datetime, timedelta

import praw
import pandas as pd
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reddit_scraper.log'),
        logging.StreamHandler()
    ]
)

class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.db = CryptoPulseDB()
        self.target_subreddits = [
            'ethereum', 'ethtrader', 'cryptocurrency',
            'cryptomarkets', 'defi', 'ethfinance'
        ]

    def scrape_subreddit_historical(self, subreddit_name, days_back=365, posts_per_day=300):
        """Scrape up to `posts_per_day` crypto-relevant posts per sorting method for the past `days_back` days."""
        logging.info(f"Starting historical scrape of r/{subreddit_name} for {days_back} days")
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []

        # different scraping slices
        scraping_methods = [
            ('top', 'week', posts_per_day // 2),
            ('top', 'month', posts_per_day // 2),
            ('top', 'year', posts_per_day),
            ('hot', None, posts_per_day // 2)
        ]

        for method, time_filter, limit in scraping_methods:
            logging.info(f"Scraping {method} ({time_filter or 'current'}) posts: limit={limit}")
            try:
                if method == 'top' and time_filter:
                    posts = subreddit.top(time_filter=time_filter, limit=limit)
                elif method == 'hot':
                    posts = subreddit.hot(limit=limit)
                else:
                    continue

                for post in posts:
                    # -- DUPLICATE CHECK: skip posts already in DB
                    if self.db.record_exists('reddit_posts', post.id):
                        logging.debug(f"Skipped existing Reddit post {post.id}")
                        continue

                    text = (post.title + " " + post.selftext).lower()
                    keywords = ['ethereum', 'eth', 'crypto', 'defi', 'blockchain', 'trading']
                    if not any(k in text for k in keywords):
                        continue

                    posts_data.append({
                        'id': post.id,
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'content': post.selftext or '',
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'url': post.url
                    })

                logging.info(f"Collected {len(posts_data)} posts so far from {method}")
                time.sleep(2)

            except Exception as e:
                logging.warning(f"Error scraping {method} posts: {e}")
                continue

        if not posts_data:
            logging.warning(f"No posts collected from r/{subreddit_name}")
            return 0

        df = pd.DataFrame(posts_data)
        df = df.drop_duplicates(subset=['id'])
        df = df.sort_values('score', ascending=False)

        inserted = self.db.insert_reddit_posts(df)
        logging.info(f"r/{subreddit_name}: {len(df)} processed → {inserted} new saved")
        return inserted

    def scrape_all_subreddits(self, days_back=180):
        """Loop over all target subreddits."""
        total = 0
        for sub in self.target_subreddits:
            total += self.scrape_subreddit_historical(sub, days_back)
            time.sleep(5)
        logging.info(f"Total new Reddit posts: {total}")
        return total

if __name__ == "__main__":
    scraper = RedditScraper()
    count = scraper.scrape_all_subreddits(days_back=180)
    print(f"Total new Reddit posts collected: {count}")
