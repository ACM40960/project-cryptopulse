#!/usr/bin/env python3
"""
Reddit Historical Backfill System

OBJECTIVE: Fill data gaps in our 2022-2025 timeframe for better ML modeling
TARGET: Add 1,000+ high-quality Reddit posts for gap days

Strategy:
1. Identify specific dates with insufficient Reddit data
2. Use PRAW (Reddit API) for historical data collection
3. Focus on high-engagement crypto subreddits
4. Prioritize ETH-related discussions during gap periods
5. Ensure temporal alignment with price movements

Gap Analysis: 477 days with NO data + 938 days with LOW data = 1,415 days to fill
"""

import praw
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import logging
import os
from dotenv import load_dotenv
from database import CryptoPulseDB

load_dotenv()

class RedditHistoricalBackfill:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.setup_logging()
        self.setup_reddit_api()
        
        # Target subreddits (prioritized by relevance)
        self.target_subreddits = [
            'ethereum',      # Primary ETH discussions
            'ethtrader',     # ETH trading community  
            'ethfinance',    # ETH financial discussions
            'cryptocurrency', # General crypto (high volume)
            'CryptoMarkets', # Trading focused
            'defi',          # DeFi discussions
            'ethstaker',     # ETH 2.0 staking
            'ethdev',        # ETH development
            'UniSwap',       # Major DeFi protocol
            '0xbitcoin'      # ETH-based projects
        ]
        
        # Collection parameters
        self.posts_collected = 0
        self.target_posts = 5000  # Realistic target for gap filling
        
    def setup_logging(self):
        """Setup logging for backfill operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/reddit_backfill.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_reddit_api(self):
        """Initialize Reddit API connection"""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'CryptoPulse:1.0 (by /u/YourUsername)')
            )
            
            # Test connection
            self.reddit.user.me()
            self.logger.info("✅ Reddit API connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Reddit API setup failed: {str(e)}")
            self.logger.info("Please ensure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set in .env")
            self.reddit = None
    
    def identify_gap_periods(self):
        """Identify specific date periods that need data backfilling"""
        self.logger.info("🔍 Identifying Reddit data gaps...")
        
        conn = sqlite3.connect(self.db.db_path)
        
        # Get daily Reddit post counts for 2022-2025
        query = """
        SELECT 
            date(datetime(created_utc, 'unixepoch')) as date,
            COUNT(*) as post_count
        FROM reddit_posts 
        WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01'
        AND datetime(created_utc, 'unixepoch') <= '2025-08-01'
        GROUP BY date(datetime(created_utc, 'unixepoch'))
        ORDER BY date
        """
        
        daily_counts = pd.read_sql_query(query, conn)
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts.set_index('date', inplace=True)
        
        # Create complete date range
        date_range = pd.date_range(start='2022-01-01', end='2025-08-01', freq='D')
        complete_range = pd.DataFrame(index=date_range)
        
        # Merge and identify gaps
        merged = complete_range.join(daily_counts, how='left').fillna(0)
        
        # Define gap criteria
        no_data_days = merged[merged['post_count'] == 0]
        low_data_days = merged[(merged['post_count'] > 0) & (merged['post_count'] < 3)]
        
        gap_periods = {
            'no_data': no_data_days.index.tolist(),
            'low_data': low_data_days.index.tolist()
        }
        
        total_gaps = len(gap_periods['no_data']) + len(gap_periods['low_data'])
        
        self.logger.info(f"📊 Gap Analysis Results:")
        self.logger.info(f"   ❌ Days with no data: {len(gap_periods['no_data'])}")
        self.logger.info(f"   ⚠️ Days with low data: {len(gap_periods['low_data'])}")
        self.logger.info(f"   🎯 Total gap days: {total_gaps}")
        
        conn.close()
        return gap_periods
    
    def collect_historical_posts_for_date(self, target_date: datetime, subreddit_name: str, posts_needed: int = 5):
        """Collect historical posts for a specific date and subreddit"""
        if not self.reddit:
            return []
        
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Search for posts from target date with ETH-related keywords
            search_queries = [
                f'ethereum timestamp:{int(target_date.timestamp())}..{int((target_date + timedelta(days=1)).timestamp())}',
                f'ETH timestamp:{int(target_date.timestamp())}..{int((target_date + timedelta(days=1)).timestamp())}',
                f'crypto timestamp:{int(target_date.timestamp())}..{int((target_date + timedelta(days=1)).timestamp())}'
            ]
            
            for query in search_queries:
                if len(posts) >= posts_needed:
                    break
                    
                try:
                    # Search posts
                    search_results = subreddit.search(query, sort='hot', time_filter='all', limit=10)
                    
                    for submission in search_results:
                        if len(posts) >= posts_needed:
                            break
                            
                        # Verify post is from target date
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        if post_date.date() == target_date.date():
                            
                            post_data = {
                                'id': submission.id,
                                'subreddit': subreddit_name,
                                'title': submission.title,
                                'content': submission.selftext,
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'created_utc': int(submission.created_utc),
                                'url': submission.url,
                                'scraped_at': datetime.now()
                            }
                            
                            posts.append(post_data)
                            
                except Exception as e:
                    self.logger.debug(f"Search query failed for {query}: {str(e)}")
                    continue
            
            # Alternative: Get top posts from that time period
            if len(posts) < posts_needed:
                try:
                    # Get top posts from around that time
                    top_posts = subreddit.top(time_filter='all', limit=50)
                    
                    for submission in top_posts:
                        if len(posts) >= posts_needed:
                            break
                            
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        date_diff = abs((post_date.date() - target_date.date()).days)
                        
                        # Accept posts within 3 days of target
                        if date_diff <= 3:
                            post_data = {
                                'id': submission.id,
                                'subreddit': subreddit_name,
                                'title': submission.title,
                                'content': submission.selftext,
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'created_utc': int(submission.created_utc),
                                'url': submission.url,
                                'scraped_at': datetime.now()
                            }
                            
                            posts.append(post_data)
                            
                except Exception as e:
                    self.logger.debug(f"Top posts query failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting from r/{subreddit_name} for {target_date.date()}: {str(e)}")
        
        return posts
    
    def save_posts_to_database(self, posts):
        """Save collected posts to database"""
        saved_count = 0
        
        for post in posts:
            try:
                # Check for duplicates
                existing = self.db.get_reddit_post_by_id(post['id'])
                if existing:
                    continue
                
                # Insert new post
                self.db.insert_reddit_post(
                    post_id=post['id'],
                    subreddit=post['subreddit'],
                    title=post['title'],
                    content=post['content'],
                    score=post['score'],
                    num_comments=post['num_comments'],
                    created_utc=post['created_utc'],
                    url=post['url']
                )
                
                saved_count += 1
                self.posts_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error saving post {post['id']}: {str(e)}")
        
        if saved_count > 0:
            self.logger.info(f"💾 Saved {saved_count} new posts to database")
        
        return saved_count
    
    def run_backfill_campaign(self):
        """Execute comprehensive Reddit backfill campaign"""
        print("🚀 REDDIT HISTORICAL BACKFILL CAMPAIGN")
        print("="*60)
        
        if not self.reddit:
            print("❌ Reddit API not available. Please check your credentials.")
            return False
        
        # Identify gaps
        gap_periods = self.identify_gap_periods()
        
        # Prioritize recent gaps first (more likely to have data)
        priority_dates = []
        
        # Add recent no-data days (higher priority)
        recent_no_data = [d for d in gap_periods['no_data'] if d >= datetime(2024, 1, 1)]
        priority_dates.extend(recent_no_data[:200])  # Limit to 200 days
        
        # Add recent low-data days  
        recent_low_data = [d for d in gap_periods['low_data'] if d >= datetime(2024, 1, 1)]
        priority_dates.extend(recent_low_data[:300])  # Limit to 300 days
        
        print(f"🎯 Targeting {len(priority_dates)} priority gap days")
        print(f"📊 Target: {self.target_posts:,} posts across {len(self.target_subreddits)} subreddits")
        print("="*60)
        
        # Collection loop
        for i, target_date in enumerate(priority_dates):
            if self.posts_collected >= self.target_posts:
                break
            
            print(f"\\n📅 Processing {target_date.strftime('%Y-%m-%d')} ({i+1}/{len(priority_dates)})")
            print(f"📊 Progress: {self.posts_collected:,}/{self.target_posts:,} posts collected")
            
            daily_posts = []
            
            # Collect from each subreddit for this date
            for subreddit in self.target_subreddits:
                if len(daily_posts) >= 10:  # Max 10 posts per day
                    break
                
                posts_needed = min(3, 10 - len(daily_posts))  # 2-3 posts per subreddit
                posts = self.collect_historical_posts_for_date(target_date, subreddit, posts_needed)
                daily_posts.extend(posts)
                
                # Rate limiting
                time.sleep(1)  # Be nice to Reddit API
            
            # Save collected posts
            if daily_posts:
                saved = self.save_posts_to_database(daily_posts)
                print(f"✅ {target_date.strftime('%Y-%m-%d')}: {saved} posts collected")
            else:
                print(f"⚠️ {target_date.strftime('%Y-%m-%d')}: No posts found")
            
            # Progress check
            if i % 10 == 0 and i > 0:
                print(f"\\n📊 Checkpoint: {self.posts_collected:,} posts collected so far")
        
        print("\\n" + "="*60)
        print("🎉 REDDIT BACKFILL CAMPAIGN COMPLETE!")
        print(f"📊 Total posts collected: {self.posts_collected:,}")
        print(f"🎯 Target achievement: {(self.posts_collected/self.target_posts)*100:.1f}%")
        print("="*60)
        
        return True

def main():
    """Main execution function"""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Check Reddit API credentials
    if not os.getenv('REDDIT_CLIENT_ID'):
        print("❌ Reddit API credentials not found!")
        print("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your .env file")
        print("Get credentials at: https://www.reddit.com/prefs/apps")
        return
    
    # Initialize backfill system
    backfill = RedditHistoricalBackfill()
    
    print("🔶 REDDIT HISTORICAL BACKFILL SYSTEM")
    print("="*50)
    print(f"🎯 Target subreddits: {len(backfill.target_subreddits)}")
    for subreddit in backfill.target_subreddits:
        print(f"   📂 r/{subreddit}")
    
    print(f"\\n📊 Target posts: {backfill.target_posts:,}")
    print("⏱️ Estimated time: 2-3 hours")
    print("="*50)
    
    # Confirm before starting
    proceed = input("\\nProceed with Reddit historical backfill? (y/N): ").lower().strip()
    
    if proceed == 'y':
        success = backfill.run_backfill_campaign()
        if success:
            print("\\n🎉 Reddit backfill completed successfully!")
            print("📊 Ready to regenerate ML dataset with expanded Reddit data.")
        else:
            print("\\n❌ Reddit backfill failed. Check logs for details.")
    else:
        print("\\n⏹️ Backfill cancelled by user.")

if __name__ == "__main__":
    main()