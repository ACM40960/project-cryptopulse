#!/usr/bin/env python3
"""
Focused Reddit Data Collector for CryptoPulse

OBJECTIVE: Fill Reddit data gaps in our 2022-2025 timeframe
APPROACH: Use PRAW (Reddit API) to target specific gap periods

TARGET: Fill 477 no-data days + 938 low-data days = 1,415 gap days
IMPACT: +1,132 ML samples (massive improvement)
"""

import praw
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class FocusedRedditCollector:
    def __init__(self):
        self.setup_logging()
        self.db_path = "db/cryptopulse.db"
        
        # High-impact subreddits for crypto/ETH data
        self.priority_subreddits = [
            # Tier 1: Ethereum focused (highest relevance)
            'ethereum',
            'ethtrader', 
            'ethfinance',
            'ethstaker',
            
            # Tier 2: General crypto (high volume)
            'cryptocurrency',
            'CryptoMarkets',
            'CryptoCurrency',
            
            # Tier 3: DeFi focused
            'defi',
            'UniSwap',
            '0xbitcoin',
            'ethdev'
        ]
        
        # Collection parameters
        self.target_start = datetime(2022, 1, 1)
        self.target_end = datetime(2025, 8, 1)
        self.collected_count = 0
        self.target_count = 3000  # Realistic target for gap filling
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/focused_reddit_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_reddit_api(self):
        """Setup Reddit API connection"""
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'CryptoPulse:1.0')
        
        if not client_id or not client_secret:
            self.logger.error("❌ Reddit API credentials missing")
            self.logger.info("💡 Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
            self.logger.info("💡 Get credentials at: https://www.reddit.com/prefs/apps")
            return None
        
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test connection
            reddit.subreddit('test').id
            self.logger.info("✅ Reddit API connection successful")
            return reddit
            
        except Exception as e:
            self.logger.error(f"❌ Reddit API setup failed: {str(e)}")
            return None
    
    def identify_gap_periods(self):
        """Identify specific dates that need Reddit data"""
        self.logger.info("🔍 Identifying Reddit data gaps for 2022-2025...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get daily Reddit post counts
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
        
        # Create complete date range for our target period
        date_range = pd.date_range(start='2022-01-01', end='2025-08-01', freq='D')
        complete_range = pd.DataFrame(index=date_range)
        
        # Merge and identify gaps
        merged = complete_range.join(daily_counts, how='left').fillna(0)
        
        # Priority gaps (most recent and most severe)
        no_data_days = merged[merged['post_count'] == 0].index.tolist()
        low_data_days = merged[(merged['post_count'] > 0) & (merged['post_count'] < 3)].index.tolist()
        
        # Focus on recent gaps (2024-2025) first
        recent_gaps = [d for d in no_data_days if d >= datetime(2024, 1, 1)][:100]  # Top 100
        recent_low = [d for d in low_data_days if d >= datetime(2024, 1, 1)][:150]   # Top 150
        
        priority_dates = recent_gaps + recent_low
        
        self.logger.info(f"📊 Gap Analysis:")
        self.logger.info(f"   ❌ Total no-data days: {len(no_data_days)}")
        self.logger.info(f"   ⚠️ Total low-data days: {len(low_data_days)}")
        self.logger.info(f"   🎯 Priority dates to fill: {len(priority_dates)}")
        
        conn.close()
        return priority_dates
    
    def collect_posts_for_date_range(self, reddit, subreddit_name, start_date, end_date, limit=10):
        """Collect Reddit posts for a specific date range"""
        posts = []
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Search for posts in date range with crypto keywords
            search_queries = [
                f'ethereum OR ETH OR crypto timestamp:{int(start_date.timestamp())}..{int(end_date.timestamp())}',
                f'price OR trading OR market timestamp:{int(start_date.timestamp())}..{int(end_date.timestamp())}',
                f'DeFi OR defi timestamp:{int(start_date.timestamp())}..{int(end_date.timestamp())}'
            ]
            
            for query in search_queries:
                if len(posts) >= limit:
                    break
                
                try:
                    search_results = subreddit.search(query, sort='hot', time_filter='all', limit=5)
                    
                    for submission in search_results:
                        if len(posts) >= limit:
                            break
                        
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        
                        # Verify post is in our target timeframe
                        if start_date <= post_date <= end_date:
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
                    self.logger.debug(f"Search failed for {query}: {str(e)}")
                    continue
            
            # Fallback: Get recent hot posts from subreddit
            if len(posts) < limit:
                try:
                    hot_posts = subreddit.hot(limit=20)
                    
                    for submission in hot_posts:
                        if len(posts) >= limit:
                            break
                        
                        post_date = datetime.fromtimestamp(submission.created_utc)
                        
                        # Accept posts within 7 days of target range
                        if abs((post_date - start_date).days) <= 7:
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
                    self.logger.debug(f"Hot posts collection failed: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"❌ Error collecting from r/{subreddit_name}: {str(e)}")
        
        return posts
    
    def save_posts_to_database(self, posts):
        """Save collected posts to database"""
        if not posts:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        saved_count = 0
        
        try:
            cursor = conn.cursor()
            
            for post in posts:
                try:
                    # Check for existing post
                    cursor.execute("SELECT id FROM reddit_posts WHERE id = ?", (post['id'],))
                    if cursor.fetchone():
                        continue  # Skip duplicates
                    
                    # Insert new post
                    cursor.execute("""
                        INSERT INTO reddit_posts 
                        (id, subreddit, title, content, score, num_comments, created_utc, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        post['id'],
                        post['subreddit'],
                        post['title'],
                        post['content'],
                        post['score'],
                        post['num_comments'],
                        post['created_utc'],
                        post['url']
                    ))
                    
                    saved_count += 1
                    self.collected_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error saving post {post['id']}: {str(e)}")
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
        finally:
            conn.close()
        
        return saved_count
    
    def run_focused_collection(self):
        """Execute focused Reddit collection to fill gaps"""
        print("🔶 FOCUSED REDDIT DATA COLLECTION")
        print("="*50)
        print(f"📅 Target timeframe: 2022-2025")
        print(f"🎯 Target posts: {self.target_count:,}")
        print(f"📂 Priority subreddits: {len(self.priority_subreddits)}")
        print("="*50)
        
        # Setup Reddit API
        reddit = self.setup_reddit_api()
        if not reddit:
            print("❌ Cannot proceed without Reddit API access")
            print("💡 Alternative: Use existing Reddit scrapers or manual collection")
            return {'error': 'no_reddit_api'}
        
        # Identify gap periods
        gap_dates = self.identify_gap_periods()
        if not gap_dates:
            print("✅ No significant gaps found in Reddit data")
            return {'status': 'no_gaps_needed'}
        
        print(f"\\n🎯 Filling {len(gap_dates)} priority gap periods...")
        
        # Collection loop
        total_collected = 0
        
        # Process gaps in weekly chunks for efficiency
        current_date = gap_dates[0]
        end_date = min(gap_dates[-1], current_date + timedelta(days=30))  # Process 30 days at a time
        
        week_count = 0
        max_weeks = 10  # Limit to prevent excessive API usage
        
        while current_date <= end_date and week_count < max_weeks and total_collected < self.target_count:
            week_end = current_date + timedelta(days=7)
            week_count += 1
            
            print(f"\\n📅 Processing week {week_count}: {current_date.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
            
            week_posts = []
            
            # Collect from each priority subreddit for this week
            for subreddit in self.priority_subreddits[:6]:  # Limit to top 6 subreddits
                if len(week_posts) >= 30:  # Max posts per week
                    break
                
                posts = self.collect_posts_for_date_range(
                    reddit, subreddit, current_date, week_end, limit=5
                )
                week_posts.extend(posts)
                
                # Rate limiting
                time.sleep(1)
            
            # Save week's collection
            if week_posts:
                saved = self.save_posts_to_database(week_posts)
                total_collected += saved
                print(f"✅ Week {week_count}: {saved} posts collected and saved")
            else:
                print(f"⚠️ Week {week_count}: No posts found")
            
            current_date = week_end
            
            # Progress update
            if week_count % 3 == 0:
                progress = (total_collected / self.target_count) * 100
                print(f"\\n📊 Progress: {total_collected:,}/{self.target_count:,} posts ({progress:.1f}%)")
        
        print("\\n" + "="*50)
        print("📊 REDDIT COLLECTION RESULTS")
        print("="*50)
        print(f"✅ Total posts collected: {total_collected:,}")
        print(f"📈 Target achievement: {(total_collected/self.target_count)*100:.1f}%")
        print(f"⏱️ Weeks processed: {week_count}")
        
        return {
            'posts_collected': total_collected,
            'weeks_processed': week_count,
            'target_achievement': (total_collected/self.target_count)*100
        }

def main():
    """Main execution"""
    collector = FocusedRedditCollector()
    
    print("🚀 CRYPTOPULSE: FOCUSED REDDIT COLLECTION")
    print("="*60)
    print("Objective: Fill data gaps in 2022-2025 timeframe")
    print("Current gaps: 477 no-data days + 938 low-data days")
    print("Expected ML impact: +1,132 samples")
    print("="*60)
    
    # Check Reddit API credentials first
    if not os.getenv('REDDIT_CLIENT_ID'):
        print("\\n❌ REDDIT API SETUP REQUIRED")
        print("="*40)
        print("1. Go to: https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Choose 'script' type")
        print("4. Add credentials to .env file:")
        print("   REDDIT_CLIENT_ID=your_client_id")
        print("   REDDIT_CLIENT_SECRET=your_client_secret")
        print("   REDDIT_USER_AGENT=CryptoPulse:1.0")
        print("\\n💡 This is free and takes 2-3 minutes to setup")
        print("="*40)
        return
    
    # Run collection
    results = collector.run_focused_collection()
    
    if 'error' in results:
        print("❌ Collection failed - check Reddit API setup")
    elif 'status' in results and results['status'] == 'no_gaps_needed':
        print("✅ Reddit data coverage is already good!")
    else:
        print("\\n🎯 COLLECTION COMPLETE!")
        print(f"📊 Posts collected: {results['posts_collected']:,}")
        print(f"📈 Target achievement: {results['target_achievement']:.1f}%")
        print("\\n💡 Ready to regenerate ML dataset with expanded Reddit data!")

if __name__ == "__main__":
    main()