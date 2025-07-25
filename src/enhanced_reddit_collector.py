# src/enhanced_reddit_collector.py

"""
Enhanced Reddit collector targeting specific historical periods
with date-range search and expanded subreddit coverage.
"""
import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_reddit_collector.log'),
        logging.StreamHandler()
    ]
)

class EnhancedRedditCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        
        # Reddit API configuration
        self.reddit_config = {
            'base_url': 'https://api.reddit.com',
            'user_agent': 'CryptoPulse/1.0 (Data Collection Bot)',
            'rate_limit_delay': 2  # seconds between requests
        }
        
        # Expanded subreddit list for historical collection
        self.target_subreddits = [
            # Core crypto subreddits
            'cryptocurrency', 'ethereum', 'ethtrader', 'ethinvestor', 
            'defi', 'ethereumnoobies', 'ethdev', 'ethstaker',
            
            # Trading and investment
            'cryptomarkets', 'bitcoinmarkets', 'altcoin', 'cryptocurrencytrading',
            'satoshistreetbets', 'altstreetbets', 'cryptotrade',
            
            # DeFi and protocols
            'uniswap', 'aave', 'compound', 'makerdao', 'yearn_finance',
            'sushiswap', 'balancer', 'synthetix_io', 'defiblockchain',
            
            # Layer 2 and scaling
            'optimism', 'arbitrum', 'polygon', 'loopringorg', 'starknet',
            'zksync', 'matic',
            
            # General and historical
            'investing', 'technology', 'futurology', 'crypto_general',
            'btc', 'bitcoin', 'cryptotech', 'blockchain'
        ]
        
        # Priority historical periods (low activity periods)
        self.priority_periods = [
            '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07',
            '2018-08', '2018-09', '2018-10', '2018-12',
            '2019-01', '2019-02', '2019-04', '2019-05', '2019-06', '2019-07', 
            '2019-08', '2019-11', '2019-12',
            '2020-01', '2020-02', '2020-04', '2020-05', '2020-06', '2020-07',
            '2020-08', '2020-10', '2020-11'
        ]
        
        # Key crypto event periods (need intensive collection)
        self.key_event_periods = {
            '2017-12': "ETH ATH period",
            '2018-01': "Crypto winter start", 
            '2020-03': "COVID crash",
            '2020-12': "DeFi summer peak",
            '2021-05': "ETH 2.0 hype",
            '2021-11': "ETH ATH period",
            '2022-06': "Terra Luna collapse",
            '2022-11': "FTX collapse"
        }
    
    def search_reddit_by_date_range(self, subreddit, start_date, end_date, search_terms=None):
        """Search Reddit posts within specific date range."""
        logging.info(f"Searching r/{subreddit} from {start_date} to {end_date}")
        
        posts = []
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        try:
            # Use Pushshift API for historical data (if available)
            pushshift_url = "https://api.pushshift.io/reddit/search/submission"
            
            params = {
                'subreddit': subreddit,
                'after': start_timestamp,
                'before': end_timestamp,
                'size': 500,  # Max per request
                'sort': 'desc',
                'sort_type': 'created_utc'
            }
            
            # Add search terms if provided
            if search_terms:
                params['q'] = ' OR '.join(search_terms)
            
            response = self.session.get(pushshift_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('data', []):
                    # Filter for Ethereum relevance
                    title = post.get('title', '').lower()
                    content = post.get('selftext', '').lower()
                    text = title + ' ' + content
                    
                    eth_keywords = ['ethereum', 'eth', 'ether', 'smart contract', 'defi', 
                                   'gas fee', 'eip', 'vitalik', 'consensys', 'metamask']
                    
                    if any(keyword in text for keyword in eth_keywords):
                        posts.append({
                            'id': post.get('id'),
                            'subreddit': subreddit,
                            'title': post.get('title', ''),
                            'content': post.get('selftext', ''),
                            'score': post.get('score', 0),
                            'num_comments': post.get('num_comments', 0),
                            'created_utc': post.get('created_utc', 0),
                            'url': f"https://reddit.com{post.get('permalink', '')}"
                        })
                
                logging.info(f"Found {len(posts)} relevant posts in r/{subreddit}")
            
            else:
                # Fallback to Reddit API search (limited historical access)
                logging.warning(f"Pushshift failed, trying Reddit API for r/{subreddit}")
                posts = self.fallback_reddit_search(subreddit, start_date, end_date)
                
        except Exception as e:
            logging.warning(f"Search failed for r/{subreddit}: {e}")
            posts = []
        
        time.sleep(self.reddit_config['rate_limit_delay'])
        return posts
    
    def fallback_reddit_search(self, subreddit, start_date, end_date):
        """Fallback method using Reddit API search."""
        posts = []
        
        try:
            # Use Reddit's search with time filters
            search_url = f"{self.reddit_config['base_url']}/r/{subreddit}/search.json"
            
            params = {
                'q': 'ethereum OR eth OR ether',
                'restrict_sr': 'true',
                'sort': 'new',
                'limit': 100,
                't': 'all'  # All time
            }
            
            headers = {'User-Agent': self.reddit_config['user_agent']}
            response = self.session.get(search_url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for post_data in data.get('data', {}).get('children', []):
                    post = post_data.get('data', {})
                    created_utc = post.get('created_utc', 0)
                    post_date = datetime.fromtimestamp(created_utc)
                    
                    # Filter by date range
                    if start_date <= post_date <= end_date:
                        posts.append({
                            'id': post.get('id'),
                            'subreddit': subreddit,
                            'title': post.get('title', ''),
                            'content': post.get('selftext', ''),
                            'score': post.get('score', 0),
                            'num_comments': post.get('num_comments', 0),
                            'created_utc': created_utc,
                            'url': f"https://reddit.com{post.get('permalink', '')}"
                        })
                
                logging.info(f"Fallback found {len(posts)} posts in r/{subreddit}")
                
        except Exception as e:
            logging.debug(f"Fallback search failed for r/{subreddit}: {e}")
        
        return posts
    
    def collect_for_period(self, year_month, intensive=False):
        """Collect Reddit posts for specific year-month period."""
        logging.info(f"{'Intensive' if intensive else 'Standard'} collection for {year_month}")
        
        try:
            year, month = map(int, year_month.split('-'))
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            all_posts = []
            
            # Define search terms for the period
            search_terms = ['ethereum', 'eth', 'ether', 'smart contract', 'defi']
            
            # If intensive collection, add event-specific terms
            if intensive and year_month in self.key_event_periods:
                event_terms = {
                    '2017-12': ['price', 'ath', 'all time high', 'bubble'],
                    '2018-01': ['crash', 'correction', 'winter', 'bear'],
                    '2020-03': ['covid', 'coronavirus', 'pandemic', 'crash'],
                    '2020-12': ['defi', 'yield farming', 'liquidity mining'],
                    '2021-05': ['eip-1559', 'london', 'proof of stake', 'merge'],
                    '2021-11': ['nft', 'opensea', 'ath', 'metaverse'],
                    '2022-06': ['terra', 'luna', 'ust', 'depeg', 'collapse'],
                    '2022-11': ['ftx', 'sbf', 'bankruptcy', 'contagion']
                }
                search_terms.extend(event_terms.get(year_month, []))
            
            # Select subreddits (more for intensive collection)
            subreddits = self.target_subreddits[:20] if intensive else self.target_subreddits[:10]
            
            for subreddit in subreddits:
                try:
                    period_posts = self.search_reddit_by_date_range(
                        subreddit, start_date, end_date, search_terms
                    )
                    all_posts.extend(period_posts)
                    
                    # More aggressive collection for key events
                    if intensive:
                        time.sleep(1)  # Shorter delay for intensive collection
                    
                except Exception as e:
                    logging.debug(f"Failed to collect from r/{subreddit} for {year_month}: {e}")
                    continue
            
            return all_posts
            
        except Exception as e:
            logging.error(f"Period collection failed for {year_month}: {e}")
            return []
    
    def save_historical_posts(self, posts, period):
        """Save collected posts to database."""
        if not posts:
            logging.info(f"No posts to save for {period}")
            return 0
        
        # Remove duplicates and filter existing
        df = pd.DataFrame(posts)
        df = df.drop_duplicates(subset=['id'])
        
        new_posts = []
        for _, post in df.iterrows():
            if not self.db.record_exists('reddit_posts', post['id']):
                new_posts.append(post.to_dict())
        
        if new_posts:
            new_df = pd.DataFrame(new_posts)
            inserted = self.db.insert_reddit_posts(new_df)
            logging.info(f"{period}: {len(posts)} collected → {inserted} new saved")
            return inserted
        
        logging.info(f"{period}: {len(posts)} collected → 0 new (all duplicates)")
        return 0
    
    def boost_historical_periods(self):
        """Main method to boost historical periods with low Reddit activity."""
        logging.info("=== Enhanced Reddit Historical Collection ===")
        
        total_collected = 0
        
        # Priority periods (standard collection)
        logging.info(f"Collecting for {len(self.priority_periods)} priority periods...")
        for period in self.priority_periods[:10]:  # Limit for testing
            posts = self.collect_for_period(period, intensive=False)
            saved = self.save_historical_posts(posts, period)
            total_collected += saved
            
            if saved > 0:
                logging.info(f"✅ {period}: Boosted with {saved} new posts")
            else:
                logging.info(f"⚪ {period}: No new posts found")
        
        # Key event periods (intensive collection)
        logging.info(f"Intensive collection for {len(self.key_event_periods)} key events...")
        for period, event in list(self.key_event_periods.items())[:3]:  # Limit for testing
            posts = self.collect_for_period(period, intensive=True)
            saved = self.save_historical_posts(posts, period)
            total_collected += saved
            
            if saved > 0:
                logging.info(f"🎯 {period} ({event}): Enhanced with {saved} new posts")
            else:
                logging.info(f"⚪ {period} ({event}): No new posts found")
        
        logging.info(f"=== Enhanced Reddit Collection Complete ===")
        logging.info(f"Total new posts collected: {total_collected}")
        
        return total_collected

def main():
    """Main function for enhanced Reddit collection."""
    collector = EnhancedRedditCollector()
    total = collector.boost_historical_periods()
    print(f"Enhanced Reddit collection complete. Collected {total} new posts.")

if __name__ == "__main__":
    main()