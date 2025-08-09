#!/usr/bin/env python3
"""
Focused Twitter Data Collector for CryptoPulse

OBJECTIVE: Collect Twitter data from key crypto influencers (2022-2025 timeframe)
APPROACH: Use Twitter API v2 (more reliable than scraping) or alternative methods

TARGET: Fill the critical Twitter gap (0 posts since 2022 → 5,000+ posts)
"""

import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import logging
import json
from dotenv import load_dotenv

load_dotenv()

class FocusedTwitterCollector:
    def __init__(self):
        self.setup_logging()
        self.db_path = "db/cryptopulse.db"
        
        # Focused list of HIGHEST IMPACT crypto accounts
        self.priority_accounts = [
            # Tier 1: Market Movers (MUST HAVE)
            "VitalikButerin",      # Ethereum creator
            "haydenzadams",        # Uniswap founder  
            "a16zcrypto",          # Major VC
            "cz_binance",          # Binance CEO
            "brian_armstrong",     # Coinbase CEO
            
            # Tier 2: High Volume Content
            "DocumentingBTC",      # Very high volume crypto news
            "WuBlockchain",        # Breaking crypto news
            "DefiIgnas",           # DeFi analysis
            "CryptoCred",          # Technical analysis
            "APompliano",          # Popular crypto podcast
            
            # Tier 3: Ethereum Ecosystem
            "evan_van_ness",       # Week in Ethereum
            "TimBeiko",            # Ethereum core dev
            "bantg",               # Yearn Finance
            "divine_economy",      # DeFi education
            "TheDeFiEdge"          # DeFi strategy
        ]
        
        # Collection parameters
        self.target_start = datetime(2022, 1, 1)
        self.target_end = datetime(2025, 8, 1)
        self.collected_count = 0
        self.target_count = 5000
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/focused_twitter_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_bearer_token(self):
        """Get Twitter API bearer token"""
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not bearer_token:
            self.logger.error("❌ TWITTER_BEARER_TOKEN not found in environment")
            self.logger.info("💡 Get it from https://developer.twitter.com/en/portal/dashboard")
            return None
        return bearer_token
    
    def search_twitter_api(self, query, max_results=100):
        """Search Twitter using API v2"""
        bearer_token = self.get_bearer_token()
        if not bearer_token:
            return []
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        
        params = {
            'query': query,
            'max_results': max_results,
            'tweet.fields': 'created_at,author_id,public_metrics,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'username'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API Error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            return []
    
    def collect_alternative_data(self):
        """Alternative collection methods when API is not available"""
        print("🔄 ALTERNATIVE DATA COLLECTION METHODS")
        print("="*50)
        
        alternative_sources = []
        
        # Method 1: Existing social media datasets
        print("📊 Option 1: Kaggle/Academic Datasets")
        print("   - Search for 'crypto twitter dataset 2022-2025'")
        print("   - Look for Ethereum/DeFi focused datasets")
        print("   - Filter by our target timeframe")
        
        # Method 2: RSS feeds of crypto news mentioning Twitter
        print("\\n📰 Option 2: Crypto News RSS Feeds") 
        rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed", 
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://blockworks.co/feed/",
            "https://theblock.co/rss.xml"
        ]
        
        for feed in rss_feeds:
            print(f"   📡 {feed}")
        
        # Method 3: Manual high-impact collection
        print("\\n✋ Option 3: Manual High-Impact Collection")
        print("   Focus on these accounts manually:")
        for i, account in enumerate(self.priority_accounts[:5]):
            print(f"   {i+1}. @{account} - Collect 50-100 key tweets (2022-2025)")
        
        print("\\n💡 Recommended: Start with manual collection of top 5 accounts")
        print("   This could give us 250-500 high-quality tweets quickly")
        
        return alternative_sources
    
    def collect_crypto_news_mentions(self):
        """Collect crypto news articles that mention Twitter discussions"""
        print("\\n📰 COLLECTING CRYPTO NEWS WITH TWITTER MENTIONS")
        print("="*50)
        
        # Simple RSS feed collection for crypto news
        import feedparser
        
        rss_feeds = [
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("Decrypt", "https://decrypt.co/feed"),
            ("The Block", "https://theblock.co/rss.xml")
        ]
        
        collected_articles = []
        
        for source, url in rss_feeds:
            try:
                print(f"📡 Fetching from {source}...")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:  # Limit to recent articles
                    # Check if article mentions Twitter or specific crypto terms
                    content = entry.get('summary', '') + entry.get('title', '')
                    
                    crypto_keywords = ['ethereum', 'ETH', 'DeFi', 'crypto', 'bitcoin', 'twitter']
                    if any(keyword.lower() in content.lower() for keyword in crypto_keywords):
                        
                        article_data = {
                            'source': source,
                            'title': entry.get('title', ''),
                            'content': content,
                            'published_at': entry.get('published', ''),
                            'url': entry.get('link', ''),
                            'scraped_at': datetime.now().isoformat()
                        }
                        
                        collected_articles.append(article_data)
                        self.collected_count += 1
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error collecting from {source}: {str(e)}")
        
        print(f"✅ Collected {len(collected_articles)} relevant articles")
        return collected_articles
    
    def save_to_database(self, data, data_type="twitter"):
        """Save collected data to database"""
        if not data:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        saved_count = 0
        
        try:
            cursor = conn.cursor()
            
            if data_type == "twitter":
                for item in data:
                    cursor.execute("""
                        INSERT OR IGNORE INTO twitter_posts 
                        (id, username, content, likes, retweets, replies, created_at, url, scraped_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.get('id', f"manual_{hash(item['content'])}")[:50],
                        item.get('username', 'manual_collection'),
                        item.get('content', ''),
                        item.get('likes', 0),
                        item.get('retweets', 0), 
                        item.get('replies', 0),
                        item.get('created_at', datetime.now().isoformat()),
                        item.get('url', ''),
                        datetime.now().isoformat()
                    ))
                    saved_count += 1
                    
            elif data_type == "news":
                for item in data:
                    cursor.execute("""
                        INSERT OR IGNORE INTO news_articles
                        (id, source, title, content, published_at, url, scraped_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"news_{hash(item['title'])}"[:50],
                        item.get('source', 'rss_collection'),
                        item.get('title', ''),
                        item.get('content', ''),
                        item.get('published_at', datetime.now().isoformat()),
                        item.get('url', ''),
                        datetime.now().isoformat()
                    ))
                    saved_count += 1
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
        finally:
            conn.close()
        
        if saved_count > 0:
            self.logger.info(f"💾 Saved {saved_count} {data_type} items to database")
        
        return saved_count
    
    def run_focused_collection(self):
        """Execute focused data collection strategy"""
        print("🎯 FOCUSED TWITTER DATA COLLECTION")
        print("="*50)
        print(f"📅 Target timeframe: 2022-2025")
        print(f"🎭 Priority accounts: {len(self.priority_accounts)}")
        print(f"🎯 Target posts: {self.target_count:,}")
        print("="*50)
        
        # Strategy 1: Try Twitter API if available
        bearer_token = self.get_bearer_token()
        if bearer_token:
            print("🔑 Twitter API token found - attempting API collection...")
            # Would implement API collection here
            print("⚠️ Twitter API v2 has limitations for historical data")
            print("💡 Focusing on alternative methods for better coverage")
        
        # Strategy 2: Crypto news collection (immediate value)
        print("\\n📰 Starting crypto news collection...")
        news_articles = self.collect_crypto_news_mentions()
        if news_articles:
            saved = self.save_to_database(news_articles, "news")
            print(f"💾 Saved {saved} news articles")
        
        # Strategy 3: Show manual collection guide
        print("\\n🎯 HIGH-IMPACT MANUAL COLLECTION STRATEGY")
        print("-" * 50)
        print("For maximum ML impact, manually collect from these accounts:")
        print("(Focus on crypto/ETH tweets from 2022-2025)")
        print()
        
        for i, account in enumerate(self.priority_accounts, 1):
            impact_level = "🔥 CRITICAL" if i <= 5 else "📈 HIGH" if i <= 10 else "📊 MEDIUM"
            tweets_target = 100 if i <= 5 else 50 if i <= 10 else 25
            print(f"{i:2d}. @{account:<20} {impact_level} (target: {tweets_target} tweets)")
        
        print(f"\\n💡 MANUAL COLLECTION TIPS:")
        print(f"   • Use Twitter's advanced search: from:username since:2022-01-01 until:2025-08-01")
        print(f"   • Focus on crypto/ethereum/DeFi keywords")
        print(f"   • Save in JSON format for easy import")
        print(f"   • Target total: {self.target_count:,} posts")
        
        # Show template for manual collection
        template_path = "collection/manual_twitter_template.json"
        self.create_manual_template(template_path)
        
        print(f"\\n📝 Manual collection template created: {template_path}")
        print("🎯 Even 500 high-quality tweets would significantly improve our ML dataset!")
        
        return {
            'news_collected': len(news_articles),
            'total_collected': self.collected_count,
            'manual_strategy': self.priority_accounts
        }
    
    def create_manual_template(self, filepath):
        """Create template for manual data collection"""
        template = {
            "collection_metadata": {
                "objective": "Fill Twitter data gap for CryptoPulse ML modeling",
                "timeframe": "2022-01-01 to 2025-08-01", 
                "target_count": self.target_count,
                "focus": "crypto/ethereum/DeFi content from key influencers"
            },
            "priority_accounts": self.priority_accounts,
            "manual_tweets": [
                {
                    "example_tweet": {
                        "username": "VitalikButerin",
                        "content": "Example tweet content about Ethereum...",
                        "created_at": "2024-01-15T10:30:00Z",
                        "url": "https://twitter.com/VitalikButerin/status/123456789",
                        "likes": 1250,
                        "retweets": 340,
                        "replies": 89,
                        "relevance_keywords": ["ethereum", "ETH", "blockchain"],
                        "collection_notes": "Market-moving announcement about Ethereum upgrade"
                    }
                }
            ],
            "instructions": [
                "1. Visit each priority account's Twitter profile",
                "2. Use advanced search: from:username since:2022-01-01 until:2025-08-01",
                "3. Focus on crypto/ETH/DeFi related tweets",
                "4. Copy tweet data into this JSON structure",
                "5. Target 50-100 tweets per high-priority account",
                "6. Import completed data using the import script"
            ]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)

def main():
    """Main execution"""
    collector = FocusedTwitterCollector()
    
    print("🚀 CRYPTOPULSE: FOCUSED DATA COLLECTION")
    print("="*60)
    print("Current situation: 0 Twitter posts since 2022")
    print("Target: 5,000+ posts for robust ML modeling") 
    print("Strategy: Multi-approach collection focusing on 2022-2025")
    print("="*60)
    
    results = collector.run_focused_collection()
    
    print("\\n" + "="*60)
    print("📊 COLLECTION RESULTS SUMMARY")
    print("="*60)
    print(f"📰 News articles collected: {results['news_collected']}")
    print(f"📊 Total items collected: {results['total_collected']}")
    print(f"🎭 Priority accounts identified: {len(results['manual_strategy'])}")
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Review manual collection template")
    print("2. Start with top 5 priority accounts")
    print("3. Collect 50-100 tweets per account") 
    print("4. Import data and regenerate ML dataset")
    print("5. Run hypothesis validation with expanded data")
    
    print("\\n💡 Even 500 high-quality tweets would provide massive ML improvement!")
    print("="*60)

if __name__ == "__main__":
    main()