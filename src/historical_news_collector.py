# src/historical_news_collector.py

"""
Historical news collector for filling specific date gaps in crypto news coverage.
Uses multiple strategies: RSS archives, Google News, Wayback Machine, and direct site search.
"""
import os
import time
import logging
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/historical_news_collector.log'),
        logging.StreamHandler()
    ]
)

class HistoricalNewsCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Historical collection strategies
        self.historical_sources = {
            'rss_feeds': {
                'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'cointelegraph': 'https://cointelegraph.com/rss',
                'decrypt': 'https://decrypt.co/feed',
                'theblock': 'https://www.theblockcrypto.com/rss.xml',
                'cryptoslate': 'https://cryptoslate.com/feed/',
                'bitcoinist': 'https://bitcoinist.com/feed/',
                'newsbtc': 'https://www.newsbtc.com/feed/',
                'ambcrypto': 'https://ambcrypto.com/feed/',
                'cryptonews': 'https://cryptonews.com/feed/',
                'u_today': 'https://u.today/rss'
            },
            'google_news': {
                'base_url': 'https://news.google.com/rss/search',
                'ethereum_queries': [
                    'ethereum price OR ETH cryptocurrency',
                    'ethereum defi OR decentralized finance',
                    'ethereum smart contracts OR solidity',
                    'ethereum merge OR proof of stake',
                    'ethereum gas fees OR network congestion'
                ],
                'date_ranges': {
                    '2024': 'after:2024-01-01 before:2024-12-31',
                    '2023': 'after:2023-01-01 before:2023-12-31',
                    '2022': 'after:2022-01-01 before:2022-12-31',
                    '2021': 'after:2021-01-01 before:2021-12-31',
                    '2020': 'after:2020-01-01 before:2020-12-31'
                }
            },
            'wayback_machine': {
                'base_url': 'https://web.archive.org/web',
                'target_sites': [
                    'coindesk.com/markets/ethereum',
                    'cointelegraph.com/tags/ethereum',
                    'decrypt.co/learn/category/ethereum'
                ]
            }
        }
        
        # Target months that need historical coverage
        self.target_months = [
            '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
            '2018-01', '2018-02', '2018-03', '2018-04', '2018-11',
            '2019-09', '2019-10',
            '2020-09',
            '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', 
            '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
            '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07',
            '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
            '2023-01', '2023-03', '2023-04', '2023-08', '2023-09', '2023-10', '2023-11',
            '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'
        ]
    
    def collect_rss_historical(self, days_back=180):
        """Collect historical articles from RSS feeds."""
        logging.info(f"Collecting RSS historical data ({days_back} days back)")
        articles = []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source, rss_url in self.historical_sources['rss_feeds'].items():
            try:
                logging.info(f"Processing RSS feed: {source}")
                
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                if not feed.entries:
                    logging.warning(f"No entries found in RSS feed for {source}")
                    continue
                
                for entry in feed.entries[:100]:  # Limit per feed
                    try:
                        # Extract article data
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse publication date
                        published_at = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_at = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published_at = datetime(*entry.updated_parsed[:6])
                        
                        # Skip if too old
                        if published_at < cutoff_date:
                            continue
                        
                        # Filter for crypto relevance
                        text = (title + " " + summary).lower()
                        crypto_keywords = ['ethereum', 'eth', 'crypto', 'blockchain', 'defi', 'smart contract']
                        if not any(keyword in text for keyword in crypto_keywords):
                            continue
                        
                        # Generate unique ID
                        article_id = hashlib.md5(link.encode()).hexdigest()
                        
                        # Check if already exists
                        if self.db.record_exists('news_articles', article_id):
                            continue
                        
                        articles.append({
                            'id': article_id,
                            'source': f'rss_{source}',
                            'title': title,
                            'content': summary,
                            'published_at': published_at,
                            'url': link
                        })
                        
                    except Exception as e:
                        logging.debug(f"Failed to process RSS entry: {e}")
                        continue
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"Failed to process RSS feed {source}: {e}")
                continue
        
        return articles
    
    def collect_google_news_historical(self, target_year='2024'):
        """Collect historical articles from Google News RSS."""
        logging.info(f"Collecting Google News historical data for {target_year}")
        articles = []
        
        if target_year not in self.historical_sources['google_news']['date_ranges']:
            logging.warning(f"No date range configured for {target_year}")
            return articles
        
        date_filter = self.historical_sources['google_news']['date_ranges'][target_year]
        queries = self.historical_sources['google_news']['ethereum_queries']
        
        for query in queries:
            try:
                # Construct Google News RSS URL with date filtering
                search_query = f"{query} {date_filter}"
                encoded_query = quote(search_query)
                rss_url = f"{self.historical_sources['google_news']['base_url']}?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                
                logging.info(f"Processing Google News query: {query}")
                
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                if not feed.entries:
                    logging.warning(f"No entries found for query: {query}")
                    continue
                
                for entry in feed.entries[:50]:  # Limit per query
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Extract publication date
                        published_at = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_at = datetime(*entry.published_parsed[:6])
                        
                        # Generate unique ID
                        article_id = hashlib.md5(link.encode()).hexdigest()
                        
                        # Check if already exists
                        if self.db.record_exists('news_articles', article_id):
                            continue
                        
                        articles.append({
                            'id': article_id,
                            'source': 'google_news',
                            'title': title,
                            'content': summary,
                            'published_at': published_at,
                            'url': link
                        })
                        
                    except Exception as e:
                        logging.debug(f"Failed to process Google News entry: {e}")
                        continue
                
                time.sleep(5)  # More conservative rate limiting for Google
                
            except Exception as e:
                logging.warning(f"Failed to process Google News query '{query}': {e}")
                continue
        
        return articles
    
    def collect_wayback_machine_snapshots(self, target_month):
        """Collect articles from Wayback Machine snapshots for specific months."""
        logging.info(f"Collecting Wayback Machine data for {target_month}")
        articles = []
        
        try:
            year, month = target_month.split('-')
            # Target middle of the month for snapshots
            target_date = f"{year}{month}15"
            
            for site in self.historical_sources['wayback_machine']['target_sites']:
                try:
                    # Get available snapshots
                    wayback_url = f"{self.historical_sources['wayback_machine']['base_url']}/{target_date}/*/{site}"
                    
                    response = self.session.get(wayback_url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find article links in archived page
                        article_links = soup.find_all('a', href=True)
                        
                        for link in article_links[:20]:  # Limit per site
                            href = link.get('href')
                            text = link.get_text()
                            
                            # Filter for Ethereum-related content
                            if any(term in text.lower() for term in ['ethereum', 'eth', 'defi', 'smart contract']):
                                # This would need more sophisticated extraction
                                # For now, we'll log the potential
                                logging.info(f"Found potential historical article: {text[:50]}")
                    
                    time.sleep(10)  # Very conservative for Wayback Machine
                    
                except Exception as e:
                    logging.warning(f"Wayback Machine query failed for {site}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Wayback Machine collection failed for {target_month}: {e}")
        
        return articles
    
    def collect_targeted_historical_search(self, target_months=None):
        """Collect historical news for specific target months using multiple strategies."""
        if target_months is None:
            target_months = self.target_months
        
        logging.info(f"Starting targeted historical collection for {len(target_months)} months")
        all_articles = []
        
        # Strategy 1: RSS feeds (last 6 months)
        rss_articles = self.collect_rss_historical(days_back=180)
        all_articles.extend(rss_articles)
        logging.info(f"RSS collection: {len(rss_articles)} articles")
        
        # Strategy 2: Google News by year
        years = set(month.split('-')[0] for month in target_months)
        for year in years:
            if int(year) >= 2020:  # Google News doesn't go back too far
                google_articles = self.collect_google_news_historical(year)
                all_articles.extend(google_articles)
                logging.info(f"Google News {year}: {len(google_articles)} articles")
        
        # Strategy 3: Site-specific searches (limited implementation)
        # This would require site-specific search implementations
        
        return all_articles
    
    def save_historical_articles(self, articles):
        """Save collected historical articles to database."""
        if not articles:
            logging.info("No historical articles to save")
            return 0
        
        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['id'])
        
        # Filter for new articles only
        new_articles = []
        for _, article in df.iterrows():
            if not self.db.record_exists('news_articles', article['id']):
                new_articles.append(article.to_dict())
        
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            inserted = self.db.insert_news_articles(new_df)
            logging.info(f"Historical collection: {len(articles)} processed → {inserted} new saved")
            return inserted
        
        logging.info(f"Historical collection: {len(articles)} processed → 0 new (all duplicates)")
        return 0
    
    def collect_historical_for_gaps(self):
        """Main method to collect historical news data for identified gaps."""
        logging.info("=== Starting Historical News Collection ===")
        
        # Collect articles using multiple strategies
        articles = self.collect_targeted_historical_search(self.target_months)
        
        # Save to database
        saved_count = self.save_historical_articles(articles)
        
        logging.info(f"=== Historical News Collection Complete ===")
        logging.info(f"Total collected: {len(articles)} articles")
        logging.info(f"Total saved: {saved_count} new entries")
        
        return saved_count

def main():
    """Main function for historical news collection."""
    collector = HistoricalNewsCollector()
    total = collector.collect_historical_for_gaps()
    print(f"Historical news collection complete. Collected {total} new articles.")

if __name__ == "__main__":
    main()