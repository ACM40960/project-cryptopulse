#!/usr/bin/env python3
# targeted_rss_boost.py

"""
Targeted RSS and News API collection for weak periods to efficiently boost daily density.
Focus on 2022-2023 timeframe with comprehensive crypto news sources.
"""
import sys
import os
import time
import logging
import sqlite3
import requests
import hashlib
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/targeted_rss_boost.log'),
        logging.StreamHandler()
    ]
)

class TargetedRSSBoost:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Comprehensive RSS feeds for crypto news
        self.crypto_rss_feeds = [
            # Major crypto news sites
            'https://cointelegraph.com/rss',
            'https://decrypt.co/feed',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://thedefiant.io/feed',
            'https://blockworks.co/feed',
            'https://bitcoinmagazine.com/.rss/full/',
            'https://u.today/rss',
            'https://cryptonews.com/news/feed/',
            'https://cryptoslate.com/feed/',
            'https://www.newsbtc.com/feed/',
            
            # Ethereum-focused feeds
            'https://blog.ethereum.org/feed.xml',
            'https://consensys.net/blog/feed/',
            'https://medium.com/feed/@VitalikButerin',
            'https://medium.com/feed/ethereum-foundation',
            
            # Financial news with crypto coverage
            'https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml&size=100&tagName=crypto',
            'https://finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US',
            'https://www.marketwatch.com/rss/topstories',
            
            # Reddit RSS feeds
            'https://www.reddit.com/r/ethereum.rss',
            'https://www.reddit.com/r/CryptoCurrency.rss',
            'https://www.reddit.com/r/ethtrader.rss',
            'https://www.reddit.com/r/defi.rss',
            
            # Additional sources
            'https://ambcrypto.com/feed/',
            'https://cryptopotato.com/feed/',
            'https://beincrypto.com/feed/',
            'https://www.crypto-news-flash.com/feed/',
            'https://cryptobriefing.com/feed/'
        ]
        
        # News API sources (requires API keys but many have free tiers)
        self.news_apis = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'currents': 'https://api.currentsapi.services/v1/search',
            'gnews': 'https://gnews.io/api/v4/search'
        }
    
    def get_weak_periods(self):
        """Get specific months/weeks that need boosting."""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        WITH weekly_data AS (
            SELECT 
                strftime('%Y-%m', datetime(created_utc, 'unixepoch')) as month,
                strftime('%Y-W%W', datetime(created_utc, 'unixepoch')) as week,
                COUNT(*) as reddit_count
            FROM reddit_posts 
            WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01' 
            AND datetime(created_utc, 'unixepoch') < '2024-01-01'
            GROUP BY strftime('%Y-W%W', datetime(created_utc, 'unixepoch'))
        ),
        weekly_news AS (
            SELECT 
                strftime('%Y-%m', datetime(published_at, 'unixepoch')) as month,
                strftime('%Y-W%W', datetime(published_at, 'unixepoch')) as week,
                COUNT(*) as news_count
            FROM news_articles 
            WHERE datetime(published_at, 'unixepoch') >= '2022-01-01' 
            AND datetime(published_at, 'unixepoch') < '2024-01-01'
            GROUP BY strftime('%Y-W%W', datetime(published_at, 'unixepoch'))
        )
        SELECT 
            COALESCE(wd.week, wn.week) as week,
            COALESCE(wd.month, wn.month) as month,
            COALESCE(wd.reddit_count, 0) as reddit,
            COALESCE(wn.news_count, 0) as news,
            COALESCE(wd.reddit_count, 0) + COALESCE(wn.news_count, 0) as total
        FROM weekly_data wd
        FULL OUTER JOIN weekly_news wn ON wd.week = wn.week
        WHERE COALESCE(wd.reddit_count, 0) + COALESCE(wn.news_count, 0) < 50
        ORDER BY total
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logging.info(f"Found {len(df)} weak weeks needing RSS boost")
        return df
    
    def scrape_rss_feed_comprehensive(self, rss_url, target_timeframe='2022-2023'):
        """Comprehensively scrape RSS feed with date filtering."""
        articles = []
        
        try:
            logging.info(f"Scraping RSS: {rss_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                logging.warning(f"No entries found in RSS feed: {rss_url}")
                return articles
            
            for entry in feed.entries:
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            pub_date = datetime(*entry.published_parsed[:6])
                        except:
                            # Try alternative date fields
                            if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Filter for target timeframe
                    if target_timeframe == '2022-2023':
                        if not (pub_date.year == 2022 or pub_date.year == 2023):
                            continue
                    
                    # Check for crypto/Ethereum relevance
                    text_content = (title + " " + summary).lower()
                    crypto_keywords = [
                        'ethereum', 'eth', 'crypto', 'cryptocurrency', 'bitcoin', 'blockchain',
                        'defi', 'decentralized', 'smart contract', 'nft', 'web3', 'dapp',
                        'yield farming', 'liquidity', 'staking', 'mining', 'trading',
                        'altcoin', 'token', 'dao', 'consensus', 'proof of stake'
                    ]
                    
                    if any(keyword in text_content for keyword in crypto_keywords):
                        article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                        month_key = pub_date.strftime('%Y-%m')
                        
                        # Extract source name from RSS URL
                        source_name = rss_url.split('//')[1].split('/')[0].replace('www.', '')
                        
                        articles.append({
                            'id': f"rss_{source_name}_{month_key}_{article_id}",
                            'source': f'rss_{source_name}_{month_key}',
                            'title': title,
                            'content': summary,
                            'published_at': pub_date,
                            'url': link
                        })
                    
                except Exception as e:
                    logging.debug(f"Failed to parse RSS entry: {e}")
                    continue
            
            logging.info(f"RSS {rss_url}: {len(articles)} relevant articles found")
            return articles
            
        except Exception as e:
            logging.warning(f"RSS scraping failed for {rss_url}: {e}")
            return articles
    
    def historical_rss_search(self, weak_periods):
        """Search for historical RSS content using Google and Wayback Machine."""
        articles = []
        
        # Target the weakest periods
        top_weak_periods = weak_periods.head(10)
        
        for _, period in top_weak_periods.iterrows():
            week = period['week']
            month = period['month']
            
            try:
                # Convert week to approximate date range
                year = int(week.split('-')[0])
                week_num = int(week.split('W')[1])
                
                # Calculate approximate start date of week
                start_date = datetime(year, 1, 1) + timedelta(weeks=week_num-1)
                end_date = start_date + timedelta(days=7)
                
                # Search for crypto news in that week using Google
                search_terms = [
                    f'ethereum price news {start_date.strftime("%Y-%m-%d")}',
                    f'crypto market {start_date.strftime("%Y-%m")}',
                    f'defi news {start_date.strftime("%Y-%m")}'
                ]
                
                for term in search_terms:
                    try:
                        # Google News search for specific date range
                        encoded_query = quote(f'{term} after:{start_date.strftime("%Y-%m-%d")} before:{end_date.strftime("%Y-%m-%d")}')
                        google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                        
                        feed = feedparser.parse(google_news_url)
                        
                        for entry in feed.entries[:20]:  # Limit per search
                            try:
                                title = entry.get('title', '')
                                link = entry.get('link', '')
                                summary = entry.get('summary', '')
                                
                                # Parse date
                                pub_date = start_date  # Default to week start
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    pub_date = datetime(*entry.published_parsed[:6])
                                
                                # Check relevance
                                text = (title + " " + summary).lower()
                                if any(keyword in text for keyword in ['ethereum', 'crypto', 'blockchain', 'defi']):
                                    article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                    
                                    articles.append({
                                        'id': f"historical_news_{week}_{article_id}",
                                        'source': f'historical_news_{month}',
                                        'title': title,
                                        'content': summary,
                                        'published_at': pub_date,
                                        'url': link
                                    })
                                
                            except Exception as e:
                                logging.debug(f"Failed to parse historical entry: {e}")
                                continue
                        
                        time.sleep(2)  # Rate limiting
                        
                    except Exception as e:
                        logging.warning(f"Historical search failed for {term}: {e}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Historical RSS search failed for period {week}: {e}")
                continue
        
        return articles
    
    def parallel_rss_collection(self, max_workers=6):
        """Collect from multiple RSS feeds in parallel."""
        logging.info("Starting parallel RSS collection...")
        
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit RSS scraping tasks
            future_to_url = {
                executor.submit(self.scrape_rss_feed_comprehensive, url): url 
                for url in self.crypto_rss_feeds
            }
            
            # Collect results
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    logging.info(f"✅ {url}: {len(articles)} articles collected")
                except Exception as e:
                    logging.error(f"❌ {url} failed: {e}")
        
        return all_articles
    
    def save_rss_data(self, articles, source_name):
        """Save RSS articles to database."""
        if not articles:
            logging.info(f"No articles collected from {source_name}")
            return 0
        
        # Remove duplicates
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['id'])
        
        # Filter for new records
        new_articles = []
        for _, item in df.iterrows():
            if not self.db.record_exists('news_articles', item['id']):
                new_articles.append(item.to_dict())
        
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            inserted = self.db.insert_news_articles(new_df)
            logging.info(f"{source_name}: {len(articles)} collected → {inserted} new saved")
            return inserted
        
        logging.info(f"{source_name}: {len(articles)} collected → 0 new (all duplicates)")
        return 0
    
    def targeted_rss_campaign(self):
        """Main RSS boost campaign."""
        logging.info("=== TARGETED RSS BOOST CAMPAIGN ===")
        
        start_time = datetime.now()
        
        # Get weak periods that need boosting
        weak_periods = self.get_weak_periods()
        
        # Phase 1: Parallel RSS collection from all feeds
        logging.info("Phase 1: Comprehensive RSS collection")
        rss_articles = self.parallel_rss_collection()
        rss_saved = self.save_rss_data(rss_articles, "comprehensive_rss")
        
        # Phase 2: Historical search for weak periods
        logging.info("Phase 2: Historical RSS search for weak periods")
        historical_articles = self.historical_rss_search(weak_periods)
        historical_saved = self.save_rss_data(historical_articles, "historical_rss")
        
        total_saved = rss_saved + historical_saved
        elapsed = datetime.now() - start_time
        
        # Summary
        logging.info("=== RSS BOOST CAMPAIGN COMPLETE ===")
        logging.info(f"RSS feeds processed: {len(self.crypto_rss_feeds)}")
        logging.info(f"Weak periods targeted: {len(weak_periods)}")
        logging.info(f"Total new articles: {total_saved}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_saved

def main():
    """Main function for targeted RSS boost."""
    booster = TargetedRSSBoost()
    
    print(f"🚀 TARGETED RSS BOOST CAMPAIGN")
    print(f"📡 Feeds: {len(booster.crypto_rss_feeds)} comprehensive RSS sources")
    print(f"🎯 Target: Weak periods in 2022-2023 timeframe")
    print(f"💡 Strategy: Parallel collection + historical search")
    
    total = booster.targeted_rss_campaign()
    
    print(f"\n🎉 RSS BOOST COMPLETE!")
    print(f"📰 Added {total} new articles")
    print(f"🎯 Next: Verify improved daily density")

if __name__ == "__main__":
    main()