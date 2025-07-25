# src/historical_news_scraper.py

"""
Historical news scraper for crypto news with archive support.
Targets: Archive.org, Google News, RSS feeds, Financial news sites
"""
import os
import time
import logging
import hashlib
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, quote

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
        logging.FileHandler('logs/historical_news_scraper.log'),
        logging.StreamHandler()
    ]
)

class HistoricalNewsScraper:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Historical search sources
        self.historical_sources = {
            'wayback_machine': {
                'base_url': 'https://web.archive.org/cdx/search/cdx',
                'params': {
                    'url': 'cointelegraph.com/ethereum*',
                    'matchType': 'prefix',
                    'collapse': 'urlkey',
                    'output': 'json',
                    'fl': 'original,timestamp',
                    'filter': 'statuscode:200',
                    'limit': 1000
                }
            },
            'google_news_rss': {
                'base_url': 'https://news.google.com/rss/search',
                'search_terms': ['ethereum', 'ETH+price', 'defi', 'smart+contracts'],
                'date_ranges': [
                    ('2020-2021', '2020-01-01', '2022-01-01'),
                    ('2022', '2022-01-01', '2023-01-01'),
                    ('2023', '2023-01-01', '2024-01-01'),
                    ('2024', '2024-01-01', '2025-01-01')
                ]
            },
            'financial_news': {
                'sources': [
                    {
                        'name': 'yahoo_finance',
                        'search_url': 'https://finance.yahoo.com/news/',
                        'ethereum_tag': 'ethereum'
                    },
                    {
                        'name': 'marketwatch',
                        'search_url': 'https://www.marketwatch.com/search',
                        'query_param': 'q=ethereum'
                    }
                ]
            }
        }
        
        # Expanded crypto news sources with RSS feeds
        self.rss_sources = {
            'coindesk_rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph_rss': 'https://cointelegraph.com/rss',
            'decrypt_rss': 'https://decrypt.co/feed',
            'theblock_rss': 'https://www.theblockcrypto.com/rss.xml',
            'cryptoslate_rss': 'https://cryptoslate.com/feed/',
            'bitcoinist_rss': 'https://bitcoinist.com/feed/',
            'newsbtc_rss': 'https://www.newsbtc.com/feed/'
        }
    
    def scrape_wayback_machine(self, target_sites, year_range=(2020, 2025)):
        """Scrape historical articles from Wayback Machine."""
        logging.info("Starting Wayback Machine historical collection")
        articles = []
        
        target_sites = target_sites or [
            'cointelegraph.com/ethereum',
            'coindesk.com/ethereum',
            'decrypt.co/ethereum'
        ]
        
        for site in target_sites:
            try:
                # Get archived URLs
                params = {
                    'url': f'{site}*',
                    'matchType': 'prefix',
                    'collapse': 'urlkey',
                    'output': 'json',
                    'fl': 'original,timestamp',
                    'filter': 'statuscode:200',
                    'from': f'{year_range[0]}0101',
                    'to': f'{year_range[1]}1231',
                    'limit': 500
                }
                
                response = self.session.get(
                    'https://web.archive.org/cdx/search/cdx',
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        # Skip header row
                        for row in data[1:]:
                            original_url, timestamp = row[0], row[1]
                            
                            # Convert timestamp to datetime
                            try:
                                archived_date = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
                                
                                # Try to scrape the archived version
                                archived_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
                                article = self.scrape_archived_article(archived_url, original_url, archived_date)
                                if article:
                                    articles.append(article)
                                    
                            except Exception as e:
                                logging.debug(f"Failed to process archived URL {original_url}: {e}")
                                continue
                                
                        time.sleep(5)  # Respectful delay
                        
            except Exception as e:
                logging.warning(f"Wayback Machine search failed for {site}: {e}")
                continue
        
        return self.save_articles(articles, "wayback_machine")
    
    def scrape_archived_article(self, archived_url, original_url, date):
        """Scrape content from an archived article."""
        try:
            response = self.session.get(archived_url, timeout=20)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_selectors = ['h1', '.post-title', '.headline', 'title']
            title = ""
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content
            content_selectors = ['.post-content p', '.article-content p', 'article p', '.content p']
            content = ""
            for selector in content_selectors:
                content_elems = soup.select(selector)
                if content_elems:
                    content = " ".join([p.get_text(strip=True) for p in content_elems[:10]])
                    break
            
            # Filter for crypto relevance
            text = (title + " " + content).lower()
            crypto_keywords = ['ethereum', 'eth', 'crypto', 'bitcoin', 'blockchain', 'defi', 'smart contract']
            if not any(keyword in text for keyword in crypto_keywords):
                return None
            
            # Generate unique ID
            article_id = hashlib.md5(original_url.encode()).hexdigest()
            
            return {
                'id': article_id,
                'source': 'wayback_archive',
                'title': title,
                'content': content,
                'published_at': date,
                'url': original_url
            }
            
        except Exception as e:
            logging.debug(f"Failed to scrape archived article {archived_url}: {e}")
            return None
    
    def scrape_rss_feeds(self, days_back=365):
        """Scrape historical articles from RSS feeds."""
        logging.info("Starting RSS feed historical collection")
        articles = []
        
        for source_name, rss_url in self.rss_sources.items():
            try:
                logging.info(f"Scraping RSS: {source_name}")
                response = self.session.get(rss_url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')
                    
                    for item in items[:50]:  # Limit per source
                        try:
                            title = item.find('title').get_text() if item.find('title') else ""
                            link = item.find('link').get_text() if item.find('link') else ""
                            pub_date = item.find('pubDate').get_text() if item.find('pubDate') else ""
                            description = item.find('description').get_text() if item.find('description') else ""
                            
                            # Parse date
                            published_at = datetime.now()
                            if pub_date:
                                try:
                                    published_at = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                                except:
                                    try:
                                        published_at = datetime.strptime(pub_date[:25], '%a, %d %b %Y %H:%M:%S')
                                    except:
                                        pass
                            
                            # Filter for crypto relevance and recency
                            text = (title + " " + description).lower()
                            crypto_keywords = ['ethereum', 'eth', 'crypto', 'defi', 'blockchain']
                            if not any(keyword in text for keyword in crypto_keywords):
                                continue
                                
                            # Skip if too old
                            if (datetime.now() - published_at).days > days_back:
                                continue
                            
                            # Generate unique ID
                            article_id = hashlib.md5(link.encode()).hexdigest()
                            
                            # Check if already exists
                            if self.db.record_exists('news_articles', article_id):
                                continue
                            
                            articles.append({
                                'id': article_id,
                                'source': source_name,
                                'title': title,
                                'content': description,
                                'published_at': published_at,
                                'url': link
                            })
                            
                        except Exception as e:
                            logging.debug(f"Failed to process RSS item: {e}")
                            continue
                            
                time.sleep(3)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"RSS scraping failed for {source_name}: {e}")
                continue
        
        return self.save_articles(articles, "rss_feeds")
    
    def scrape_google_news_search(self, search_terms=None, months_back=12):
        """Search Google News for historical crypto articles."""
        logging.info("Starting Google News search collection")
        articles = []
        
        search_terms = search_terms or [
            'ethereum price', 'ETH cryptocurrency', 'ethereum defi',
            'smart contracts ethereum', 'ethereum upgrade', 'vitalik buterin'
        ]
        
        # Generate date ranges
        end_date = datetime.now()
        for i in range(months_back):
            start_month = end_date - timedelta(days=30 * (i + 1))
            end_month = end_date - timedelta(days=30 * i)
            
            for term in search_terms:
                try:
                    # Format dates for Google News
                    after_date = start_month.strftime('%Y-%m-%d')
                    before_date = end_month.strftime('%Y-%m-%d')
                    
                    # Google News RSS search with date range
                    search_query = f"{term} after:{after_date} before:{before_date}"
                    encoded_query = quote(search_query)
                    
                    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                    
                    response = self.session.get(url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'xml')
                        items = soup.find_all('item')
                        
                        for item in items[:10]:  # Limit per search
                            try:
                                title = item.find('title').get_text() if item.find('title') else ""
                                link = item.find('link').get_text() if item.find('link') else ""
                                pub_date = item.find('pubDate').get_text() if item.find('pubDate') else ""
                                
                                # Generate unique ID
                                article_id = hashlib.md5(link.encode()).hexdigest()
                                
                                # Check if already exists
                                if self.db.record_exists('news_articles', article_id):
                                    continue
                                
                                # Parse date
                                published_at = datetime.now()
                                if pub_date:
                                    try:
                                        published_at = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                                    except:
                                        pass
                                
                                articles.append({
                                    'id': article_id,
                                    'source': 'google_news_search',
                                    'title': title,
                                    'content': title,  # Limited content from RSS
                                    'published_at': published_at,
                                    'url': link
                                })
                                
                            except Exception as e:
                                logging.debug(f"Failed to process Google News item: {e}")
                                continue
                                
                    time.sleep(5)  # Rate limiting for Google
                    
                except Exception as e:
                    logging.warning(f"Google News search failed for '{term}': {e}")
                    continue
        
        return self.save_articles(articles, "google_news_search")
    
    def save_articles(self, articles, source_type):
        """Save articles to database."""
        if not articles:
            logging.info(f"No articles collected from {source_type}")
            return 0
        
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['id'])
        
        inserted = self.db.insert_news_articles(df)
        logging.info(f"{source_type}: {len(articles)} processed → {inserted} new saved")
        return inserted
    
    def collect_all_historical(self):
        """Run all historical collection methods."""
        logging.info("=== Starting Comprehensive Historical News Collection ===")
        total = 0
        
        # RSS feeds (most reliable)
        total += self.scrape_rss_feeds(days_back=180)
        
        # Google News search (recent historical)
        total += self.scrape_google_news_search(months_back=6)
        
        # Wayback Machine (deep historical - optional, slower)
        # total += self.scrape_wayback_machine(None, (2022, 2024))
        
        logging.info(f"=== Historical Collection Complete: {total} new articles ===")
        return total

def main():
    """Main function for historical news collection."""
    scraper = HistoricalNewsScraper()
    total = scraper.collect_all_historical()
    print(f"Historical news collection complete. Collected {total} new articles.")

if __name__ == "__main__":
    main()