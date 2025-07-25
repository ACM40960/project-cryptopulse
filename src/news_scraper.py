# src/news_scraper.py

"""
News scraper for crypto news sites with duplicate skipping.
Sites: CoinDesk, CoinTelegraph, Decrypt
"""
import os
import time
import logging
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
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
        logging.FileHandler('logs/news_scraper.log'),
        logging.StreamHandler()
    ]
)

class NewsScraper:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # News sources configuration
        self.sources = {
            'cointelegraph': {
                'base_url': 'https://cointelegraph.com',
                'search_paths': ['/tags/ethereum', '/news/', '/analysis/', '/tags/defi', '/tags/altcoin'],
                'historical_paths': ['/search?query=ethereum&from=2024-01-01', '/search?query=DeFi&from=2024-01-01'],
                'article_selector': '.post-card-inline__title-link, .posts-listing__item a, .search-post a',
                'title_selector': 'h1, .post__title',
                'content_selector': '.post-content p, .post__content p',
                'date_selector': '.post__date, .post-meta time'
            },
            'coindesk': {
                'base_url': 'https://www.coindesk.com',
                'search_paths': ['/tag/ethereum/', '/markets/', '/tech/', '/policy/', '/business/'],
                'article_selector': 'h3 a, h4 a, .card-title a, .headline a',
                'title_selector': 'h1, .headline',
                'content_selector': '.at-content, .entry-content, article p',
                'date_selector': '.metadata time, .published-date, time'
            },
            'decrypt': {
                'base_url': 'https://decrypt.co',
                'search_paths': ['/learn/category/ethereum', '/news/', '/', '/defi/', '/nft/'],
                'article_selector': '.PostCard a, .post-title a, .news-item a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .content p',
                'date_selector': '.post-meta time, .date'
            },
            # EXPANDED NEWS SOURCES - More accessible sites
            'bitcoinist': {
                'base_url': 'https://bitcoinist.com',
                'search_paths': ['/ethereum/', '/defi/', '/altcoins/', '/news/'],
                'article_selector': 'h3 a, .post-title a, article a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .entry-content p',
                'date_selector': '.post-meta time, .date'
            },
            'newsbtc': {
                'base_url': 'https://www.newsbtc.com',
                'search_paths': ['/ethereum/', '/altcoins/', '/defi/', '/analysis/'],
                'article_selector': 'h3 a, .post-title a, .news-item a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .entry-content p',
                'date_selector': '.post-meta time, .date'
            },
            'ambcrypto': {
                'base_url': 'https://ambcrypto.com',
                'search_paths': ['/ethereum/', '/defi/', '/altcoins/', '/news/'],
                'article_selector': 'h3 a, .post-title a, .card-title a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .entry-content p',
                'date_selector': '.post-meta time, .published-date'
            },
            'coinjournal': {
                'base_url': 'https://coinjournal.net',
                'search_paths': ['/ethereum/', '/defi/', '/altcoins/', '/news/'],
                'article_selector': 'h3 a, .post-title a, .news-title a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .article-content p',
                'date_selector': '.post-meta time, .date'
            },
            'coinspeaker': {
                'base_url': 'https://www.coinspeaker.com',
                'search_paths': ['/category/ethereum/', '/category/defi/', '/news/'],
                'article_selector': 'h3 a, .post-title a, .news-item a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .entry-content p',
                'date_selector': '.post-meta time, .date'
            },
            'cryptonews': {
                'base_url': 'https://cryptonews.com',
                'search_paths': ['/news/ethereum/', '/news/defi/', '/news/altcoins/'],
                'article_selector': 'h3 a, .post-title a, .news-title a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .article-content p',
                'date_selector': '.post-meta time, .date'
            },
            'cryptopotato': {
                'base_url': 'https://cryptopotato.com',
                'search_paths': ['/ethereum/', '/defi/', '/altcoins/', '/daily-overview/'],
                'article_selector': 'h3 a, .post-title a, .card-title a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .entry-content p',
                'date_selector': '.post-meta time, .published-date'
            },
            'u_today': {
                'base_url': 'https://u.today',
                'search_paths': ['/ethereum', '/defi', '/altcoins'],
                'article_selector': 'h3 a, .post-title a, .news-item a',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-content p, .article-body p',
                'date_selector': '.post-meta time, .date'
            }
        }
    
    def get_article_links(self, source_name, max_articles=50):
        """Get article links from source homepage and category pages."""
        source = self.sources[source_name]
        links = set()
        
        for path in source['search_paths']:
            try:
                url = urljoin(source['base_url'], path)
                logging.info(f"Scraping links from: {url}")
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                article_links = soup.select(source['article_selector'])
                
                for link in article_links[:max_articles//len(source['search_paths'])]:
                    href = link.get('href')
                    if href:
                        if href.startswith('/'):
                            href = urljoin(source['base_url'], href)
                        links.add(href)
                
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                logging.warning(f"Error getting links from {url}: {e}")
                continue
        
        return list(links)[:max_articles]
    
    def scrape_article(self, url, source_name):
        """Scrape individual article content."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            source = self.sources[source_name]
            
            # Extract title
            title_elem = soup.select_one(source['title_selector'])
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract content
            content_elems = soup.select(source['content_selector'])
            content = " ".join([p.get_text(strip=True) for p in content_elems[:10]])  # First 10 paragraphs
            
            # Extract date
            date_elem = soup.select_one(source['date_selector'])
            published_at = None
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                try:
                    # Try common date formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%b %d, %Y']:
                        try:
                            published_at = datetime.strptime(date_text[:19], fmt)
                            break
                        except ValueError:
                            continue
                except:
                    published_at = datetime.now()  # Fallback to current time
            
            if not published_at:
                published_at = datetime.now()
            
            # Generate unique ID from URL
            article_id = hashlib.md5(url.encode()).hexdigest()
            
            # Filter for crypto relevance
            text = (title + " " + content).lower()
            crypto_keywords = ['ethereum', 'eth', 'crypto', 'bitcoin', 'blockchain', 'defi', 'nft', 'trading']
            if not any(keyword in text for keyword in crypto_keywords):
                return None
            
            return {
                'id': article_id,
                'source': source_name,
                'title': title,
                'content': content,
                'published_at': published_at,
                'url': url
            }
            
        except Exception as e:
            logging.warning(f"Error scraping article {url}: {e}")
            return None
    
    def scrape_source(self, source_name, max_articles=50):
        """Scrape articles from a specific news source."""
        logging.info(f"Starting scrape of {source_name}")
        
        # Get article links
        links = self.get_article_links(source_name, max_articles)
        logging.info(f"Found {len(links)} article links from {source_name}")
        
        articles = []
        for i, url in enumerate(links):
            # Check if article already exists
            article_id = hashlib.md5(url.encode()).hexdigest()
            if self.db.record_exists('news_articles', article_id):
                logging.debug(f"Skipped existing article {article_id}")
                continue
            
            article = self.scrape_article(url, source_name)
            if article:
                articles.append(article)
                logging.info(f"Scraped article {i+1}/{len(links)}: {article['title'][:50]}...")
            
            time.sleep(1)  # Rate limiting
        
        # Save to database
        if articles:
            df = pd.DataFrame(articles)
            inserted = self.db.insert_news_articles(df)
            logging.info(f"{source_name}: {len(articles)} processed → {inserted} new saved")
            return inserted
        
        return 0
    
    def scrape_all_sources(self, max_articles_per_source=30):
        """Scrape all configured news sources."""
        total = 0
        for source_name in self.sources.keys():
            total += self.scrape_source(source_name, max_articles_per_source)
            time.sleep(5)  # Pause between sources
        
        logging.info(f"Total new news articles: {total}")
        return total

if __name__ == "__main__":
    scraper = NewsScraper()
    count = scraper.scrape_all_sources(max_articles_per_source=25)
    print(f"Total new news articles collected: {count}")