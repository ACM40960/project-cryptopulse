#!/usr/bin/env python3
# src/alternative_historical_collector.py

"""
Alternative historical data collection using multiple strategies
when primary APIs are unavailable.
"""
import os
import sys
import time
import logging
import requests
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote, urljoin
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.append('src')
from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alternative_historical_collector.log'),
        logging.StreamHandler()
    ]
)

class AlternativeHistoricalCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Alternative data sources
        self.data_sources = {
            'reddit_alternative': self.collect_reddit_alternative,
            'twitter_historical': self.collect_twitter_historical,
            'news_wayback': self.collect_news_wayback,
            'forum_archives': self.collect_forum_archives,
            'github_historical': self.collect_github_historical
        }
        
        # Priority historical periods
        self.priority_periods = [
            '2018-02', '2018-03', '2018-04', '2020-03', '2020-12',
            '2021-05', '2021-11', '2022-06', '2022-11'
        ]
    
    def collect_reddit_alternative(self):
        """Alternative Reddit collection using web scraping and cached data."""
        logging.info("Starting alternative Reddit collection...")
        posts = []
        
        # Use Google cache and archived Reddit pages
        search_queries = [
            'site:reddit.com/r/ethereum "eth" OR "ethereum" 2018',
            'site:reddit.com/r/cryptocurrency "ethereum price" 2018',
            'site:reddit.com/r/ethtrader "crypto winter" 2018',
            'site:reddit.com/r/ethereum "defi summer" 2020',
            'site:reddit.com/r/ethereum "eth 2.0" OR "proof of stake" 2021'
        ]
        
        for query in search_queries:
            try:
                # Use Google search to find cached Reddit pages
                google_url = f"https://www.google.com/search?q={quote(query)}&num=50"
                response = self.session.get(google_url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract Reddit URLs from search results
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if href and 'reddit.com' in href and '/comments/' in href:
                            try:
                                # Extract Reddit post info from cached pages
                                post_data = self.extract_reddit_post_data(href)
                                if post_data:
                                    posts.append(post_data)
                            except Exception as e:
                                logging.debug(f"Failed to extract Reddit post: {e}")
                                continue
                
                time.sleep(5)  # Respectful delay for Google
                
            except Exception as e:
                logging.warning(f"Google search failed for query '{query}': {e}")
                continue
        
        return posts[:100]  # Limit results
    
    def extract_reddit_post_data(self, url):
        """Extract post data from Reddit URL."""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic post info (simplified extraction)
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Generate ID from URL
            post_id = hashlib.md5(url.encode()).hexdigest()[:10]
            
            # Check for Ethereum relevance
            if any(term in title.lower() for term in ['ethereum', 'eth', 'defi', 'smart contract']):
                return {
                    'id': f"alt_reddit_{post_id}",
                    'source': 'reddit_alternative',
                    'title': title,
                    'content': '',  # Content extraction would need more sophisticated parsing
                    'published_at': datetime.now() - timedelta(days=365*2),  # Approximate historical date
                    'url': url
                }
                
        except Exception as e:
            logging.debug(f"Failed to extract Reddit post from {url}: {e}")
            return None
    
    def collect_twitter_historical(self):
        """Collect historical Twitter data using alternative methods."""
        logging.info("Starting alternative Twitter collection...")
        tweets = []
        
        # Alternative Twitter sources
        sources = [
            'https://web.archive.org/web/20180301000000*/twitter.com/VitalikButerin',
            'https://web.archive.org/web/20201201000000*/twitter.com/ethereum',
            'https://web.archive.org/web/20211101000000*/twitter.com/BTCTN'
        ]
        
        for source_url in sources:
            try:
                response = self.session.get(source_url, timeout=20)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract archived tweet data (simplified)
                    for tweet_elem in soup.find_all('div', class_='tweet'):
                        try:
                            text_elem = tweet_elem.find('p', class_='tweet-text')
                            if text_elem:
                                text = text_elem.get_text(strip=True)
                                
                                # Check for Ethereum relevance
                                if any(term in text.lower() for term in ['ethereum', 'eth', 'defi']):
                                    tweet_id = hashlib.md5(text.encode()).hexdigest()[:10]
                                    
                                    tweets.append({
                                        'id': f"alt_twitter_{tweet_id}",
                                        'source': 'twitter_alternative',
                                        'title': text[:100],
                                        'content': text,
                                        'published_at': datetime.now() - timedelta(days=365),
                                        'url': source_url
                                    })
                        except Exception as e:
                            continue
                
                time.sleep(10)  # Wayback Machine rate limiting
                
            except Exception as e:
                logging.warning(f"Twitter alternative collection failed for {source_url}: {e}")
                continue
        
        return tweets[:50]  # Limit results
    
    def collect_news_wayback(self):
        """Collect historical news using Wayback Machine."""
        logging.info("Starting Wayback Machine news collection...")
        articles = []
        
        # Target historical snapshots of major crypto news sites
        wayback_targets = [
            ('20180201', 'coindesk.com/markets/ethereum'),
            ('20180301', 'cointelegraph.com/tags/ethereum'),
            ('20200301', 'decrypt.co/learn/category/ethereum'),
            ('20201201', 'theblock.co/category/crypto'),
            ('20210501', 'bitcoinist.com/ethereum')
        ]
        
        for date, site in wayback_targets:
            try:
                wayback_url = f"https://web.archive.org/web/{date}000000/{site}"
                response = self.session.get(wayback_url, timeout=20)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract article links from archived pages
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        text = link.get_text(strip=True)
                        
                        if (href and len(text) > 20 and 
                            any(term in text.lower() for term in ['ethereum', 'eth', 'defi', 'crypto'])):
                            
                            # Create article entry
                            article_id = hashlib.md5(href.encode()).hexdigest()[:10]
                            articles.append({
                                'id': f"wayback_{article_id}",
                                'source': f'wayback_{site.split(".")[0]}',
                                'title': text,
                                'content': '',  # Would need individual page scraping
                                'published_at': datetime.strptime(date, '%Y%m%d'),
                                'url': wayback_url
                            })
                
                time.sleep(15)  # Conservative rate limiting for Wayback Machine
                
            except Exception as e:
                logging.warning(f"Wayback collection failed for {site}: {e}")
                continue
        
        return articles[:100]  # Limit results
    
    def collect_forum_archives(self):
        """Collect from crypto forum archives."""
        logging.info("Starting forum archive collection...")
        posts = []
        
        # Target crypto forums with archives
        forum_targets = [
            'https://bitcointalk.org/index.php?board=160.0',  # Ethereum board
            'https://ethereum-magicians.org/c/working-groups/5',
            'https://forum.ethereum.org/categories/ethereum-discussion'
        ]
        
        for forum_url in forum_targets:
            try:
                response = self.session.get(forum_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract forum post titles and links
                    for link in soup.find_all('a', href=True):
                        text = link.get_text(strip=True)
                        href = link.get('href')
                        
                        if (len(text) > 10 and href and
                            any(term in text.lower() for term in ['ethereum', 'eth', 'eip', 'smart contract'])):
                            
                            post_id = hashlib.md5(href.encode()).hexdigest()[:10]
                            posts.append({
                                'id': f"forum_{post_id}",
                                'source': 'crypto_forums',
                                'title': text,
                                'content': '',
                                'published_at': datetime.now() - timedelta(days=500),
                                'url': urljoin(forum_url, href)
                            })
                
                time.sleep(5)
                
            except Exception as e:
                logging.warning(f"Forum collection failed for {forum_url}: {e}")
                continue
        
        return posts[:50]  # Limit results
    
    def collect_github_historical(self):
        """Collect historical GitHub discussions and issues."""
        logging.info("Starting GitHub historical collection...")
        issues = []
        
        # Ethereum-related repositories
        repos = [
            'ethereum/ethereum',
            'ethereum/EIPs',
            'ethereum/solidity',
            'ethereum/go-ethereum'
        ]
        
        for repo in repos:
            try:
                # Use GitHub search API for historical issues
                search_url = f"https://api.github.com/search/issues"
                params = {
                    'q': f'repo:{repo} ethereum created:2018-01-01..2018-12-31',
                    'sort': 'created',
                    'per_page': 30
                }
                
                response = self.session.get(search_url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for issue in data.get('items', []):
                        title = issue.get('title', '')
                        body = issue.get('body', '') or ''
                        created_at = issue.get('created_at', '')
                        html_url = issue.get('html_url', '')
                        
                        if any(term in (title + body).lower() for term in ['ethereum', 'eth', 'gas', 'block']):
                            issue_id = str(issue.get('id', ''))
                            
                            try:
                                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            except:
                                created_date = datetime.now() - timedelta(days=365*2)
                            
                            issues.append({
                                'id': f"github_{issue_id}",
                                'source': f'github_{repo.replace("/", "_")}',
                                'title': title,
                                'content': body[:500],  # Limit content length
                                'published_at': created_date.replace(tzinfo=None),
                                'url': html_url
                            })
                
                time.sleep(3)  # GitHub API rate limiting
                
            except Exception as e:
                logging.warning(f"GitHub collection failed for {repo}: {e}")
                continue
        
        return issues[:100]  # Limit results
    
    def save_alternative_data(self, data, source_name):
        """Save collected alternative data to database."""
        if not data:
            logging.info(f"No data collected from {source_name}")
            return 0
        
        # Use news_articles table for all alternative sources
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset=['id'])
        
        # Filter for new records
        new_articles = []
        for _, item in df.iterrows():
            if not self.db.record_exists('news_articles', item['id']):
                new_articles.append(item.to_dict())
        
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            inserted = self.db.insert_news_articles(new_df)
            logging.info(f"{source_name}: {len(data)} collected → {inserted} new saved")
            return inserted
        
        logging.info(f"{source_name}: {len(data)} collected → 0 new (all duplicates)")
        return 0
    
    def collect_all_alternative_sources(self):
        """Run all alternative collection methods."""
        logging.info("=== Starting Alternative Historical Collection ===")
        total_collected = 0
        
        for source_name, collect_method in self.data_sources.items():
            try:
                logging.info(f"Running {source_name} collection...")
                data = collect_method()
                saved = self.save_alternative_data(data, source_name)
                total_collected += saved
                
                if saved > 0:
                    logging.info(f"✅ {source_name}: {saved} new entries")
                else:
                    logging.info(f"⚪ {source_name}: No new entries")
                    
            except Exception as e:
                logging.error(f"❌ {source_name} failed: {e}")
                continue
        
        logging.info(f"=== Alternative Collection Complete ===")
        logging.info(f"Total new entries: {total_collected}")
        
        return total_collected

def main():
    """Main function for alternative historical collection."""
    collector = AlternativeHistoricalCollector()
    total = collector.collect_all_alternative_sources()
    print(f"Alternative historical collection complete. Collected {total} new entries.")

if __name__ == "__main__":
    main()