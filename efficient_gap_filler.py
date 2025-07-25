#!/usr/bin/env python3
# efficient_gap_filler.py

"""
Efficient gap-filling system focusing on most productive sources
to quickly achieve professional-grade daily density.
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

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/efficient_gap_filler.log'),
        logging.StreamHandler()
    ]
)

class EfficientGapFiller:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_low_density_months(self):
        """Get months that need density improvement."""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        WITH monthly_data AS (
            SELECT 
                strftime('%Y-%m', datetime(created_utc, 'unixepoch')) as month,
                COUNT(*) as reddit_count
            FROM reddit_posts 
            WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01' 
            AND datetime(created_utc, 'unixepoch') < '2024-01-01'
            GROUP BY strftime('%Y-%m', datetime(created_utc, 'unixepoch'))
        ),
        monthly_news AS (
            SELECT 
                strftime('%Y-%m', datetime(published_at, 'unixepoch')) as month,
                COUNT(*) as news_count
            FROM news_articles 
            WHERE datetime(published_at, 'unixepoch') >= '2022-01-01' 
            AND datetime(published_at, 'unixepoch') < '2024-01-01'
            GROUP BY strftime('%Y-%m', datetime(published_at, 'unixepoch'))
        )
        SELECT 
            COALESCE(md.month, mn.month) as month,
            COALESCE(md.reddit_count, 0) as reddit,
            COALESCE(mn.news_count, 0) as news,
            COALESCE(md.reddit_count, 0) + COALESCE(mn.news_count, 0) as total
        FROM monthly_data md
        FULL OUTER JOIN monthly_news mn ON md.month = mn.month
        WHERE COALESCE(md.reddit_count, 0) + COALESCE(mn.news_count, 0) < 300
        ORDER BY total
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def bulk_google_news_collection(self, target_months):
        """Efficient bulk Google News collection for multiple months."""
        logging.info(f"Starting bulk Google News collection for {len(target_months)} months")
        
        all_articles = []
        
        # Enhanced search strategies for 2022-2023
        search_strategies = [
            {
                'terms': ['ethereum price crash', 'ethereum bear market', 'ethereum merge'],
                'year_filter': '2022'
            },
            {
                'terms': ['ethereum recovery', 'ethereum etf', 'ethereum institutional'],
                'year_filter': '2023'
            },
            {
                'terms': ['defi ethereum', 'ethereum layer 2', 'ethereum staking'],
                'year_filter': '2022 OR 2023'
            },
            {
                'terms': ['ethereum hack', 'ethereum exploit', 'ethereum security'],
                'year_filter': '2022 OR 2023'
            }
        ]
        
        for strategy in search_strategies:
            for term in strategy['terms']:
                try:
                    # Create comprehensive search query
                    search_query = f'{term} {strategy["year_filter"]}'
                    encoded_query = quote(search_query)
                    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                    
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:30]:  # More entries per search
                        try:
                            title = entry.get('title', '')
                            link = entry.get('link', '')
                            summary = entry.get('summary', '')
                            
                            # Parse publication date
                            pub_date = datetime.now()
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            
                            # Check if in target timeframe
                            month_key = pub_date.strftime('%Y-%m')
                            if month_key not in [row['month'] for _, row in target_months.iterrows()]:
                                continue
                            
                            # Check for crypto relevance
                            text = (title + " " + summary).lower()
                            if any(keyword in text for keyword in ['ethereum', 'crypto', 'bitcoin', 'blockchain', 'defi']):
                                article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                
                                all_articles.append({
                                    'id': f"bulk_news_{month_key}_{article_id}",
                                    'source': f'bulk_news_{month_key}',
                                    'title': title,
                                    'content': summary,
                                    'published_at': pub_date,
                                    'url': link
                                })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse bulk news entry: {e}")
                            continue
                    
                    time.sleep(3)  # Rate limiting
                    
                except Exception as e:
                    logging.warning(f"Bulk Google News failed for '{term}': {e}")
                    continue
        
        return all_articles
    
    def targeted_reddit_enhancement(self, target_months):
        """Enhanced Reddit collection for specific months."""
        logging.info(f"Starting targeted Reddit enhancement for {len(target_months)} months")
        
        posts = []
        
        # Use Google to find more Reddit discussions
        for _, month_row in target_months.head(5).iterrows():  # Top 5 lowest months
            month = month_row['month']
            year, month_num = month.split('-')
            
            try:
                # Search for Reddit discussions in that month
                reddit_queries = [
                    f'site:reddit.com/r/ethereum "{year}" "{month_num}"',
                    f'site:reddit.com/r/cryptocurrency ethereum "{year}-{month_num}"',
                    f'site:reddit.com/r/ethtrader "{year}" crypto'
                ]
                
                for query in reddit_queries:
                    try:
                        # Use Google search to find Reddit posts
                        google_url = f"https://www.google.com/search?q={quote(query)}&num=20"
                        response = self.session.get(google_url, timeout=15)
                        
                        if response.status_code == 200:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            # Extract Reddit URLs from search results
                            for result_div in soup.find_all('div', class_='g'):
                                try:
                                    link_elem = result_div.find('a', href=True)
                                    if not link_elem:
                                        continue
                                        
                                    url = link_elem.get('href')
                                    if 'reddit.com' not in url or '/comments/' not in url:
                                        continue
                                    
                                    # Extract title from search result
                                    title_elem = result_div.find('h3')
                                    title = title_elem.get_text(strip=True) if title_elem else ""
                                    
                                    # Extract snippet
                                    snippet_elem = result_div.find('span')
                                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                    
                                    # Check for Ethereum relevance
                                    text = (title + " " + snippet).lower()
                                    if any(term in text for term in ['ethereum', 'eth', 'crypto', 'defi']):
                                        post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                        
                                        posts.append({
                                            'id': f"enhanced_reddit_{month}_{post_id}",
                                            'source': f'enhanced_reddit_{month}',
                                            'title': title,
                                            'content': snippet,
                                            'published_at': datetime.strptime(f"{year}-{month_num}-15", '%Y-%m-%d'),
                                            'url': url
                                        })
                                        
                                except Exception as e:
                                    logging.debug(f"Failed to parse Google result: {e}")
                                    continue
                        
                        time.sleep(5)  # Respectful delay for Google
                        
                    except Exception as e:
                        logging.warning(f"Reddit enhancement failed for query '{query}': {e}")
                        continue
                        
            except Exception as e:
                logging.warning(f"Reddit enhancement failed for month {month}: {e}")
                continue
        
        return posts
    
    def github_bulk_collection(self):
        """Bulk collection from GitHub for 2022-2023 period."""
        logging.info("Starting GitHub bulk collection")
        
        issues = []
        
        # Enhanced GitHub search for 2022-2023
        github_queries = [
            'ethereum created:2022-01-01..2022-06-30',
            'ethereum created:2022-07-01..2022-12-31', 
            'ethereum created:2023-01-01..2023-06-30',
            'ethereum created:2023-07-01..2023-12-31'
        ]
        
        for query in github_queries:
            try:
                search_url = "https://api.github.com/search/issues"
                params = {
                    'q': query,
                    'sort': 'created',
                    'per_page': 50
                }
                
                response = self.session.get(search_url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for issue in data.get('items', []):
                        try:
                            title = issue.get('title', '')
                            body = issue.get('body', '') or ''
                            created_at = issue.get('created_at', '')
                            html_url = issue.get('html_url', '')
                            
                            # Parse date
                            try:
                                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                month_key = created_date.strftime('%Y-%m')
                            except:
                                continue
                            
                            issue_id = str(issue.get('id', ''))
                            
                            issues.append({
                                'id': f"bulk_github_{month_key}_{issue_id}",
                                'source': f'bulk_github_{month_key}',
                                'title': title,
                                'content': body[:800],  # More content
                                'published_at': created_date.replace(tzinfo=None),
                                'url': html_url
                            })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse GitHub issue: {e}")
                            continue
                
                time.sleep(2)  # GitHub API rate limiting
                
            except Exception as e:
                logging.warning(f"GitHub bulk collection failed for query '{query}': {e}")
                continue
        
        return issues
    
    def save_bulk_data(self, data, source_name):
        """Save bulk collected data to database."""
        if not data:
            logging.info(f"No data collected from {source_name}")
            return 0
        
        # Use news_articles table for all sources
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
    
    def efficient_density_boost(self):
        """Main method for efficient density improvement."""
        logging.info("=== EFFICIENT DENSITY BOOST CAMPAIGN ===")
        
        start_time = datetime.now()
        
        # Get low-density months
        target_months = self.get_low_density_months()
        logging.info(f"Targeting {len(target_months)} low-density months")
        
        # Run collection strategies in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            tasks = {
                executor.submit(self.bulk_google_news_collection, target_months): "bulk_google_news",
                executor.submit(self.targeted_reddit_enhancement, target_months): "reddit_enhancement", 
                executor.submit(self.github_bulk_collection): "github_bulk"
            }
            
            total_collected = 0
            
            # Collect results
            for future in as_completed(tasks):
                task_name = tasks[future]
                try:
                    result = future.result()
                    saved = self.save_bulk_data(result, task_name)
                    total_collected += saved
                    logging.info(f"✅ {task_name}: {saved} new entries")
                except Exception as e:
                    logging.error(f"❌ {task_name} failed: {e}")
        
        elapsed = datetime.now() - start_time
        
        # Summary
        logging.info("=== EFFICIENT DENSITY BOOST COMPLETE ===")
        logging.info(f"Total new entries: {total_collected}")
        logging.info(f"Target months: {len(target_months)}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for efficient gap filling."""
    filler = EfficientGapFiller()
    
    print(f"🚀 EFFICIENT DENSITY BOOST")
    print(f"🎯 Goal: Improve daily density for professional ML standards")
    
    total = filler.efficient_density_boost()
    
    print(f"\n✅ DENSITY BOOST COMPLETE!")
    print(f"📈 Added {total} new entries")
    print(f"💡 Next: Verify improved dataset quality")

if __name__ == "__main__":
    main()