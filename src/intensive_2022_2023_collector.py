#!/usr/bin/env python3
# src/intensive_2022_2023_collector.py

"""
Intensive data collection targeting 2022-2023 period for optimal ML training dataset.
Focus: Crypto winter, major events, bear market sentiment, recovery patterns.
"""
import sys
import os
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
        logging.FileHandler('logs/intensive_2022_2023_collector.log'),
        logging.StreamHandler()
    ]
)

class Intensive2022_2023Collector:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Target months with key events
        self.target_months = {
            '2022-01': {'events': ['Market correction start'], 'priority': 'high'},
            '2022-02': {'events': ['Russia-Ukraine war impact'], 'priority': 'high'},
            '2022-03': {'events': ['Fed rate hikes begin'], 'priority': 'medium'},
            '2022-04': {'events': ['Tech stocks decline'], 'priority': 'medium'},
            '2022-05': {'events': ['Terra Luna collapse prep'], 'priority': 'critical'},
            '2022-06': {'events': ['Terra Luna/UST collapse'], 'priority': 'critical'},
            '2022-07': {'events': ['Crypto winter deepens'], 'priority': 'high'},
            '2022-08': {'events': ['Ethereum merge anticipation'], 'priority': 'high'},
            '2022-09': {'events': ['Ethereum merge completion'], 'priority': 'critical'},
            '2022-10': {'events': ['Market uncertainty'], 'priority': 'medium'},
            '2022-11': {'events': ['FTX collapse'], 'priority': 'critical'},
            '2022-12': {'events': ['Contagion spread'], 'priority': 'high'},
            '2023-01': {'events': ['Bear market bottom'], 'priority': 'high'},
            '2023-02': {'events': ['Recovery signs'], 'priority': 'medium'},
            '2023-03': {'events': ['Silicon Valley Bank'], 'priority': 'critical'},
            '2023-04': {'events': ['Recovery momentum'], 'priority': 'medium'},
            '2023-05': {'events': ['Regulatory clarity'], 'priority': 'medium'},
            '2023-06': {'events': ['Institution re-entry'], 'priority': 'medium'},
            '2023-07': {'events': ['Bull market start'], 'priority': 'high'},
            '2023-08': {'events': ['Momentum building'], 'priority': 'high'},
            '2023-09': {'events': ['ETF anticipation'], 'priority': 'high'},
            '2023-10': {'events': ['Bitcoin ETF hype'], 'priority': 'critical'},
            '2023-11': {'events': ['ETF approval buzz'], 'priority': 'critical'},
            '2023-12': {'events': ['Year-end rally'], 'priority': 'high'}
        }
        
        # Event-specific search terms
        self.event_keywords = {
            'terra_luna': ['terra', 'luna', 'ust', 'do kwon', 'anchor', 'depeg', 'stablecoin collapse'],
            'ftx_collapse': ['ftx', 'sbf', 'sam bankman', 'alameda', 'bankruptcy', 'contagion'],
            'ethereum_merge': ['ethereum merge', 'proof of stake', 'pos', 'beacon chain', 'staking'],
            'crypto_winter': ['crypto winter', 'bear market', 'capitulation', 'bottom', 'crash'],
            'svb_crisis': ['silicon valley bank', 'svb', 'bank crisis', 'usdc depeg', 'credit suisse'],
            'recovery': ['recovery', 'bull market', 'institutional adoption', 'etf approval']
        }
    
    def get_reddit_alternative_sources(self):
        """Alternative Reddit data sources when API is limited."""
        sources = []
        
        # Google cache searches for Reddit discussions
        reddit_search_queries = [
            'site:reddit.com/r/cryptocurrency "ethereum" "2022"',
            'site:reddit.com/r/ethereum "terra luna" OR "ftx" 2022',
            'site:reddit.com/r/ethtrader "crypto winter" OR "bear market" 2022',
            'site:reddit.com/r/cryptocurrency "ethereum merge" 2022',
            'site:reddit.com/r/ethereum "proof of stake" 2022',
            'site:reddit.com/r/cryptocurrency "silicon valley bank" 2023',
            'site:reddit.com/r/ethereum "etf" OR "institutional" 2023'
        ]
        
        for query in reddit_search_queries:
            sources.append({
                'type': 'google_reddit_search',
                'query': query,
                'expected_count': 20
            })
        
        return sources
    
    def collect_google_reddit_cache(self, query, max_results=20):
        """Collect Reddit discussions from Google cache."""
        logging.info(f"Searching Google cache for: {query}")
        posts = []
        
        try:
            # Use Google search to find cached Reddit pages
            google_url = f"https://www.google.com/search?q={quote(query)}&num={max_results}"
            response = self.session.get(google_url, timeout=15)
            
            if response.status_code == 200:
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
                        snippet_elem = result_div.find('span', class_='aCOpRe')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Check for Ethereum relevance
                        text = (title + " " + snippet).lower()
                        if any(term in text for term in ['ethereum', 'eth', 'crypto', 'defi', 'merge', 'staking']):
                            post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                            
                            posts.append({
                                'id': f"google_reddit_{post_id}",
                                'source': 'google_reddit_cache',
                                'title': title,
                                'content': snippet,
                                'published_at': self.estimate_date_from_query(query),
                                'url': url
                            })
                            
                    except Exception as e:
                        logging.debug(f"Failed to parse Google result: {e}")
                        continue
            
            time.sleep(5)  # Respectful delay for Google
            
        except Exception as e:
            logging.warning(f"Google Reddit cache search failed: {e}")
        
        return posts
    
    def collect_news_for_events(self, target_month):
        """Collect news articles for specific events in target month."""
        logging.info(f"Collecting event-specific news for {target_month}")
        articles = []
        
        events = self.target_months.get(target_month, {}).get('events', [])
        if not events:
            return articles
        
        # Create search terms based on events
        search_terms = []
        for event in events:
            if 'terra' in event.lower() or 'luna' in event.lower():
                search_terms.extend(self.event_keywords['terra_luna'])
            elif 'ftx' in event.lower():
                search_terms.extend(self.event_keywords['ftx_collapse'])
            elif 'merge' in event.lower():
                search_terms.extend(self.event_keywords['ethereum_merge'])
            elif 'winter' in event.lower() or 'bear' in event.lower():
                search_terms.extend(self.event_keywords['crypto_winter'])
            elif 'bank' in event.lower():
                search_terms.extend(self.event_keywords['svb_crisis'])
            elif 'recovery' in event.lower() or 'bull' in event.lower():
                search_terms.extend(self.event_keywords['recovery'])
        
        # Google News search for specific events
        year, month = target_month.split('-')
        for term in search_terms[:5]:  # Limit to avoid rate limiting
            try:
                # Create date-specific Google News search
                news_query = f'"{term}" ethereum OR crypto {year}-{month}'
                news_url = f"https://news.google.com/rss/search?q={quote(news_query)}&hl=en&gl=US&ceid=US:en"
                
                import feedparser
                feed = feedparser.parse(news_url)
                
                for entry in feed.entries[:10]:  # Limit per term
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse publication date
                        pub_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Check if date matches target month
                        if pub_date.strftime('%Y-%m') != target_month:
                            continue
                        
                        article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                        articles.append({
                            'id': f"event_news_{article_id}",
                            'source': f'event_news_{target_month}',
                            'title': title,
                            'content': summary,
                            'published_at': pub_date,
                            'url': link
                        })
                        
                    except Exception as e:
                        logging.debug(f"Failed to parse news entry: {e}")
                        continue
                
                time.sleep(3)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"Event news search failed for {term}: {e}")
                continue
        
        return articles
    
    def collect_github_discussions_2022_2023(self):
        """Collect GitHub discussions from Ethereum repos during 2022-2023."""
        logging.info("Collecting GitHub discussions for 2022-2023 period")
        discussions = []
        
        # Target Ethereum repositories
        repos = [
            'ethereum/ethereum',
            'ethereum/EIPs',
            'ethereum/consensus-specs',
            'ethereum/execution-specs'
        ]
        
        for repo in repos:
            try:
                # Search for issues/discussions in 2022-2023
                search_url = "https://api.github.com/search/issues"
                params = {
                    'q': f'repo:{repo} created:2022-01-01..2023-12-31 ethereum OR merge OR pos',
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
                            except:
                                continue
                            
                            # Check if in target period
                            if not ('2022' in created_at or '2023' in created_at):
                                continue
                            
                            issue_id = str(issue.get('id', ''))
                            discussions.append({
                                'id': f"github_2022_23_{issue_id}",
                                'source': f'github_{repo.replace("/", "_")}',
                                'title': title,
                                'content': body[:1000],  # Limit content length
                                'published_at': created_date.replace(tzinfo=None),
                                'url': html_url
                            })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse GitHub issue: {e}")
                            continue
                
                time.sleep(2)  # GitHub API rate limiting
                
            except Exception as e:
                logging.warning(f"GitHub collection failed for {repo}: {e}")
                continue
        
        return discussions
    
    def estimate_date_from_query(self, query):
        """Estimate date from search query."""
        if '2022' in query:
            return datetime(2022, 6, 15)  # Mid-2022
        elif '2023' in query:
            return datetime(2023, 6, 15)  # Mid-2023
        else:
            return datetime(2022, 12, 15)  # Default to end of 2022
    
    def save_collected_data(self, data, source_name):
        """Save collected data to database."""
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
    
    def intensive_2022_2023_collection(self):
        """Main method for intensive 2022-2023 data collection."""
        logging.info("=== INTENSIVE 2022-2023 COLLECTION START ===")
        
        start_time = datetime.now()
        total_collected = 0
        
        # 1. Google Reddit cache collection
        logging.info("Phase 1: Google Reddit cache collection...")
        reddit_sources = self.get_reddit_alternative_sources()
        
        for source in reddit_sources[:3]:  # Limit for testing
            try:
                posts = self.collect_google_reddit_cache(source['query'], source['expected_count'])
                saved = self.save_collected_data(posts, f"reddit_cache_{source['query'][:20]}")
                total_collected += saved
            except Exception as e:
                logging.error(f"Reddit cache collection failed: {e}")
                continue
        
        # 2. Event-specific news collection
        logging.info("Phase 2: Event-specific news collection...")
        critical_months = [month for month, data in self.target_months.items() 
                          if data['priority'] == 'critical']
        
        for month in critical_months[:4]:  # Focus on most critical events
            try:
                articles = self.collect_news_for_events(month)
                saved = self.save_collected_data(articles, f"event_news_{month}")
                total_collected += saved
            except Exception as e:
                logging.error(f"Event news collection failed for {month}: {e}")
                continue
        
        # 3. GitHub discussions
        logging.info("Phase 3: GitHub discussions collection...")
        try:
            discussions = self.collect_github_discussions_2022_2023()
            saved = self.save_collected_data(discussions, "github_2022_2023")
            total_collected += saved
        except Exception as e:
            logging.error(f"GitHub collection failed: {e}")
        
        elapsed = datetime.now() - start_time
        
        # Summary
        logging.info("=== INTENSIVE 2022-2023 COLLECTION COMPLETE ===")
        logging.info(f"Total new entries collected: {total_collected}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for intensive 2022-2023 collection."""
    collector = Intensive2022_2023Collector()
    total = collector.intensive_2022_2023_collection()
    
    print(f"\n🎯 INTENSIVE 2022-2023 COLLECTION COMPLETE")
    print(f"📊 Total new entries: {total}")
    print(f"🎪 Focus: Crypto winter, major events, recovery patterns")
    print(f"💡 Next: Run metrics processing on new data")

if __name__ == "__main__":
    main()