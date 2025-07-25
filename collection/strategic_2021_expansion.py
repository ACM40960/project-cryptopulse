#!/usr/bin/env python3
# strategic_2021_expansion.py

"""
Strategic expansion to 2021-2023 timeframe to achieve professional ML standards.
This approach increases both volume and daily density by adding the high-activity 2021 bull market period.
"""
import sys
import os
import time
import logging
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
        logging.FileHandler('logs/strategic_2021_expansion.log'),
        logging.StreamHandler()
    ]
)

class Strategic2021Expansion:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 2021 key events and periods
        self.key_2021_events = {
            '2021-01': 'DeFi boom expansion',
            '2021-02': 'Institutional adoption wave',
            '2021-03': 'NFT explosion start',
            '2021-04': 'ETH price surge to $4000',
            '2021-05': 'Market correction and EIP-1559 hype',
            '2021-06': 'London hard fork anticipation',
            '2021-07': 'ETH deflationary narrative',
            '2021-08': 'London hard fork implementation',
            '2021-09': 'Layer 2 scaling solutions',
            '2021-10': 'Bitcoin ETF approval anticipation',
            '2021-11': 'ETH all-time high period',
            '2021-12': 'Web3 and metaverse hype'
        }
    
    def google_news_2021_comprehensive(self):
        """Comprehensive Google News collection for 2021 bull market."""
        logging.info("Starting comprehensive 2021 Google News collection...")
        
        articles = []
        
        # 2021-specific search terms focusing on major events
        search_terms_2021 = [
            # Bull market and price action
            'ethereum bull market 2021', 'ethereum price 4000 2021', 'ethereum all time high 2021',
            'ethereum institutional adoption 2021', 'ethereum corporate investment 2021',
            
            # DeFi explosion
            'ethereum defi boom 2021', 'ethereum yield farming 2021', 'ethereum liquidity mining 2021',
            'ethereum uniswap 2021', 'ethereum aave compound 2021', 'ethereum defi protocols 2021',
            
            # NFT mania
            'ethereum nft 2021', 'ethereum opensea 2021', 'ethereum digital art 2021',
            'ethereum collectibles 2021', 'ethereum metaverse 2021',
            
            # Technical developments
            'ethereum london hard fork 2021', 'ethereum EIP-1559 2021', 'ethereum fee burn 2021',
            'ethereum layer 2 2021', 'ethereum scaling 2021', 'ethereum arbitrum optimism 2021',
            
            # Market dynamics
            'ethereum vs bitcoin 2021', 'ethereum market cap 2021', 'ethereum dominance 2021',
            'ethereum trading volume 2021', 'ethereum whale activity 2021',
            
            # Ecosystem growth
            'ethereum developers 2021', 'ethereum ecosystem 2021', 'ethereum adoption 2021',
            'ethereum enterprise 2021', 'ethereum mainstream 2021'
        ]
        
        for search_term in search_terms_2021:
            try:
                # Create 2021-specific search
                search_query = f'{search_term} after:2021-01-01 before:2022-01-01'
                encoded_query = quote(search_query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:15]:  # More entries for 2021
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse publication date
                        pub_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Filter for 2021
                        if pub_date.year != 2021:
                            continue
                        
                        # Check for crypto relevance
                        text = (title + " " + summary).lower()
                        if any(keyword in text for keyword in ['ethereum', 'crypto', 'bitcoin', 'blockchain', 'defi', 'nft']):
                            article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                            month_key = pub_date.strftime('%Y-%m')
                            
                            articles.append({
                                'id': f"2021_news_{month_key}_{article_id}",
                                'source': f'2021_expansion_{month_key}',
                                'title': title,
                                'content': summary,
                                'published_at': pub_date,
                                'url': link
                            })
                        
                    except Exception as e:
                        logging.debug(f"Failed to parse 2021 news entry: {e}")
                        continue
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"2021 Google News failed for '{search_term}': {e}")
                continue
        
        return articles
    
    def reddit_2021_targeted_search(self):
        """Targeted Reddit search for 2021 content."""
        logging.info("Starting targeted 2021 Reddit search...")
        
        posts = []
        
        # Use Google to find 2021 Reddit discussions
        reddit_2021_searches = [
            'site:reddit.com/r/ethereum "2021" "bull market" OR "ath"',
            'site:reddit.com/r/ethereum "2021" "defi" OR "yield farming"',
            'site:reddit.com/r/ethereum "2021" "nft" OR "opensea"',
            'site:reddit.com/r/ethereum "2021" "london" OR "eip-1559"',
            'site:reddit.com/r/cryptocurrency ethereum "2021" price OR market',
            'site:reddit.com/r/ethtrader "2021" ethereum trading',
            'site:reddit.com/r/ethereum "2021" "layer 2" OR "arbitrum"',
            'site:reddit.com/r/defi "2021" ethereum protocols'
        ]
        
        for search_query in reddit_2021_searches:
            try:
                google_url = f"https://www.google.com/search?q={quote(search_query)}&num=30"
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
                            
                            # Check for Ethereum relevance and 2021 content
                            text = (title + " " + snippet).lower()
                            if (any(term in text for term in ['ethereum', 'eth', 'crypto', 'defi', 'nft']) and
                                '2021' in text):
                                
                                post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                
                                posts.append({
                                    'id': f"2021_reddit_{post_id}",
                                    'source': '2021_reddit_expansion',
                                    'title': title,
                                    'content': snippet,
                                    'published_at': datetime(2021, 6, 15),  # Mid-2021 approximate
                                    'url': url
                                })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse Reddit result: {e}")
                            continue
                
                time.sleep(5)  # Respectful delay for Google
                
            except Exception as e:
                logging.warning(f"Reddit 2021 search failed for query: {e}")
                continue
        
        return posts
    
    def github_2021_ethereum_activity(self):
        """Collect GitHub Ethereum activity from 2021."""
        logging.info("Starting GitHub 2021 Ethereum activity collection...")
        
        issues = []
        
        # GitHub searches for 2021 Ethereum development
        github_2021_queries = [
            'ethereum created:2021-01-01..2021-04-30',  # Q1 2021
            'ethereum created:2021-05-01..2021-08-31',  # Q2-Q3 2021
            'ethereum created:2021-09-01..2021-12-31',  # Q4 2021
            'EIP-1559 created:2021-01-01..2021-12-31',  # EIP-1559 specific
            'ethereum layer 2 created:2021-01-01..2021-12-31',  # Layer 2 development
            'ethereum london hard fork created:2021-01-01..2021-12-31'  # London hard fork
        ]
        
        for query in github_2021_queries:
            try:
                search_url = "https://api.github.com/search/issues"
                params = {
                    'q': query,
                    'sort': 'created',
                    'per_page': 30
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
                            
                            # Ensure it's 2021
                            if created_date.year != 2021:
                                continue
                            
                            issue_id = str(issue.get('id', ''))
                            
                            issues.append({
                                'id': f"2021_github_{month_key}_{issue_id}",
                                'source': f'2021_github_{month_key}',
                                'title': title,
                                'content': body[:800],
                                'published_at': created_date.replace(tzinfo=None),
                                'url': html_url
                            })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse GitHub issue: {e}")
                            continue
                
                time.sleep(2)  # GitHub API rate limiting
                
            except Exception as e:
                logging.warning(f"GitHub 2021 collection failed for query '{query}': {e}")
                continue
        
        return issues
    
    def crypto_media_2021_archives(self):
        """Collect crypto media archives from 2021."""
        logging.info("Starting crypto media 2021 archives...")
        
        articles = []
        
        # Major crypto media RSS feeds with 2021 focus
        crypto_media_sources = [
            'https://cointelegraph.com/rss',
            'https://decrypt.co/feed',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://thedefiant.io/feed',
            'https://blockworks.co/feed'
        ]
        
        for rss_url in crypto_media_sources:
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:30]:  # More entries for 2021
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse publication date
                        pub_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Filter for 2021 (some feeds may have recent articles about 2021)
                        text_to_check = (title + " " + summary).lower()
                        if not ('2021' in text_to_check or pub_date.year == 2021):
                            continue
                        
                        # Check for Ethereum relevance
                        if any(keyword in text_to_check for keyword in ['ethereum', 'eth', 'defi', 'nft', 'layer 2']):
                            article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                            
                            # Use 2021 date if content is about 2021
                            if '2021' in text_to_check and pub_date.year != 2021:
                                pub_date = datetime(2021, 6, 15)  # Mid-2021 default
                            
                            month_key = pub_date.strftime('%Y-%m')
                            
                            articles.append({
                                'id': f"2021_media_{month_key}_{article_id}",
                                'source': f'2021_media_{month_key}',
                                'title': title,
                                'content': summary,
                                'published_at': pub_date,
                                'url': link
                            })
                        
                    except Exception as e:
                        logging.debug(f"Failed to parse media entry: {e}")
                        continue
                
                time.sleep(3)
                
            except Exception as e:
                logging.warning(f"Crypto media 2021 collection failed for {rss_url}: {e}")
                continue
        
        return articles
    
    def defi_ecosystem_2021_data(self):
        """Collect DeFi ecosystem data from 2021 boom period."""
        logging.info("Starting DeFi ecosystem 2021 data collection...")
        
        defi_content = []
        
        # DeFi-specific searches for 2021
        defi_2021_searches = [
            'ethereum defi total value locked 2021',
            'ethereum yield farming protocols 2021',
            'ethereum liquidity mining rewards 2021',
            'ethereum defi governance tokens 2021',
            'ethereum uniswap v3 launch 2021',
            'ethereum aave compound maker 2021',
            'ethereum defi summer continuation 2021',
            'ethereum automated market makers 2021'
        ]
        
        for search_term in defi_2021_searches:
            try:
                encoded_query = quote(search_term)
                search_url = f"https://www.google.com/search?q={encoded_query}&num=15"
                
                response = self.session.get(search_url, timeout=15)
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract search results
                    for result_div in soup.find_all('div', class_='g')[:8]:
                        try:
                            link_elem = result_div.find('a', href=True)
                            if not link_elem:
                                continue
                                
                            url = link_elem.get('href')
                            
                            # Extract title from search result
                            title_elem = result_div.find('h3')
                            title = title_elem.get_text(strip=True) if title_elem else ""
                            
                            # Extract snippet
                            snippet_elem = result_div.find('span')
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            # Check for DeFi and Ethereum relevance
                            text = (title + " " + snippet).lower()
                            if (any(term in text for term in ['ethereum', 'defi', 'yield', 'liquidity', 'protocol']) and
                                '2021' in text):
                                
                                content_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                
                                defi_content.append({
                                    'id': f"2021_defi_{content_id}",
                                    'source': '2021_defi_expansion',
                                    'title': title,
                                    'content': snippet,
                                    'published_at': datetime(2021, 7, 15),  # Mid-2021 DeFi peak
                                    'url': url
                                })
                                
                        except Exception as e:
                            logging.debug(f"Failed to parse DeFi result: {e}")
                            continue
                
                time.sleep(5)  # Respectful delay for Google
                
            except Exception as e:
                logging.warning(f"DeFi 2021 search failed for '{search_term}': {e}")
                continue
        
        return defi_content
    
    def save_2021_data(self, data, source_name):
        """Save 2021 expansion data to database."""
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
    
    def strategic_2021_expansion_campaign(self):
        """Main method for strategic 2021 expansion."""
        logging.info("=== STRATEGIC 2021 EXPANSION CAMPAIGN ===")
        
        start_time = datetime.now()
        
        # Run all 2021 collection strategies in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks
            tasks = {
                executor.submit(self.google_news_2021_comprehensive): "2021_google_news",
                executor.submit(self.reddit_2021_targeted_search): "2021_reddit_search",
                executor.submit(self.github_2021_ethereum_activity): "2021_github_activity",
                executor.submit(self.crypto_media_2021_archives): "2021_media_archives"
            }
            
            total_collected = 0
            
            # Collect results
            for future in as_completed(tasks):
                task_name = tasks[future]
                try:
                    result = future.result()
                    saved = self.save_2021_data(result, task_name)
                    total_collected += saved
                    logging.info(f"✅ {task_name}: {saved} new entries")
                except Exception as e:
                    logging.error(f"❌ {task_name} failed: {e}")
        
        # Additional targeted collection
        logging.info("Running additional targeted 2021 DeFi collection...")
        defi_data = self.defi_ecosystem_2021_data()
        defi_saved = self.save_2021_data(defi_data, "2021_defi_ecosystem")
        total_collected += defi_saved
        logging.info(f"✅ 2021_defi_ecosystem: {defi_saved} new entries")
        
        elapsed = datetime.now() - start_time
        
        # Summary
        logging.info("=== STRATEGIC 2021 EXPANSION COMPLETE ===")
        logging.info(f"Total new entries: {total_collected}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for strategic 2021 expansion."""
    expander = Strategic2021Expansion()
    
    print(f"🚀 STRATEGIC 2021 EXPANSION CAMPAIGN")
    print(f"🎯 Goal: Add 2021 bull market data for improved daily density")
    print(f"📈 Strategy: Comprehensive multi-source 2021 collection")
    print(f"💡 Expected: 2,000+ additional entries from high-activity period")
    
    total = expander.strategic_2021_expansion_campaign()
    
    print(f"\n🎉 2021 EXPANSION COMPLETE!")
    print(f"📊 Added {total} new entries from 2021 bull market period")
    print(f"🎯 Next: Verify improved dataset density and ML readiness")

if __name__ == "__main__":
    main()