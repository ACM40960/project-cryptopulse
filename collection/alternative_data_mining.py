#!/usr/bin/env python3
# collection/alternative_data_mining.py

"""
Alternative data mining strategies to extract more content from 2022-2023 period.
Focus on overlooked sources, different search terms, and deeper content mining.
"""
import sys
import os
import time
import logging
import requests
import hashlib
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alternative_data_mining.log'),
        logging.StreamHandler()
    ]
)

class AlternativeDataMining:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Alternative search terms we might have missed
        self.alternative_keywords = [
            # Technical terms
            'ethereum gas', 'ethereum gwei', 'ethereum mempool', 'ethereum validator',
            'ethereum beacon', 'ethereum consensus', 'ethereum execution', 'ethereum withdrawal',
            'ethereum staking pool', 'ethereum liquid staking', 'ethereum MEV', 'ethereum flashloan',
            
            # Market terms  
            'ethereum whale', 'ethereum accumulation', 'ethereum distribution', 'ethereum OI',
            'ethereum funding rate', 'ethereum perp', 'ethereum spot', 'ethereum premium',
            'ethereum ratio', 'ethereum dominance', 'ethereum mcap', 'ethereum circulation',
            
            # Ecosystem terms
            'ethereum dapp', 'ethereum protocol', 'ethereum ecosystem', 'ethereum developer',
            'ethereum upgrade', 'ethereum proposal', 'ethereum governance', 'ethereum fork',
            'ethereum testnet', 'ethereum mainnet', 'ethereum client', 'ethereum node',
            
            # Event-specific terms
            'ethereum shanghai', 'ethereum cancun', 'ethereum dencun', 'ethereum arrow glacier',
            'ethereum gray glacier', 'ethereum kiln', 'ethereum ropsten', 'ethereum goerli',
            
            # DeFi specific
            'ethereum yield', 'ethereum liquidity', 'ethereum farming', 'ethereum mining',
            'ethereum swap', 'ethereum bridge', 'ethereum cross chain', 'ethereum interop',
            
            # NFT and Web3
            'ethereum mint', 'ethereum collection', 'ethereum marketplace', 'ethereum creator',
            'ethereum royalty', 'ethereum metadata', 'ethereum standard', 'ethereum token'
        ]
        
        # Alternative news sources that might have been missed
        self.alternative_sources = [
            # Crypto-native sources
            'https://theblock.co/rss.xml',
            'https://cryptonews.com/news/feed/',
            'https://crypto.news/feed/',
            'https://www.cryptopolitan.com/feed/',
            'https://cryptodaily.co.uk/feed/',
            'https://blockonomi.com/feed/',
            'https://ethereumworldnews.com/feed/',
            'https://www.ethnews.com/rss.xml',
            
            # Technical/Developer sources
            'https://blog.ethereum.org/feed.xml',
            'https://hackernoon.com/tagged/ethereum/feed',
            'https://medium.com/feed/ethereum-foundation',
            'https://ethereum.org/en/blog/feed.xml',
            'https://blog.openzeppelin.com/feed.xml',
            'https://blog.alchemy.com/feed',
            'https://blog.infura.io/feed/',
            'https://moralis.io/blog/feed/',
            
            # Financial/Institutional  
            'https://www.theblock.co/data/feed',
            'https://insights.deribit.com/feed/',
            'https://blog.bitfinex.com/feed/',
            'https://blog.coinbase.com/feed',
            'https://blog.kraken.com/feed/',
            'https://blog.binance.com/en/rss.xml',
            'https://blog.bybit.com/feed/',
            
            # Regional sources
            'https://en.bitcoin.com/feed/',
            'https://bitcoinist.com/feed/',
            'https://bitcoinmagazine.com/.rss/full/',
            'https://news.bitcoin.com/feed/',
            'https://coingeek.com/feed/',
            'https://cryptonews.net/rss/',
            
            # Research/Analysis
            'https://research.binance.com/en/feed',
            'https://messari.io/feed',
            'https://coinmetrics.io/feed/',
            'https://insights.glassnode.com/feed/',
            'https://blog.chainalysis.com/feed/'
        ]
    
    def deep_reddit_mining(self):
        """Mine Reddit more deeply with alternative search strategies."""
        logging.info("Starting deep Reddit mining with alternative strategies...")
        
        posts = []
        
        # Use Google to find more Reddit content with different search patterns
        reddit_search_patterns = [
            # Technical discussions
            'site:reddit.com/r/ethereum "gas fee" OR "gas price" 2022..2023',
            'site:reddit.com/r/ethereum "validator" OR "staking" 2022..2023',
            'site:reddit.com/r/ethereum "merge" OR "pos" 2022..2023',
            'site:reddit.com/r/ethereum "upgrade" OR "fork" 2022..2023',
            
            # Market discussions
            'site:reddit.com/r/ethtrader "price" OR "market" 2022..2023',
            'site:reddit.com/r/ethfinance "bullish" OR "bearish" 2022..2023',
            'site:reddit.com/r/cryptocurrency ethereum "buy" OR "sell" 2022..2023',
            'site:reddit.com/r/investing ethereum 2022..2023',
            
            # DeFi discussions
            'site:reddit.com/r/defi ethereum "yield" OR "farming" 2022..2023',
            'site:reddit.com/r/defi "liquidity" OR "pool" ethereum 2022..2023',
            'site:reddit.com/r/ethereum "defi" OR "dapp" 2022..2023',
            
            # Event-specific
            'site:reddit.com ethereum "shanghai" OR "withdrawals" 2022..2023',
            'site:reddit.com ethereum "ftx" OR "collapse" 2022..2023',
            'site:reddit.com ethereum "luna" OR "terra" 2022..2023',
            'site:reddit.com ethereum "celsius" OR "3ac" 2022..2023'
        ]
        
        for search_pattern in reddit_search_patterns:
            try:
                google_url = f"https://www.google.com/search?q={quote(search_pattern)}&num=30"
                response = self.session.get(google_url, timeout=15)
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for result in soup.find_all('div', class_='g')[:20]:
                        try:
                            link_elem = result.find('a', href=True)
                            if not link_elem:
                                continue
                                
                            url = link_elem.get('href')
                            if 'reddit.com' not in url:
                                continue
                            
                            title_elem = result.find('h3')
                            title = title_elem.get_text(strip=True) if title_elem else ""
                            
                            snippet_elem = result.find('span')
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            # Check relevance
                            text = (title + " " + snippet).lower()
                            if any(term in text for term in ['ethereum', 'eth', 'crypto', 'defi']):
                                post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                
                                posts.append({
                                    'id': f"deep_reddit_{post_id}",
                                    'source': 'deep_reddit_mining',
                                    'title': title,
                                    'content': snippet,
                                    'published_at': datetime(2022, 6, 15),  # Default to mid-period
                                    'url': url
                                })
                        
                        except Exception as e:
                            logging.debug(f"Failed to parse Reddit result: {e}")
                            continue
                
                time.sleep(random.uniform(3, 7))  # Variable delay
                
            except Exception as e:
                logging.warning(f"Deep Reddit mining failed for pattern: {e}")
                continue
        
        return posts
    
    def alternative_keyword_search(self):
        """Search Google News with alternative keywords we might have missed."""
        logging.info("Starting alternative keyword search...")
        
        articles = []
        
        # Time periods for focused search
        periods = [
            ('2022-01-01', '2022-06-30'),
            ('2022-07-01', '2022-12-31'),
            ('2023-01-01', '2023-06-30'),
            ('2023-07-01', '2023-12-31')
        ]
        
        for keyword in self.alternative_keywords[:15]:  # Limit to avoid timeout
            for start_date, end_date in periods:
                try:
                    search_query = f'"{keyword}" after:{start_date} before:{end_date}'
                    encoded_query = quote(search_query)
                    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                    
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:10]:
                        try:
                            title = entry.get('title', '')
                            link = entry.get('link', '')
                            summary = entry.get('summary', '')
                            
                            # Parse date
                            pub_date = datetime.now()
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            
                            # Verify timeframe
                            range_start = datetime.strptime(start_date, '%Y-%m-%d')
                            range_end = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            if not (range_start <= pub_date <= range_end):
                                continue
                            
                            # Check relevance
                            text = (title + " " + summary).lower()
                            if any(kw in text for kw in ['ethereum', 'crypto', 'blockchain', 'defi']):
                                article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                month_key = pub_date.strftime('%Y-%m')
                                
                                articles.append({
                                    'id': f"alt_keyword_{month_key}_{article_id}",
                                    'source': f'alt_keyword_{month_key}',
                                    'title': title,
                                    'content': summary,
                                    'published_at': pub_date,
                                    'url': link,
                                    'search_keyword': keyword
                                })
                        
                        except Exception as e:
                            logging.debug(f"Failed to parse alternative keyword entry: {e}")
                            continue
                    
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    logging.warning(f"Alternative keyword search failed for {keyword}: {e}")
                    continue
        
        return articles
    
    def mine_alternative_sources(self):
        """Mine alternative RSS sources that might have been overlooked."""
        logging.info("Mining alternative RSS sources...")
        
        articles = []
        
        for source_url in self.alternative_sources:
            try:
                logging.info(f"Mining: {source_url}")
                feed = feedparser.parse(source_url)
                
                if not feed.entries:
                    continue
                
                for entry in feed.entries[:50]:  # More entries per source
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '') or entry.get('description', '')
                        
                        # Parse date
                        pub_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Filter for 2022-2023
                        if not (datetime(2022, 1, 1) <= pub_date <= datetime(2024, 1, 1)):
                            continue
                        
                        # Enhanced relevance check
                        text = (title + " " + summary).lower()
                        ethereum_score = sum([
                            3 if 'ethereum' in text else 0,
                            2 if 'eth ' in text else 0,
                            2 if 'defi' in text else 0,
                            1 if 'crypto' in text else 0,
                            1 if 'blockchain' in text else 0
                        ])
                        
                        if ethereum_score >= 3:  # Require good relevance
                            article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                            month_key = pub_date.strftime('%Y-%m')
                            source_name = source_url.split('//')[1].split('/')[0].replace('www.', '').split('.')[0]
                            
                            articles.append({
                                'id': f"alt_source_{source_name}_{month_key}_{article_id}",
                                'source': f'alt_source_{source_name}_{month_key}',
                                'title': title,
                                'content': summary,
                                'published_at': pub_date,
                                'url': link,
                                'relevance_score': ethereum_score
                            })
                    
                    except Exception as e:
                        logging.debug(f"Failed to parse alternative source entry: {e}")
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                logging.warning(f"Alternative source mining failed for {source_url}: {e}")
                continue
        
        return articles
    
    def forum_deep_dive(self):
        """Deep dive into crypto forums for missed discussions."""
        logging.info("Starting forum deep dive...")
        
        posts = []
        
        # Forum sources with deeper search
        forum_searches = [
            'site:bitcointalk.org ethereum 2022..2023',
            'site:ethereum-magicians.org 2022..2023',
            'site:research.ethereum.org 2022..2023',
            'site:ethresear.ch 2022..2023',
            'site:gov.gitcoin.co ethereum 2022..2023',
            'site:forum.openzeppelin.com ethereum 2022..2023',
            'site:forum.makerdao.com 2022..2023',
            'site:gov.aave.com 2022..2023',
            'site:forum.uniswap.org 2022..2023',
            'site:forum.compound.finance 2022..2023'
        ]
        
        for search_query in forum_searches:
            try:
                google_url = f"https://www.google.com/search?q={quote(search_query)}&num=20"
                response = self.session.get(google_url, timeout=15)
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for result in soup.find_all('div', class_='g')[:15]:
                        try:
                            link_elem = result.find('a', href=True)
                            if not link_elem:
                                continue
                                
                            url = link_elem.get('href')
                            
                            title_elem = result.find('h3')
                            title = title_elem.get_text(strip=True) if title_elem else ""
                            
                            snippet_elem = result.find('span')
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            # Check relevance
                            text = (title + " " + snippet).lower()
                            if any(term in text for term in ['ethereum', 'eth', 'defi', 'dao', 'governance']):
                                post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                
                                posts.append({
                                    'id': f"forum_deep_{post_id}",
                                    'source': 'forum_deep_dive',
                                    'title': title,
                                    'content': snippet,
                                    'published_at': datetime(2022, 9, 15),  # Default
                                    'url': url
                                })
                        
                        except Exception as e:
                            logging.debug(f"Failed to parse forum result: {e}")
                            continue
                
                time.sleep(random.uniform(4, 8))
                
            except Exception as e:
                logging.warning(f"Forum deep dive failed for query: {e}")
                continue
        
        return posts
    
    def save_alternative_data(self, data, source_name):
        """Save alternative mining data to database."""
        if not data:
            logging.info(f"No data from {source_name}")
            return 0
        
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
        
        logging.info(f"{source_name}: {len(data)} collected → 0 new (duplicates)")
        return 0
    
    def alternative_mining_campaign(self):
        """Execute alternative mining campaign."""
        logging.info("=== ALTERNATIVE DATA MINING CAMPAIGN ===")
        
        start_time = datetime.now()
        
        # Run alternative strategies
        strategies = [
            ("Deep Reddit Mining", self.deep_reddit_mining),
            ("Alternative Keywords", self.alternative_keyword_search),
            ("Alternative Sources", self.mine_alternative_sources),
            ("Forum Deep Dive", self.forum_deep_dive)
        ]
        
        total_collected = 0
        
        for strategy_name, strategy_func in strategies:
            try:
                logging.info(f"Executing: {strategy_name}")
                data = strategy_func()
                saved = self.save_alternative_data(data, strategy_name.lower().replace(' ', '_'))
                total_collected += saved
                logging.info(f"✅ {strategy_name}: {saved} new entries")
                
            except Exception as e:
                logging.error(f"❌ {strategy_name} failed: {e}")
        
        elapsed = datetime.now() - start_time
        
        logging.info("=== ALTERNATIVE MINING COMPLETE ===")
        logging.info(f"Total new entries: {total_collected}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for alternative data mining."""
    miner = AlternativeDataMining()
    
    print(f"🔍 ALTERNATIVE DATA MINING")
    print(f"🎯 Goal: Extract more data from 2022-2023 period")
    print(f"📊 Strategy: Alternative keywords, sources, and deep mining")
    print(f"💡 Expected: 500-1000 additional entries")
    
    total = miner.alternative_mining_campaign()
    
    print(f"\n✅ ALTERNATIVE MINING COMPLETE!")
    print(f"📈 Added {total} new entries")
    print(f"🎯 Run dataset analysis to verify improvements")

if __name__ == "__main__":
    main()