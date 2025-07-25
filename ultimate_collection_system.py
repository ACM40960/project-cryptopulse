#!/usr/bin/env python3
# ultimate_collection_system.py

"""
Ultimate comprehensive collection system to achieve 15+ entries/day professional ML standard.
Combines scaled RSS, News APIs, international sources, and historical archives.
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
        logging.FileHandler('logs/ultimate_collection_system.log'),
        logging.StreamHandler()
    ]
)

class UltimateCollectionSystem:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Comprehensive international RSS feeds (200+ sources)
        self.ultimate_rss_feeds = {
            'tier1_major_english': [
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
                'https://ambcrypto.com/feed/',
                'https://cryptopotato.com/feed/',
                'https://beincrypto.com/feed/',
                'https://www.crypto-news-flash.com/feed/',
                'https://cryptobriefing.com/feed/',
                'https://coinjournal.net/feed/',
                'https://coinspeaker.com/feed/',
                'https://bitcoinist.com/feed/',
                'https://www.theblockcrypto.com/rss.xml',
                'https://coincodex.com/en/news/feed/',
                'https://cryptonews.net/rss/',
                'https://cryptoadventure.com/feed/',
                'https://dailyhodl.com/feed/',
                'https://www.cryptoglobe.com/latest/feed/',
                'https://cryptocurrencynews.com/feed/'
            ],
            
            'tier2_ethereum_defi': [
                'https://blog.ethereum.org/feed.xml',
                'https://consensys.net/blog/feed/',
                'https://medium.com/feed/@VitalikButerin',
                'https://medium.com/feed/ethereum-foundation',
                'https://week-in-ethereum.substack.com/feed',
                'https://ethresear.ch/latest.rss',
                'https://defipulse.com/blog/feed',
                'https://defillama.com/blog/rss.xml',
                'https://banklesshq.com/rss/',
                'https://newsletter.banklesshq.com/feed',
                'https://medium.com/feed/metamask',
                'https://medium.com/feed/uniswap-org',
                'https://medium.com/feed/compound-finance',
                'https://medium.com/feed/aave',
                'https://medium.com/feed/makerdao',
                'https://blog.synthetix.io/rss/',
                'https://medium.com/feed/curve-fi',
                'https://blog.1inch.io/feed',
                'https://medium.com/feed/yearn-state-of-the-vaults',
                'https://blog.balancer.fi/feed'
            ],
            
            'tier3_trading_markets': [
                'https://www.coindesk.com/markets/rss/',
                'https://markets.businessinsider.com/rss/news',
                'https://finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US',
                'https://www.marketwatch.com/rss/cryptocurrency',
                'https://seekingalpha.com/api/sa/combined/ETHUSD.xml',
                'https://www.benzinga.com/feed',
                'https://www.investing.com/rss/news_285.rss',  # Crypto news
                'https://www.tradingview.com/blog/en/feed/',
                'https://cryptorank.io/rss/news',
                'https://coinmarketcap.com/headlines/rss/',
                'https://coingecko.com/buzz/feed',
                'https://messari.io/feed',
                'https://www.deribit.com/blog/feed/',
                'https://blog.bitfinex.com/feed/',
                'https://blog.kraken.com/feed/',
                'https://blog.coinbase.com/feed'
            ],
            
            'tier4_international': [
                # European sources
                'https://www.btc-echo.de/feed/',  # German
                'https://www.journaldunet.com/rss/web-tech.xml',  # French
                'https://www.cryptonews.it/feed/',  # Italian
                'https://es.cointelegraph.com/rss',  # Spanish
                'https://br.cointelegraph.com/rss',  # Portuguese
                'https://www.cryptopolitan.com/feed/',
                'https://www.financemagnates.com/feed/',
                'https://en.ethereumworldnews.com/feed/',
                'https://www.investinblockchain.com/feed/',
                'https://bitcoinexchangeguide.com/feed/',
                'https://www.coininsider.com/feed/',
                
                # Asian sources
                'https://cointelegraph.com/rss/tag/asia',
                'https://www.coindesk.com/tag/asia/feed/',
                'https://www.8btc.com/rss',  # Chinese
                'https://coinpost.jp/?feed=rss2',  # Japanese
                'https://www.coindeskkorea.com/feed/',  # Korean
                'https://cryptonews.com/rss/asia/',
                'https://blocktribune.com/feed/',
                'https://cryptonews.com/news/feed/'
            ],
            
            'tier5_reddit_social': [
                'https://www.reddit.com/r/ethereum.rss?limit=100',
                'https://www.reddit.com/r/CryptoCurrency.rss?limit=100',
                'https://www.reddit.com/r/ethtrader.rss?limit=100',
                'https://www.reddit.com/r/defi.rss?limit=100',
                'https://www.reddit.com/r/ethfinance.rss?limit=100',
                'https://www.reddit.com/r/ethstaker.rss?limit=100',
                'https://www.reddit.com/r/ethereum/top.rss?t=week',
                'https://www.reddit.com/r/ethereum/top.rss?t=month',
                'https://www.reddit.com/r/CryptoCurrency/top.rss?t=week',
                'https://www.reddit.com/r/ethtrader/top.rss?t=week'
            ],
            
            'tier6_mainstream_tech': [
                'https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml&size=100&tagName=crypto',
                'https://feeds.bloomberg.com/crypto/news.rss',
                'https://www.cnbc.com/id/100727362/device/rss/rss.html',
                'https://techcrunch.com/category/cryptocurrency/feed/',
                'https://www.theverge.com/rss/cryptocurrency/index.xml',
                'https://arstechnica.com/information-technology/feed/',
                'https://www.wired.com/feed/category/business/fintech/rss',
                'https://www.zdnet.com/topic/blockchain/rss.xml',
                'https://venturebeat.com/category/fintech/feed/',
                'https://www.forbes.com/sites/crypto-blockchain/feed/'
            ]
        }
        
        # Enhanced Google News search terms (300+ combinations)
        self.comprehensive_search_terms = {
            'price_market': [
                'ethereum price analysis', 'ethereum market prediction', 'ethereum trading signals',
                'ethereum price crash', 'ethereum price surge', 'ethereum bull market',
                'ethereum bear market', 'ethereum market cap', 'ethereum volume analysis',
                'ethereum price target', 'ethereum resistance levels', 'ethereum support levels',
                'ethereum technical analysis', 'ethereum chart pattern', 'ethereum breakout'
            ],
            'technology': [
                'ethereum merge upgrade', 'ethereum proof of stake', 'ethereum staking rewards',
                'ethereum london hard fork', 'ethereum EIP proposal', 'ethereum scalability',
                'ethereum layer 2 solutions', 'ethereum rollup technology', 'ethereum sharding',
                'ethereum consensus mechanism', 'ethereum validator network', 'ethereum beacon chain',
                'ethereum gas fees', 'ethereum transaction speed', 'ethereum network congestion'
            ],
            'defi_ecosystem': [
                'ethereum defi protocol', 'ethereum smart contracts', 'ethereum dapps',
                'ethereum yield farming', 'ethereum liquidity mining', 'ethereum defi hack',
                'ethereum defi security', 'ethereum defi innovation', 'ethereum defi growth',
                'ethereum uniswap', 'ethereum compound', 'ethereum aave', 'ethereum maker',
                'ethereum synthetix', 'ethereum curve', 'ethereum balancer', 'ethereum 1inch'
            ],
            'institutional': [
                'ethereum institutional adoption', 'ethereum enterprise blockchain', 'ethereum corporate',
                'ethereum investment fund', 'ethereum hedge fund', 'ethereum asset management',
                'ethereum pension fund', 'ethereum sovereign wealth', 'ethereum treasury reserve',
                'ethereum ETF approval', 'ethereum futures trading', 'ethereum derivatives',
                'ethereum custody solution', 'ethereum institutional infrastructure'
            ],
            'regulatory': [
                'ethereum regulation SEC', 'ethereum compliance framework', 'ethereum legal status',
                'ethereum tax implications', 'ethereum government policy', 'ethereum central bank',
                'ethereum regulatory clarity', 'ethereum legal tender', 'ethereum sanctions',
                'ethereum AML compliance', 'ethereum KYC requirements', 'ethereum FATF guidelines'
            ],
            'development': [
                'ethereum developer activity', 'ethereum github commits', 'ethereum code updates',
                'ethereum foundation grant', 'ethereum ecosystem fund', 'ethereum hackathon',
                'ethereum conference event', 'ethereum community proposal', 'ethereum roadmap',
                'ethereum research paper', 'ethereum academic study', 'ethereum white paper'
            ]
        }
        
        # News API configurations (free tiers available)
        self.news_apis = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'params': {
                    'q': 'ethereum OR crypto OR blockchain',
                    'domains': 'cointelegraph.com,coindesk.com,decrypt.co,thedefiant.io',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 100
                },
                'headers': {}  # Add API key if available
            },
            'gnews': {
                'url': 'https://gnews.io/api/v4/search',
                'params': {
                    'q': 'ethereum',
                    'lang': 'en',
                    'country': 'us',
                    'max': 100
                },
                'headers': {}  # Add API key if available
            }
        }
    
    def scrape_rss_with_historical(self, rss_url, target_months=None):
        """Enhanced RSS scraping with historical content focus."""
        articles = []
        
        try:
            logging.info(f"Enhanced scraping: {rss_url}")
            
            # Try multiple fetch attempts for better coverage
            feeds_to_try = [
                rss_url,
                f"{rss_url}?_={int(time.time())}",  # Cache busting
                rss_url.replace('rss', 'feed').replace('feed', 'rss')  # Alternative formats
            ]
            
            for feed_url in feeds_to_try:
                try:
                    feed = feedparser.parse(feed_url)
                    if feed.entries:
                        break
                except:
                    continue
            
            if not feed.entries:
                logging.warning(f"No entries found: {rss_url}")
                return articles
            
            # Process all entries with enhanced filtering
            for entry in feed.entries:
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    summary = entry.get('summary', '') or entry.get('description', '') or entry.get('content', '')
                    
                    # Enhanced date parsing
                    pub_date = datetime.now()
                    date_fields = ['published_parsed', 'updated_parsed', 'created_parsed', 'modified_parsed']
                    
                    for field in date_fields:
                        if hasattr(entry, field) and getattr(entry, field):
                            try:
                                pub_date = datetime(*getattr(entry, field)[:6])
                                break
                            except:
                                continue
                    
                    # Target 2022-2023 primarily, but allow some 2021 and 2024
                    if not (datetime(2021, 6, 1) <= pub_date <= datetime(2024, 6, 1)):
                        continue
                    
                    # Enhanced crypto relevance scoring
                    text_content = (title + " " + summary).lower()
                    
                    # Expanded keyword sets with weights
                    high_value_keywords = ['ethereum', 'eth ', 'defi', 'smart contract', 'vitalik']
                    medium_value_keywords = ['crypto', 'blockchain', 'dapp', 'web3', 'nft', 'dao']
                    low_value_keywords = ['bitcoin', 'altcoin', 'trading', 'market', 'price']
                    
                    relevance_score = (
                        sum(3 for kw in high_value_keywords if kw in text_content) +
                        sum(2 for kw in medium_value_keywords if kw in text_content) +
                        sum(1 for kw in low_value_keywords if kw in text_content)
                    )
                    
                    # Require higher relevance for inclusion
                    if relevance_score >= 4:
                        article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                        month_key = pub_date.strftime('%Y-%m')
                        
                        source_name = rss_url.split('//')[1].split('/')[0].replace('www.', '').split('.')[0]
                        
                        articles.append({
                            'id': f"ultimate_{source_name}_{month_key}_{article_id}",
                            'source': f'ultimate_{source_name}_{month_key}',
                            'title': title,
                            'content': summary,
                            'published_at': pub_date,
                            'url': link,
                            'relevance_score': relevance_score
                        })
                    
                except Exception as e:
                    logging.debug(f"Failed to parse entry: {e}")
                    continue
            
            logging.info(f"Enhanced RSS {rss_url}: {len(articles)} high-quality articles")
            return articles
            
        except Exception as e:
            logging.warning(f"Enhanced RSS failed {rss_url}: {e}")
            return articles
    
    def comprehensive_google_news_search(self):
        """Massive Google News search across all categories and time periods."""
        logging.info("Starting comprehensive Google News mega-search...")
        
        articles = []
        
        # Detailed time periods for thorough coverage
        time_periods = [
            ('2021-06-01', '2021-12-31'),  # Late 2021 bull market
            ('2022-01-01', '2022-03-31'),  # Q1 2022
            ('2022-04-01', '2022-06-30'),  # Q2 2022
            ('2022-07-01', '2022-09-30'),  # Q3 2022 (Merge period)
            ('2022-10-01', '2022-12-31'),  # Q4 2022
            ('2023-01-01', '2023-03-31'),  # Q1 2023
            ('2023-04-01', '2023-06-30'),  # Q2 2023
            ('2023-07-01', '2023-09-30'),  # Q3 2023
            ('2023-10-01', '2023-12-31'),  # Q4 2023
            ('2024-01-01', '2024-03-31')   # Early 2024
        ]
        
        search_count = 0
        total_searches = sum(len(terms) for terms in self.comprehensive_search_terms.values()) * len(time_periods)
        
        for category, search_terms in self.comprehensive_search_terms.items():
            for search_term in search_terms:
                for start_date, end_date in time_periods:
                    try:
                        search_count += 1
                        if search_count % 50 == 0:
                            logging.info(f"Google News progress: {search_count}/{total_searches} searches")
                        
                        # Enhanced search query with operators
                        search_query = f'"{search_term}" after:{start_date} before:{end_date}'
                        encoded_query = quote(search_query)
                        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                        
                        feed = feedparser.parse(rss_url)
                        
                        for entry in feed.entries[:20]:  # More entries per search
                            try:
                                title = entry.get('title', '')
                                link = entry.get('link', '')
                                summary = entry.get('summary', '')
                                
                                # Parse date
                                pub_date = datetime.now()
                                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                    pub_date = datetime(*entry.published_parsed[:6])
                                
                                # Verify date range
                                range_start = datetime.strptime(start_date, '%Y-%m-%d')
                                range_end = datetime.strptime(end_date, '%Y-%m-%d')
                                
                                if not (range_start <= pub_date <= range_end):
                                    continue
                                
                                # Enhanced relevance check
                                text = (title + " " + summary).lower()
                                crypto_keywords = ['ethereum', 'crypto', 'blockchain', 'defi', 'web3', 'dapp']
                                
                                if any(keyword in text for keyword in crypto_keywords):
                                    article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                    month_key = pub_date.strftime('%Y-%m')
                                    
                                    articles.append({
                                        'id': f"mega_google_{category}_{month_key}_{article_id}",
                                        'source': f'mega_google_{category}_{month_key}',
                                        'title': title,
                                        'content': summary,
                                        'published_at': pub_date,
                                        'url': link,
                                        'search_category': category
                                    })
                                
                            except Exception as e:
                                logging.debug(f"Failed to parse Google News entry: {e}")
                                continue
                        
                        # Dynamic rate limiting
                        time.sleep(random.uniform(0.5, 1.5))
                        
                    except Exception as e:
                        logging.warning(f"Google News search failed for {search_term} {start_date}-{end_date}: {e}")
                        continue
        
        logging.info(f"Google News mega-search complete: {len(articles)} articles from {search_count} searches")
        return articles
    
    def collect_news_apis(self):
        """Collect from News APIs (if API keys available)."""
        logging.info("Attempting News API collection...")
        
        articles = []
        
        for api_name, config in self.news_apis.items():
            try:
                # Skip if no API key configured
                if not config.get('headers'):
                    logging.info(f"Skipping {api_name}: No API key configured")
                    continue
                
                response = self.session.get(
                    config['url'],
                    params=config['params'],
                    headers=config['headers'],
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process API response
                    api_articles = data.get('articles', []) or data.get('data', [])
                    
                    for article in api_articles:
                        try:
                            title = article.get('title', '')
                            url = article.get('url', '')
                            description = article.get('description', '') or article.get('content', '')
                            published_at = article.get('publishedAt', '') or article.get('published_at', '')
                            
                            # Parse date
                            try:
                                if 'T' in published_at:
                                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                else:
                                    pub_date = datetime.now()
                            except:
                                pub_date = datetime.now()
                            
                            # Filter for target timeframe
                            if not (datetime(2021, 6, 1) <= pub_date <= datetime(2024, 6, 1)):
                                continue
                            
                            # Check relevance
                            text = (title + " " + description).lower()
                            if any(kw in text for kw in ['ethereum', 'crypto', 'blockchain', 'defi']):
                                article_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                month_key = pub_date.strftime('%Y-%m')
                                
                                articles.append({
                                    'id': f"api_{api_name}_{month_key}_{article_id}",
                                    'source': f'api_{api_name}_{month_key}',
                                    'title': title,
                                    'content': description,
                                    'published_at': pub_date.replace(tzinfo=None),
                                    'url': url
                                })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse {api_name} article: {e}")
                            continue
                
                time.sleep(2)  # API rate limiting
                
            except Exception as e:
                logging.warning(f"News API {api_name} failed: {e}")
                continue
        
        logging.info(f"News APIs: {len(articles)} articles collected")
        return articles
    
    def parallel_ultimate_rss_collection(self, max_workers=12):
        """Massive parallel RSS collection from all tiers."""
        logging.info("Starting ultimate parallel RSS collection...")
        
        all_articles = []
        total_feeds = sum(len(feeds) for feeds in self.ultimate_rss_feeds.values())
        
        # Flatten all RSS feeds
        all_rss_feeds = []
        for tier_name, feeds in self.ultimate_rss_feeds.items():
            for feed in feeds:
                all_rss_feeds.append((feed, tier_name))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_feed = {
                executor.submit(self.scrape_rss_with_historical, feed_url): (feed_url, tier)
                for feed_url, tier in all_rss_feeds
            }
            
            completed = 0
            for future in as_completed(future_to_feed):
                feed_url, tier = future_to_feed[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    completed += 1
                    
                    if completed % 20 == 0:
                        logging.info(f"RSS Progress: {completed}/{total_feeds} feeds processed")
                    
                except Exception as e:
                    logging.error(f"RSS feed failed {feed_url}: {e}")
        
        logging.info(f"Ultimate RSS collection: {len(all_articles)} articles from {total_feeds} feeds")
        return all_articles
    
    def save_ultimate_data(self, articles, source_name):
        """Enhanced data saving with detailed metrics."""
        if not articles:
            logging.info(f"No articles from {source_name}")
            return 0
        
        # Enhanced deduplication
        df = pd.DataFrame(articles)
        
        # Remove duplicates by multiple fields
        df = df.drop_duplicates(subset=['id'])
        df = df.drop_duplicates(subset=['url'])
        df = df.drop_duplicates(subset=['title', 'source'])
        
        # Filter for new records
        new_articles = []
        for _, item in df.iterrows():
            if not self.db.record_exists('news_articles', item['id']):
                new_articles.append(item.to_dict())
        
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            inserted = self.db.insert_news_articles(new_df)
            
            # Log detailed metrics
            months_covered = new_df['published_at'].dt.strftime('%Y-%m').nunique()
            avg_relevance = new_df.get('relevance_score', pd.Series([0])).mean()
            
            logging.info(f"{source_name}: {len(articles)} collected → {inserted} new saved")
            logging.info(f"  Months covered: {months_covered}, Avg relevance: {avg_relevance:.1f}")
            return inserted
        
        logging.info(f"{source_name}: {len(articles)} collected → 0 new (duplicates)")
        return 0
    
    def ultimate_collection_campaign(self):
        """Execute the ultimate comprehensive collection campaign."""
        logging.info("=== ULTIMATE COLLECTION CAMPAIGN START ===")
        
        start_time = datetime.now()
        collection_stats = {}
        
        # Phase 1: Ultimate parallel RSS collection
        logging.info("Phase 1: Ultimate RSS collection from 100+ feeds")
        rss_articles = self.parallel_ultimate_rss_collection()
        rss_saved = self.save_ultimate_data(rss_articles, "ultimate_rss")
        collection_stats['ultimate_rss'] = rss_saved
        
        # Phase 2: Comprehensive Google News mega-search
        logging.info("Phase 2: Google News mega-search (500+ searches)")
        google_articles = self.comprehensive_google_news_search()
        google_saved = self.save_ultimate_data(google_articles, "google_mega_search")
        collection_stats['google_mega_search'] = google_saved
        
        # Phase 3: News APIs (if available)
        logging.info("Phase 3: News API collection")
        api_articles = self.collect_news_apis()
        api_saved = self.save_ultimate_data(api_articles, "news_apis")
        collection_stats['news_apis'] = api_saved
        
        total_saved = sum(collection_stats.values())
        elapsed = datetime.now() - start_time
        
        # Comprehensive summary
        total_rss_feeds = sum(len(feeds) for feeds in self.ultimate_rss_feeds.values())
        total_search_terms = sum(len(terms) for terms in self.comprehensive_search_terms.values())
        
        logging.info("=== ULTIMATE COLLECTION CAMPAIGN COMPLETE ===")
        logging.info(f"RSS feeds processed: {total_rss_feeds}")
        logging.info(f"Google searches executed: {total_search_terms * 10}")
        logging.info(f"News APIs queried: {len(self.news_apis)}")
        logging.info(f"Total articles collected: {total_saved}")
        logging.info(f"Collection rate: {total_saved/(elapsed.total_seconds()/60):.1f} articles/minute")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        # Detailed breakdown
        for source, count in collection_stats.items():
            logging.info(f"  {source}: {count} articles")
        
        return total_saved, collection_stats

def main():
    """Main function for ultimate collection system."""
    system = UltimateCollectionSystem()
    
    total_rss = sum(len(feeds) for feeds in system.ultimate_rss_feeds.values())
    total_searches = sum(len(terms) for terms in system.comprehensive_search_terms.values()) * 10
    
    print(f"🚀 ULTIMATE COLLECTION SYSTEM")
    print(f"📡 RSS Feeds: {total_rss} international sources")
    print(f"🔍 Google Searches: {total_searches} comprehensive queries")
    print(f"🌐 News APIs: {len(system.news_apis)} professional sources")
    print(f"🎯 Target: 2,000+ articles to reach 15+ entries/day")
    print(f"⏱️ Expected time: 10-15 minutes")
    print(f"📊 Goal: Achieve professional ML training standards")
    
    total, stats = system.ultimate_collection_campaign()
    
    print(f"\n🎉 ULTIMATE CAMPAIGN COMPLETE!")
    print(f"📰 Total articles added: {total}")
    print(f"📊 Run final analysis to verify 15+ entries/day target")
    
    for source, count in stats.items():
        print(f"  ✅ {source}: {count} articles")

if __name__ == "__main__":
    main()