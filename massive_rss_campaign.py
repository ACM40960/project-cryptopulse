#!/usr/bin/env python3
# massive_rss_campaign.py

"""
Massive RSS campaign to reach professional ML standards through comprehensive feed collection.
Target: 6,000+ additional entries to achieve 15+ entries/day average.
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
import itertools

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/massive_rss_campaign.log'),
        logging.StreamHandler()
    ]
)

class MassiveRSSCampaign:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Expanded comprehensive RSS feeds
        self.rss_feeds = {
            'tier1_major': [
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed', 
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://thedefiant.io/feed',
                'https://blockworks.co/feed',
                'https://bitcoinmagazine.com/.rss/full/',
            ],
            'tier2_crypto_news': [
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
                'https://www.bitcoininsider.org/feed',
                'https://bitcoinist.com/feed/',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://www.theblockcrypto.com/rss.xml',
            ],
            'tier3_ethereum_focused': [
                'https://blog.ethereum.org/feed.xml',
                'https://consensys.net/blog/feed/',
                'https://medium.com/feed/@VitalikButerin',
                'https://medium.com/feed/ethereum-foundation',
                'https://medium.com/feed/ethereum-cat-herders',
                'https://medium.com/feed/ethereum-foundation',
                'https://week-in-ethereum.substack.com/feed',
                'https://ethresear.ch/latest.rss',
            ],
            'tier4_defi_web3': [
                'https://thedefiant.io/feed',
                'https://defipulse.com/blog/feed',
                'https://defillama.com/blog/rss.xml',
                'https://banklesshq.com/rss/',
                'https://newsletter.banklesshq.com/feed',
            ],
            'tier5_trading_markets': [
                'https://www.coindesk.com/markets/rss/',
                'https://markets.businessinsider.com/rss/news',
                'https://finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US',
                'https://www.marketwatch.com/rss/cryptocurrency',
                'https://seekingalpha.com/api/sa/combined/ETHUSD.xml',
            ],
            'tier6_reddit_social': [
                'https://www.reddit.com/r/ethereum.rss',
                'https://www.reddit.com/r/CryptoCurrency.rss', 
                'https://www.reddit.com/r/ethtrader.rss',
                'https://www.reddit.com/r/defi.rss',
                'https://www.reddit.com/r/ethereum.rss?limit=100',
                'https://www.reddit.com/r/ethfinance.rss',
                'https://www.reddit.com/r/ethstaker.rss',
            ],
            'tier7_mainstream_crypto': [
                'https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml&size=100&tagName=crypto',
                'https://www.bloomberg.com/feeds/podcasts/crypto.xml',
                'https://feeds.bloomberg.com/crypto/news.rss',
                'https://www.cnbc.com/id/100727362/device/rss/rss.html',
                'https://cnn.com/services/rss/?arch=rss',
            ],
            'tier8_international': [
                'https://www.financemagnates.com/feed/',
                'https://en.ethereumworldnews.com/feed/',
                'https://www.investinblockchain.com/feed/',
                'https://bitcoinexchangeguide.com/feed/',
                'https://www.coininsider.com/feed/',
                'https://cryptoslate.com/feed/',
            ]
        }
        
        # Google News search terms for comprehensive coverage
        self.google_news_terms = [
            # Price and market terms
            'ethereum price 2022', 'ethereum price 2023', 'ETH cryptocurrency analysis',
            'ethereum market analysis', 'ethereum price prediction', 'ethereum trading',
            
            # Technology terms  
            'ethereum merge upgrade', 'ethereum proof of stake transition', 'ethereum staking',
            'ethereum london hard fork', 'ethereum EIP upgrade', 'ethereum scalability',
            
            # DeFi and ecosystem
            'ethereum defi protocol news', 'ethereum smart contracts', 'ethereum dapps',
            'ethereum layer 2 solutions', 'ethereum arbitrum optimism', 'ethereum polygon',
            
            # Market events
            'ethereum institutional adoption', 'ethereum ETF approval', 'ethereum regulation',
            'ethereum hack exploit news', 'ethereum security audit', 'ethereum vulnerability',
            
            # Ecosystem and community
            'vitalik buterin ethereum', 'ethereum foundation news', 'ethereum developer update',
            'ethereum conference event', 'ethereum partnership announcement', 'ethereum integration'
        ]
    
    def scrape_rss_comprehensive(self, rss_url, timeframe_start='2022-01-01', timeframe_end='2024-01-01'):
        """Comprehensive RSS scraping with broader date range."""
        articles = []
        
        try:
            logging.info(f"Scraping: {rss_url}")
            
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                logging.warning(f"No entries in feed: {rss_url}")
                return articles
            
            for entry in feed.entries:
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    
                    # Parse publication date with multiple fallbacks
                    pub_date = datetime.now()
                    date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
                    
                    for field in date_fields:
                        if hasattr(entry, field) and getattr(entry, field):
                            try:
                                pub_date = datetime(*getattr(entry, field)[:6])
                                break
                            except:
                                continue
                    
                    # Filter timeframe (more permissive)
                    start_date = datetime.strptime(timeframe_start, '%Y-%m-%d')
                    end_date = datetime.strptime(timeframe_end, '%Y-%m-%d')
                    
                    if not (start_date <= pub_date <= end_date):
                        continue
                    
                    # Expanded crypto relevance check
                    text_content = (title + " " + summary).lower()
                    
                    crypto_keywords = [
                        # Core terms
                        'ethereum', 'eth', 'crypto', 'cryptocurrency', 'bitcoin', 'blockchain',
                        # DeFi terms
                        'defi', 'decentralized', 'smart contract', 'dapp', 'protocol', 'yield',
                        'liquidity', 'staking', 'farming', 'swap', 'dex', 'dao', 'governance',
                        # NFT and Web3
                        'nft', 'web3', 'metaverse', 'opensea', 'collectible', 'digital art',
                        # Technical terms
                        'consensus', 'proof of stake', 'proof of work', 'mining', 'validator',
                        'gas fee', 'transaction', 'block', 'hash', 'wallet', 'exchange',
                        # Market terms
                        'trading', 'price', 'market', 'bull', 'bear', 'rally', 'crash',
                        'investment', 'institutional', 'etf', 'futures', 'derivative',
                        # Layer 2 and scaling
                        'arbitrum', 'optimism', 'polygon', 'layer 2', 'scaling', 'rollup'
                    ]
                    
                    relevance_score = sum(1 for keyword in crypto_keywords if keyword in text_content)
                    
                    # Require at least 2 crypto keywords for better relevance
                    if relevance_score >= 2 or 'ethereum' in text_content:
                        article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                        month_key = pub_date.strftime('%Y-%m')
                        
                        # Extract clean source name
                        source_name = rss_url.split('//')[1].split('/')[0].replace('www.', '').replace('.com', '').replace('.org', '')
                        
                        articles.append({
                            'id': f"massive_rss_{source_name}_{month_key}_{article_id}",
                            'source': f'massive_rss_{source_name}_{month_key}',
                            'title': title,
                            'content': summary,
                            'published_at': pub_date,
                            'url': link
                        })
                    
                except Exception as e:
                    logging.debug(f"Failed to parse entry: {e}")
                    continue
            
            logging.info(f"RSS {rss_url}: {len(articles)} articles extracted")
            return articles
            
        except Exception as e:
            logging.warning(f"RSS failed {rss_url}: {e}")
            return articles
    
    def google_news_comprehensive_search(self):
        """Comprehensive Google News search for 2022-2023."""
        logging.info("Starting comprehensive Google News search...")
        
        articles = []
        
        # Date ranges for thorough coverage
        date_ranges = [
            ('2022-01-01', '2022-06-30'),
            ('2022-07-01', '2022-12-31'), 
            ('2023-01-01', '2023-06-30'),
            ('2023-07-01', '2023-12-31')
        ]
        
        for search_term in self.google_news_terms:
            for start_date, end_date in date_ranges:
                try:
                    # Create comprehensive search query
                    search_query = f'{search_term} after:{start_date} before:{end_date}'
                    encoded_query = quote(search_query)
                    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                    
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:15]:  # More entries per search
                        try:
                            title = entry.get('title', '')
                            link = entry.get('link', '')
                            summary = entry.get('summary', '')
                            
                            # Parse date
                            pub_date = datetime.now()
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            
                            # Check timeframe
                            range_start = datetime.strptime(start_date, '%Y-%m-%d')
                            range_end = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            if not (range_start <= pub_date <= range_end):
                                continue
                            
                            # Check relevance
                            text = (title + " " + summary).lower()
                            if any(keyword in text for keyword in ['ethereum', 'crypto', 'blockchain', 'defi']):
                                article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                month_key = pub_date.strftime('%Y-%m')
                                
                                articles.append({
                                    'id': f"massive_google_{month_key}_{article_id}",
                                    'source': f'massive_google_{month_key}',
                                    'title': title,
                                    'content': summary,
                                    'published_at': pub_date,
                                    'url': link
                                })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse Google News entry: {e}")
                            continue
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logging.warning(f"Google News search failed for {search_term} {start_date}-{end_date}: {e}")
                    continue
        
        return articles
    
    def parallel_rss_tier_collection(self, tier_name, rss_list, max_workers=8):
        """Collect from RSS tier in parallel."""
        logging.info(f"Processing {tier_name}: {len(rss_list)} feeds")
        
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.scrape_rss_comprehensive, url): url 
                for url in rss_list
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    logging.info(f"✅ {url}: {len(articles)} articles")
                except Exception as e:
                    logging.error(f"❌ {url}: {e}")
        
        logging.info(f"{tier_name} complete: {len(all_articles)} total articles")
        return all_articles
    
    def save_massive_data(self, articles, source_name):
        """Save articles with deduplication."""
        if not articles:
            return 0
        
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
            logging.info(f"{source_name}: {len(articles)} collected → {inserted} new")
            return inserted
        
        logging.info(f"{source_name}: {len(articles)} collected → 0 new (duplicates)")
        return 0
    
    def massive_rss_campaign(self):
        """Execute massive RSS collection campaign."""
        logging.info("=== MASSIVE RSS CAMPAIGN START ===")
        
        start_time = datetime.now()
        total_collected = 0
        
        # Phase 1: Process all RSS tiers
        for tier_name, rss_list in self.rss_feeds.items():
            try:
                tier_articles = self.parallel_rss_tier_collection(tier_name, rss_list)
                saved = self.save_massive_data(tier_articles, tier_name)
                total_collected += saved
                
                logging.info(f"🎯 {tier_name}: {saved} new articles saved")
                
            except Exception as e:
                logging.error(f"❌ {tier_name} failed: {e}")
                continue
        
        # Phase 2: Comprehensive Google News search
        try:
            logging.info("Phase 2: Comprehensive Google News search")
            google_articles = self.google_news_comprehensive_search()
            google_saved = self.save_massive_data(google_articles, "comprehensive_google_news")
            total_collected += google_saved
            
            logging.info(f"🎯 Google News: {google_saved} new articles saved")
            
        except Exception as e:
            logging.error(f"❌ Google News phase failed: {e}")
        
        elapsed = datetime.now() - start_time
        
        # Summary
        total_feeds = sum(len(feeds) for feeds in self.rss_feeds.values())
        
        logging.info("=== MASSIVE RSS CAMPAIGN COMPLETE ===")
        logging.info(f"RSS feeds processed: {total_feeds}")
        logging.info(f"Google News searches: {len(self.google_news_terms) * 4}")
        logging.info(f"Total new articles: {total_collected}")
        logging.info(f"Collection rate: {total_collected/(elapsed.total_seconds()/60):.1f} articles/minute")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for massive RSS campaign."""
    campaign = MassiveRSSCampaign()
    
    total_feeds = sum(len(feeds) for feeds in campaign.rss_feeds.values())
    
    print(f"🚀 MASSIVE RSS COLLECTION CAMPAIGN")
    print(f"📡 RSS Feeds: {total_feeds} comprehensive sources")
    print(f"🔍 Google News: {len(campaign.google_news_terms)} search terms × 4 periods") 
    print(f"🎯 Target: 1,000+ articles for significant density boost")
    print(f"⏱️ Expected time: 5-10 minutes")
    
    total = campaign.massive_rss_campaign()
    
    print(f"\n🎉 MASSIVE CAMPAIGN COMPLETE!")
    print(f"📰 Added {total} new articles")
    print(f"🎯 Run final dataset analysis to verify improvements")

if __name__ == "__main__":
    main()