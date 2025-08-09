#!/usr/bin/env python3
# collection/event_focused_collection.py

"""
Event-focused collection targeting specific crypto events in 2022-2023.
Focus on major market events that would have generated significant discussion.
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
        logging.FileHandler('logs/event_focused_collection.log'),
        logging.StreamHandler()
    ]
)

class EventFocusedCollection:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Major crypto events in 2022-2023 that would generate discussion
        self.major_events = {
            '2022-01': [
                'crypto market crash january 2022',
                'ethereum gas fees spike 2022',
                'defi protocol hacks 2022'
            ],
            '2022-02': [
                'russia ukraine crypto sanctions',
                'ethereum layer 2 growth',
                'nft market peak 2022'
            ],
            '2022-03': [
                'terra luna ecosystem growth',
                'ethereum scaling solutions',
                'crypto regulation europe'
            ],
            '2022-04': [
                'ethereum merge testnet',
                'defi tvl decline 2022',
                'crypto venture funding'
            ],
            '2022-05': [
                'terra luna collapse UST',
                'ethereum merge delay',
                'crypto market crash may'
            ],
            '2022-06': [
                'celsius network freeze',
                'three arrows capital 3AC',
                'ethereum proof of stake'
            ],
            '2022-07': [
                'ethereum merge goerli testnet',
                'crypto lending crisis',
                'ethereum gas optimization'
            ],
            '2022-08': [
                'ethereum merge september date',
                'tornado cash sanctions',
                'ethereum beacon chain'
            ],
            '2022-09': [
                'ethereum merge successful',
                'ethereum proof of stake complete',
                'ethereum energy consumption'
            ],
            '2022-10': [
                'ethereum staking withdrawals',
                'defi autumn 2022',
                'ethereum scalability improvements'
            ],
            '2022-11': [
                'ftx exchange collapse',
                'sam bankman fried arrest',
                'crypto contagion fears'
            ],
            '2022-12': [
                'ethereum shanghai upgrade',
                'crypto winter deepens',
                'ethereum development roadmap'
            ],
            '2023-01': [
                'ethereum staking surge',
                'crypto regulation clarity',
                'ethereum layer 2 adoption'
            ],
            '2023-02': [
                'ethereum shanghai testnet',
                'crypto market recovery',
                'ethereum validator queue'
            ],
            '2023-03': [
                'silicon valley bank crypto',
                'ethereum staking derivatives',
                'crypto banking crisis'
            ],
            '2023-04': [
                'ethereum shanghai upgrade live',
                'ethereum withdrawals enabled',
                'liquid staking boom'
            ],
            '2023-05': [
                'ethereum staking centralization',
                'crypto market bull run',
                'ethereum mev boost'
            ],
            '2023-06': [
                'ethereum blob transactions',
                'crypto etf applications',
                'ethereum dencun upgrade'
            ],
            '2023-07': [
                'ethereum layer 2 summer',
                'crypto institutional adoption',
                'ethereum restaking protocols'
            ],
            '2023-08': [
                'ethereum fee burn mechanism',
                'crypto market volatility',
                'ethereum scaling wars'
            ],
            '2023-09': [
                'ethereum one year merge',
                'crypto regulation progress',
                'ethereum roadmap updates'
            ],
            '2023-10': [
                'ethereum cancun upgrade prep',
                'crypto etf optimism',
                'ethereum scaling solutions'
            ],
            '2023-11': [
                'ethereum dencun testnet',
                'crypto market surge',
                'ethereum proto danksharding'
            ],
            '2023-12': [
                'ethereum year end review',
                'crypto market 2024 outlook',
                'ethereum scaling roadmap'
            ]
        }
    
    def collect_event_news(self, event_terms, month):
        """Collect news for specific events in a month."""
        articles = []
        
        year = int(month.split('-')[0])
        month_num = int(month.split('-')[1])
        
        # Create date range for the month
        start_date = datetime(year, month_num, 1)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
        
        for event_term in event_terms:
            try:
                # Create targeted search query
                search_query = f'"{event_term}" after:{start_date.strftime("%Y-%m-%d")} before:{end_date.strftime("%Y-%m-%d")}'
                encoded_query = quote(search_query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:15]:  # Limit per search
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse date
                        pub_date = start_date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Verify date is in target month
                        if not (start_date <= pub_date <= end_date):
                            continue
                        
                        # Check for crypto relevance
                        text = (title + " " + summary).lower()
                        if any(keyword in text for keyword in ['ethereum', 'crypto', 'bitcoin', 'blockchain', 'defi']):
                            article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                            
                            articles.append({
                                'id': f"event_{month}_{article_id}",
                                'source': f'event_focus_{month}',
                                'title': title,
                                'content': summary,
                                'published_at': pub_date,
                                'url': link,
                                'event_term': event_term
                            })
                    
                    except Exception as e:
                        logging.debug(f"Failed to parse event entry: {e}")
                        continue
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"Event search failed for {event_term}: {e}")
                continue
        
        return articles
    
    def collect_social_discussions(self, event_terms, month):
        """Collect social media discussions around events."""
        posts = []
        
        # Use Google to find Reddit/Twitter discussions about events
        for event_term in event_terms[:2]:  # Limit for efficiency
            try:
                # Search Reddit discussions
                reddit_query = f'site:reddit.com/r/ethereum "{event_term}" OR site:reddit.com/r/cryptocurrency "{event_term}"'
                google_url = f"https://www.google.com/search?q={quote(reddit_query)}&num=10"
                
                response = self.session.get(google_url, timeout=15)
                
                if response.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for result in soup.find_all('div', class_='g')[:5]:
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
                            if any(term in text for term in ['ethereum', 'eth', 'crypto']):
                                post_id = hashlib.md5(url.encode()).hexdigest()[:10]
                                
                                posts.append({
                                    'id': f"social_event_{month}_{post_id}",
                                    'source': f'social_event_{month}',
                                    'title': title,
                                    'content': snippet,
                                    'published_at': datetime.strptime(f"{month}-15", '%Y-%m-%d'),
                                    'url': url,
                                    'event_term': event_term
                                })
                        
                        except Exception as e:
                            logging.debug(f"Failed to parse social result: {e}")
                            continue
                
                time.sleep(3)
                
            except Exception as e:
                logging.warning(f"Social collection failed for {event_term}: {e}")
                continue
        
        return posts
    
    def parallel_event_collection(self):
        """Collect event-focused data in parallel."""
        logging.info("Starting parallel event-focused collection...")
        
        all_data = []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit tasks for each month
            futures = []
            
            for month, events in list(self.major_events.items())[:12]:  # First year for efficiency
                future_news = executor.submit(self.collect_event_news, events, month)
                future_social = executor.submit(self.collect_social_discussions, events, month)
                
                futures.append((future_news, f"news_{month}"))
                futures.append((future_social, f"social_{month}"))
            
            # Collect results
            for future, task_name in futures:
                try:
                    data = future.result()
                    all_data.extend(data)
                    logging.info(f"✅ {task_name}: {len(data)} items collected")
                except Exception as e:
                    logging.error(f"❌ {task_name} failed: {e}")
        
        return all_data
    
    def save_event_data(self, data, source_name):
        """Save event-focused data to database."""
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
    
    def event_focused_campaign(self):
        """Execute event-focused collection campaign."""
        logging.info("=== EVENT-FOCUSED COLLECTION CAMPAIGN ===")
        
        start_time = datetime.now()
        
        # Collect event-focused data
        event_data = self.parallel_event_collection()
        saved = self.save_event_data(event_data, "event_focused")
        
        elapsed = datetime.now() - start_time
        
        logging.info("=== EVENT-FOCUSED COLLECTION COMPLETE ===")
        logging.info(f"Total events processed: {len(self.major_events)}")
        logging.info(f"Total new entries: {saved}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return saved

def main():
    """Main function for event-focused collection."""
    collector = EventFocusedCollection()
    
    print(f"🎯 EVENT-FOCUSED COLLECTION")
    print(f"📅 Events: {len(collector.major_events)} months of major crypto events")
    print(f"🔍 Strategy: Target specific events that generated discussion")
    print(f"💡 Expected: 200-500 additional entries from event coverage")
    
    total = collector.event_focused_campaign()
    
    print(f"\n✅ EVENT COLLECTION COMPLETE!")
    print(f"📈 Added {total} new entries")
    print(f"🎯 Focused on major crypto events in 2022-2023")

if __name__ == "__main__":
    main()