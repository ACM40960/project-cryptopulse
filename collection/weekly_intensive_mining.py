#!/usr/bin/env python3
# collection/weekly_intensive_mining.py

"""
Weekly intensive mining - break down collection to weekly periods
and use different search strategies to find missed content.
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
import calendar

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/weekly_intensive_mining.log'),
        logging.StreamHandler()
    ]
)

class WeeklyIntensiveMining:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Different search term strategies
        self.search_strategies = {
            'price_focused': [
                'ethereum price', 'ETH price', 'ethereum trading', 'ethereum market',
                'ethereum bull', 'ethereum bear', 'ethereum pump', 'ethereum dump'
            ],
            'tech_focused': [
                'ethereum upgrade', 'ethereum merge', 'ethereum staking', 'ethereum validator',
                'ethereum gas', 'ethereum layer', 'ethereum scaling', 'ethereum fork'
            ],
            'defi_focused': [
                'ethereum defi', 'ethereum yield', 'ethereum liquidity', 'ethereum protocol',
                'ethereum dapp', 'ethereum smart contract', 'ethereum dao', 'ethereum governance'
            ],
            'news_focused': [
                'ethereum news', 'ethereum update', 'ethereum announcement', 'ethereum development',
                'ethereum partnership', 'ethereum integration', 'ethereum adoption', 'ethereum regulation'
            ]
        }
    
    def get_weekly_periods(self, year):
        """Generate weekly periods for intensive collection."""
        weeks = []
        start_date = datetime(year, 1, 1)
        
        # Find first Monday of the year
        while start_date.weekday() != 0:
            start_date += timedelta(days=1)
        
        current_date = start_date
        end_of_year = datetime(year, 12, 31)
        
        while current_date <= end_of_year:
            week_end = current_date + timedelta(days=6)
            if week_end > end_of_year:
                week_end = end_of_year
            
            weeks.append((current_date, week_end))
            current_date = week_end + timedelta(days=1)
        
        return weeks
    
    def intensive_weekly_search(self, start_date, end_date, strategy_name, search_terms):
        """Intensive search for a specific week using specific strategy."""
        articles = []
        
        week_str = start_date.strftime('%Y-W%U')
        
        for search_term in search_terms[:3]:  # Limit terms per week
            try:
                # Create precise weekly search
                search_query = f'"{search_term}" after:{start_date.strftime("%Y-%m-%d")} before:{end_date.strftime("%Y-%m-%d")}'
                encoded_query = quote(search_query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:8]:  # Limit per search
                    try:
                        title = entry.get('title', '')
                        link = entry.get('link', '')
                        summary = entry.get('summary', '')
                        
                        # Parse date
                        pub_date = start_date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Verify date is in target week
                        if not (start_date <= pub_date <= end_date):
                            continue
                        
                        # Check relevance
                        text = (title + " " + summary).lower()
                        if any(keyword in text for keyword in ['ethereum', 'crypto', 'blockchain', 'defi']):
                            article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                            
                            articles.append({
                                'id': f"weekly_{strategy_name}_{week_str}_{article_id}",
                                'source': f'weekly_{strategy_name}_{week_str}',
                                'title': title,
                                'content': summary,
                                'published_at': pub_date,
                                'url': link,
                                'strategy': strategy_name,
                                'search_term': search_term
                            })
                    
                    except Exception as e:
                        logging.debug(f"Failed to parse weekly entry: {e}")
                        continue
                
                time.sleep(0.5)  # Quick rate limiting
                
            except Exception as e:
                logging.warning(f"Weekly search failed for {search_term}: {e}")
                continue
        
        return articles
    
    def collect_high_activity_weeks(self, year, max_weeks=20):
        """Focus on high-activity weeks that likely had more content."""
        logging.info(f"Collecting high-activity weeks for {year}...")
        
        all_articles = []
        weeks = self.get_weekly_periods(year)
        
        # Focus on key weeks (every 2-3 weeks to spread coverage)
        selected_weeks = weeks[::3][:max_weeks]  # Every 3rd week, up to max_weeks
        
        for i, (start_date, end_date) in enumerate(selected_weeks):
            week_str = start_date.strftime('%Y-W%U')
            logging.info(f"Processing week {i+1}/{len(selected_weeks)}: {week_str}")
            
            # Rotate through strategies for different weeks
            strategy_name = list(self.search_strategies.keys())[i % len(self.search_strategies)]
            search_terms = self.search_strategies[strategy_name]
            
            try:
                week_articles = self.intensive_weekly_search(start_date, end_date, strategy_name, search_terms)
                all_articles.extend(week_articles)
                logging.info(f"Week {week_str}: {len(week_articles)} articles collected")
                
            except Exception as e:
                logging.error(f"Week {week_str} failed: {e}")
                continue
        
        return all_articles
    
    def targeted_month_boost(self, year, month):
        """Boost specific months with comprehensive daily searches."""
        logging.info(f"Boosting {year}-{month:02d}...")
        
        articles = []
        
        # Get all days in the month
        start_date = datetime(year, month, 1)
        _, last_day = calendar.monthrange(year, month)
        end_date = datetime(year, month, last_day)
        
        # Create daily searches for the month
        current_date = start_date
        while current_date <= end_date:
            day_str = current_date.strftime('%Y-%m-%d')
            
            # Use mixed strategy for daily search
            mixed_terms = [
                f'ethereum {day_str}',
                f'crypto news {day_str}',
                f'ETH price {day_str}'
            ]
            
            for term in mixed_terms:
                try:
                    search_query = f'"{term}"'
                    encoded_query = quote(search_query)
                    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
                    
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries[:5]:
                        try:
                            title = entry.get('title', '')
                            link = entry.get('link', '')
                            summary = entry.get('summary', '')
                            
                            # Parse date
                            pub_date = current_date
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            
                            # Check if same day
                            if pub_date.date() == current_date.date():
                                text = (title + " " + summary).lower()
                                if any(keyword in text for keyword in ['ethereum', 'crypto', 'blockchain']):
                                    article_id = hashlib.md5(link.encode()).hexdigest()[:10]
                                    
                                    articles.append({
                                        'id': f"daily_boost_{day_str}_{article_id}",
                                        'source': f'daily_boost_{year}_{month:02d}',
                                        'title': title,
                                        'content': summary,
                                        'published_at': pub_date,
                                        'url': link
                                    })
                        
                        except Exception as e:
                            logging.debug(f"Failed to parse daily entry: {e}")
                            continue
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    logging.debug(f"Daily search failed: {e}")
                    continue
            
            current_date += timedelta(days=1)
        
        return articles
    
    def save_weekly_data(self, data, source_name):
        """Save weekly mining data to database."""
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
    
    def weekly_intensive_campaign(self):
        """Execute weekly intensive mining campaign."""
        logging.info("=== WEEKLY INTENSIVE MINING CAMPAIGN ===")
        
        start_time = datetime.now()
        total_collected = 0
        
        # Strategy 1: High-activity weeks for 2022 and 2023
        logging.info("Phase 1: High-activity weeks collection")
        weeks_2022 = self.collect_high_activity_weeks(2022, max_weeks=15)
        saved_2022 = self.save_weekly_data(weeks_2022, "weekly_2022")
        total_collected += saved_2022
        
        weeks_2023 = self.collect_high_activity_weeks(2023, max_weeks=15)
        saved_2023 = self.save_weekly_data(weeks_2023, "weekly_2023")
        total_collected += saved_2023
        
        # Strategy 2: Targeted month boost for key months
        logging.info("Phase 2: Targeted month boost")
        key_months = [
            (2022, 5),  # Terra Luna collapse
            (2022, 9),  # Ethereum merge
            (2022, 11), # FTX collapse
            (2023, 4),  # Shanghai upgrade
        ]
        
        for year, month in key_months:
            month_data = self.targeted_month_boost(year, month)
            saved_month = self.save_weekly_data(month_data, f"month_boost_{year}_{month}")
            total_collected += saved_month
        
        elapsed = datetime.now() - start_time
        
        logging.info("=== WEEKLY INTENSIVE MINING COMPLETE ===")
        logging.info(f"Total new entries: {total_collected}")
        logging.info(f"Execution time: {elapsed.total_seconds():.1f} seconds")
        
        return total_collected

def main():
    """Main function for weekly intensive mining."""
    miner = WeeklyIntensiveMining()
    
    print(f"📅 WEEKLY INTENSIVE MINING")
    print(f"🎯 Strategy: Break down to weekly periods for deeper coverage")
    print(f"🔍 Focus: High-activity weeks + key event months")
    print(f"💡 Expected: 100-300 additional entries from intensive mining")
    
    total = miner.weekly_intensive_campaign()
    
    print(f"\n✅ WEEKLY MINING COMPLETE!")
    print(f"📈 Added {total} new entries")
    print(f"🎯 Intensive weekly and daily searches completed")

if __name__ == "__main__":
    main()