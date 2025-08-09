#!/usr/bin/env python3
"""
Comprehensive Twitter Data Expansion System

OBJECTIVE: Collect 25,000+ high-quality tweets from crypto influencers (2022-2025)
TARGET: 10x improvement in ML dataset size (178 → 1,720+ samples)

Strategy:
1. Exhaustive influencer list across all crypto categories
2. Deep historical scraping of complete feeds 
3. Smart rate limiting and authentication handling
4. Deduplication and quality filtering
5. Temporal alignment with price data for ML features

IMPORTANT: Handles Twitter login requirements and rate limits gracefully
"""

import os
import time
import random
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from database import CryptoPulseDB

load_dotenv()

@dataclass
class InfluencerProfile:
    username: str
    category: str
    priority: int  # 1=highest, 5=lowest
    description: str
    estimated_tweets_per_day: int

class ComprehensiveTwitterExpansion:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.driver = None
        self.setup_logging()
        
        # Comprehensive influencer database
        self.influencers = self.build_comprehensive_influencer_list()
        
        # Collection parameters
        self.target_start_date = datetime(2022, 1, 1)
        self.target_end_date = datetime(2025, 8, 1)
        self.tweets_collected = 0
        self.target_tweets = 25000
        
        # Rate limiting
        self.requests_made = 0
        self.last_request_time = time.time()
        self.rate_limit_delay = 2  # seconds between requests
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/twitter_expansion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def build_comprehensive_influencer_list(self) -> List[InfluencerProfile]:
        """Build exhaustive list of crypto influencers across all categories"""
        
        influencers = []
        
        # 🏛️ ETHEREUM FOUNDERS & CORE DEVELOPERS (Highest Priority)
        ethereum_core = [
            ("VitalikButerin", "Ethereum Founder", 1, "Ethereum Creator", 2),
            ("TimBeiko", "Ethereum Core Dev", 1, "EIP Editor & Core Dev", 3),
            ("evan_van_ness", "Ethereum Community", 1, "Week in Ethereum", 5),
            ("drakefjustin", "Ethereum Research", 1, "Ethereum Foundation", 2),
            ("econoar", "Ethereum Core", 1, "Ethereum Core Developer", 3),
            ("dannyryan", "Ethereum Research", 1, "Ethereum 2.0 Researcher", 2),
            ("djrtwo", "Ethereum Research", 1, "Ethereum 2.0 Lead", 1),
            ("JustinDrake_", "Ethereum Research", 1, "Ethereum Researcher", 2),
            ("terencechain", "Ethereum Dev", 1, "Prysm Client Lead", 3),
            ("preston_vanloon", "Ethereum Dev", 1, "Prysm Co-founder", 2)
        ]
        
        # 🏦 DEFI PROTOCOL LEADERS (High Priority)
        defi_leaders = [
            ("haydenzadams", "DeFi", 1, "Uniswap Founder", 4),
            ("bantg", "DeFi", 1, "Yearn Core", 3),
            ("AndreCronjeTech", "DeFi", 1, "DeFi Architect", 2),
            ("divine_economy", "DeFi", 2, "DeFi Educator", 5),
            ("TheDeFiEdge", "DeFi", 2, "DeFi Strategy", 4),
            ("DefiIgnas", "DeFi", 2, "DeFi Analysis", 6),
            ("sassal0x", "DeFi", 2, "DeFi Researcher", 3),
            ("ryandscott", "DeFi", 2, "DeFi Builder", 2),
            ("statelayer", "DeFi", 2, "Maker Protocol", 3),
            ("RuneKek", "DeFi", 2, "THORChain", 4)
        ]
        
        # 💰 CRYPTO VCs & INVESTORS (High Priority)
        crypto_vcs = [
            ("arjunblj", "VC", 1, "Variant Fund", 2),
            ("ljin18", "VC", 1, "Variant Fund", 1),
            ("daostack", "VC", 2, "DAOs & Investment", 3),
            ("austingriffith", "Builder/VC", 2, "BuidlGuidl", 4),
            ("pet3rpan_", "VC", 2, "1kx Network", 2),
            ("jamiepittman", "VC", 2, "Crypto Investor", 1),
            ("CBVentures", "VC", 2, "Coinbase Ventures", 2),
            ("panteracapital", "VC", 2, "Pantera Capital", 1),
            ("a16zcrypto", "VC", 1, "Andreessen Horowitz", 1),
            ("polychain", "VC", 2, "Polychain Capital", 1)
        ]
        
        # 📊 TRADING & ANALYSIS EXPERTS (Medium-High Priority)
        trading_experts = [
            ("CryptoCred", "Trading", 2, "Technical Analysis", 8),
            ("pentosh1", "Trading", 2, "Crypto Analyst", 6),
            ("CryptoKaleo", "Trading", 3, "Swing Trader", 5),
            ("CryptoBirb", "Trading", 3, "Chart Analysis", 7),
            ("il_capo_of_crypto", "Trading", 3, "Market Analysis", 4),
            ("TechDev_52", "Trading", 2, "Technical Analysis", 3),
            ("CryptoMichNL", "Trading", 3, "Market Commentary", 5),
            ("AlexKruger", "Trading", 2, "Macro Analysis", 2),
            ("CryptoHamster", "Trading", 3, "DeFi Trading", 4),
            ("TraderKoz", "Trading", 3, "Derivatives Trading", 3)
        ]
        
        # 📰 CRYPTO MEDIA & JOURNALISTS (Medium Priority)
        crypto_media = [
            ("laurashin", "Media", 2, "Unchained Podcast", 3),
            ("nlw", "Media", 2, "The Breakdown", 4),
            ("ErikVoorhees", "Media", 2, "ShapeShift CEO", 2),
            ("nic__carter", "Media", 2, "Castle Island VC", 3),
            ("APompliano", "Media", 3, "Pomp Podcast", 5),
            ("DocumentingBTC", "Media", 3, "Bitcoin News", 8),
            ("WuBlockchain", "Media", 2, "Crypto Journalist", 10),
            ("adamscochran", "Media", 2, "Crypto Analysis", 4),
            ("MessariCrypto", "Media", 2, "Crypto Research", 6),
            ("coindesk", "Media", 2, "Crypto News", 12)
        ]
        
        # 🏢 EXCHANGE & CEX LEADERS (Medium Priority)
        exchange_leaders = [
            ("cz_binance", "CEX", 2, "Binance CEO", 3),
            ("brian_armstrong", "CEX", 2, "Coinbase CEO", 1),
            ("SBF_FTX", "CEX", 4, "Former FTX (Historical)", 0),  # Historical data only
            ("krakenfx", "CEX", 2, "Kraken Exchange", 4),
            ("kucoincom", "CEX", 3, "KuCoin Exchange", 3),
            ("OKEx", "CEX", 3, "OKX Exchange", 2),
            ("gate_io", "CEX", 3, "Gate.io Exchange", 3),
            ("HuobiGlobal", "CEX", 3, "Huobi Exchange", 2),
            ("cryptocom", "CEX", 2, "Crypto.com", 4),
            ("GeminiTrust", "CEX", 3, "Gemini Exchange", 2)
        ]
        
        # 🎨 NFT & WEB3 INFLUENCERS (Lower Priority but High Volume)
        nft_web3 = [
            ("punk6529", "NFT", 3, "NFT Collector", 6),
            ("garyvee", "NFT", 3, "VeeFriends", 4),
            ("dfinzer", "NFT", 2, "OpenSea CEO", 2),
            ("seedphrase", "Web3", 3, "Web3 Builder", 3),
            ("coopahtroopa", "Web3", 3, "Web3 Investor", 4),
            ("kylesamani", "Web3", 3, "Multicoin Capital", 2),
            ("balajis", "Web3", 2, "Former Coinbase CTO", 3),
            ("naval", "Web3", 2, "AngelList Founder", 1),
            ("elonmusk", "General", 4, "Tesla/SpaceX (Crypto mentions)", 8),
            ("michael_saylor", "General", 3, "MicroStrategy", 2)
        ]
        
        # Build comprehensive list
        all_categories = [
            ethereum_core, defi_leaders, crypto_vcs, trading_experts,
            crypto_media, exchange_leaders, nft_web3
        ]
        
        for category in all_categories:
            for username, cat, priority, desc, tweets_per_day in category:
                influencers.append(InfluencerProfile(
                    username=username,
                    category=cat,
                    priority=priority,
                    description=desc,
                    estimated_tweets_per_day=tweets_per_day
                ))
        
        # Sort by priority (1=highest)
        influencers.sort(key=lambda x: (x.priority, -x.estimated_tweets_per_day))
        
        self.logger.info(f"🎭 Built comprehensive influencer list: {len(influencers)} profiles")
        return influencers
    
    def setup_selenium_driver(self):
        """Setup Selenium with optimized Chrome profile"""
        chrome_options = Options()
        
        # Use existing Chrome profile if available
        chrome_profile_path = "/home/thej/Desktop/CryptoPulse/twitter_profile"
        if os.path.exists(chrome_profile_path):
            chrome_options.add_argument(f"--user-data-dir={chrome_profile_path}")
            self.logger.info("📂 Using existing Chrome profile for authentication")
        
        # Essential Chrome arguments for stability
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Window size for consistency
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Try headless first, fallback to headed if needed
        try_headless = os.getenv('TWITTER_HEADLESS', 'true').lower() == 'true'
        if try_headless:
            chrome_options.add_argument("--headless")
            self.logger.info("🤖 Running in headless mode")
        
        # Stealth mode
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        
        try:
            # Use webdriver-manager to automatically handle Chrome driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Test basic functionality
            self.driver.get("https://www.google.com")
            self.logger.info("✅ Chrome driver initialized and tested successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Chrome driver: {str(e)}")
            if try_headless:
                self.logger.info("🔄 Retrying without headless mode...")
                # Remove headless and try again
                chrome_options.arguments = [arg for arg in chrome_options.arguments if arg != "--headless"]
                try:
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                    self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                    self.driver.get("https://www.google.com")
                    self.logger.info("✅ Chrome driver initialized in headed mode")
                    return True
                except Exception as e2:
                    self.logger.error(f"❌ Failed even in headed mode: {str(e2)}")
            return False
    
    def check_twitter_login(self) -> bool:
        """Check if already logged into Twitter"""
        try:
            self.driver.get("https://twitter.com/home")
            time.sleep(3)
            
            # Check if we're on the home timeline (logged in)
            if "home" in self.driver.current_url.lower():
                self.logger.info("✅ Already logged into Twitter")
                return True
            else:
                self.logger.warning("⚠️ Not logged into Twitter - manual login required")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error checking Twitter login: {str(e)}")
            return False
    
    def prompt_manual_login(self):
        """Prompt user for manual Twitter login"""
        print("\\n" + "="*60)
        print("🔐 TWITTER LOGIN REQUIRED")
        print("="*60)
        print("Please log into Twitter manually in the opened browser window.")
        print("After logging in, press ENTER to continue...")
        print("="*60)
        
        # Open Twitter login page
        self.driver.get("https://twitter.com/login")
        
        # Wait for user to login
        input("Press ENTER after you've logged into Twitter: ")
        
        # Verify login
        if self.check_twitter_login():
            print("✅ Twitter login successful!")
            return True
        else:
            print("❌ Login verification failed. Please try again.")
            return False
    
    def scrape_user_tweets(self, username: str, target_count: int = 500) -> List[Dict]:
        """Scrape tweets from a specific user's timeline"""
        tweets = []
        
        try:
            self.logger.info(f"🐦 Scraping @{username} (target: {target_count} tweets)")
            
            # Navigate to user profile
            profile_url = f"https://twitter.com/{username}"
            self.driver.get(profile_url)
            time.sleep(3)
            
            # Check if profile exists
            if "doesn't exist" in self.driver.page_source.lower():
                self.logger.warning(f"⚠️ Profile @{username} doesn't exist or is suspended")
                return tweets
            
            # Scroll and collect tweets
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 50  # Limit scrolling
            
            while len(tweets) < target_count and scroll_attempts < max_scroll_attempts:
                # Find tweet elements
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                for tweet_element in tweet_elements:
                    if len(tweets) >= target_count:
                        break
                        
                    try:
                        # Extract tweet data
                        tweet_data = self.extract_tweet_data(tweet_element, username)
                        if tweet_data and self.is_within_target_timeframe(tweet_data['created_at']):
                            tweets.append(tweet_data)
                            
                    except Exception as e:
                        self.logger.debug(f"Error extracting tweet: {str(e)}")
                        continue
                
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Check if page height changed (more content loaded)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                    last_height = new_height
                
                # Rate limiting
                self.respect_rate_limits()
            
            self.logger.info(f"✅ Collected {len(tweets)} tweets from @{username}")
            
        except Exception as e:
            self.logger.error(f"❌ Error scraping @{username}: {str(e)}")
        
        return tweets
    
    def extract_tweet_data(self, tweet_element, username: str) -> Optional[Dict]:
        """Extract structured data from a tweet element"""
        try:
            # Tweet text
            text_element = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            content = text_element.text if text_element else ""
            
            # Tweet time (this is tricky, might need adjustment)
            time_element = tweet_element.find_element(By.CSS_SELECTOR, 'time')
            created_at = time_element.get_attribute('datetime') if time_element else datetime.now().isoformat()
            
            # Engagement metrics
            likes = self.extract_metric(tweet_element, 'like')
            retweets = self.extract_metric(tweet_element, 'retweet')
            replies = self.extract_metric(tweet_element, 'reply')
            
            return {
                'username': username,
                'content': content,
                'created_at': created_at,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"Error extracting tweet data: {str(e)}")
            return None
    
    def extract_metric(self, tweet_element, metric_type: str) -> int:
        """Extract engagement metrics from tweet"""
        try:
            selector_map = {
                'like': '[data-testid="like"]',
                'retweet': '[data-testid="retweet"]', 
                'reply': '[data-testid="reply"]'
            }
            
            element = tweet_element.find_element(By.CSS_SELECTOR, selector_map.get(metric_type, ''))
            text = element.text.strip()
            
            # Convert text to number
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            elif 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            else:
                return int(text) if text.isdigit() else 0
                
        except:
            return 0
    
    def is_within_target_timeframe(self, created_at_str: str) -> bool:
        """Check if tweet is within our target timeframe"""
        try:
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            return self.target_start_date <= created_at <= self.target_end_date
        except:
            return True  # Include if we can't parse date
    
    def respect_rate_limits(self):
        """Implement rate limiting to avoid being blocked"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.requests_made += 1
        
        # Longer break every 100 requests
        if self.requests_made % 100 == 0:
            self.logger.info(f"💤 Taking longer break after {self.requests_made} requests...")
            time.sleep(30)
    
    def save_tweets_to_database(self, tweets: List[Dict]):
        """Save collected tweets to the database"""
        if not tweets:
            return
        
        saved_count = 0
        for tweet in tweets:
            try:
                # Use the existing insert_twitter_post method
                tweet_data = {
                    'id': f"{tweet['username']}_{hash(tweet['content'])}_{tweet['created_at']}",
                    'username': tweet['username'],
                    'content': tweet['content'],
                    'likes': tweet.get('likes', 0),
                    'retweets': tweet.get('retweets', 0),
                    'replies': tweet.get('replies', 0),
                    'created_at': tweet['created_at'],
                    'url': tweet.get('url', ''),
                    'scraped_at': datetime.now().isoformat()
                }
                
                self.db.insert_twitter_post(tweet_data)
                saved_count += 1
                self.tweets_collected += 1
                
            except Exception as e:
                self.logger.error(f"Error saving tweet: {str(e)}")
        
        if saved_count > 0:
            self.logger.info(f"💾 Saved {saved_count} tweets to database")
    
    def run_comprehensive_collection(self):
        """Execute comprehensive Twitter data collection"""
        print("🚀 COMPREHENSIVE TWITTER DATA EXPANSION")
        print("="*60)
        print(f"🎯 Target: {self.target_tweets:,} tweets from {len(self.influencers)} influencers")
        print(f"📅 Timeframe: {self.target_start_date.strftime('%Y-%m-%d')} to {self.target_end_date.strftime('%Y-%m-%d')}")
        print("="*60)
        
        # Setup browser
        if not self.setup_selenium_driver():
            self.logger.error("❌ Failed to setup browser. Exiting.")
            return False
        
        # Check/prompt for login
        if not self.check_twitter_login():
            if not self.prompt_manual_login():
                self.logger.error("❌ Twitter login failed. Exiting.")
                return False
        
        # Collection loop
        for i, influencer in enumerate(self.influencers):
            if self.tweets_collected >= self.target_tweets:
                self.logger.info(f"🎯 Target reached! Collected {self.tweets_collected:,} tweets")
                break
            
            print(f"\\n📊 Progress: {i+1}/{len(self.influencers)} ({self.tweets_collected:,}/{self.target_tweets:,} tweets)")
            print(f"🎭 Collecting from @{influencer.username} ({influencer.category})")
            
            # Calculate tweets to collect from this user
            remaining_tweets = self.target_tweets - self.tweets_collected
            tweets_for_user = min(500, remaining_tweets)  # Max 500 per user
            
            # Collect tweets
            user_tweets = self.scrape_user_tweets(influencer.username, tweets_for_user)
            
            # Save to database
            if user_tweets:
                self.save_tweets_to_database(user_tweets)
            
            # Progress update
            print(f"✅ @{influencer.username}: {len(user_tweets)} tweets collected")
            
            # Break between users
            time.sleep(5)
        
        # Cleanup
        if self.driver:
            self.driver.quit()
        
        print("\\n" + "="*60)
        print("🎉 TWITTER EXPANSION COMPLETE!")
        print(f"📊 Total tweets collected: {self.tweets_collected:,}")
        print(f"🎯 Target achievement: {(self.tweets_collected/self.target_tweets)*100:.1f}%")
        print("="*60)
        
        return True

def main():
    """Main execution function"""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Initialize expansion system
    expansion = ComprehensiveTwitterExpansion()
    
    # Show influencer overview
    print("🎭 COMPREHENSIVE INFLUENCER DATABASE")
    print("="*50)
    
    categories = {}
    for influencer in expansion.influencers:
        if influencer.category not in categories:
            categories[influencer.category] = []
        categories[influencer.category].append(influencer)
    
    for category, influencers in categories.items():
        total_estimated = sum(inf.estimated_tweets_per_day for inf in influencers)
        print(f"📂 {category}: {len(influencers)} accounts (~{total_estimated} tweets/day)")
    
    print(f"\\n🎯 Total: {len(expansion.influencers)} influencers")
    print("="*50)
    
    # Confirm before starting
    print("\\n⚠️ This will open a browser window and may require Twitter login.")
    print("The collection process may take several hours to complete.")
    
    proceed = input("\\nProceed with comprehensive Twitter expansion? (y/N): ").lower().strip()
    
    if proceed == 'y':
        success = expansion.run_comprehensive_collection()
        if success:
            print("\\n🎉 Data expansion completed successfully!")
            print("📊 Ready to regenerate ML dataset with expanded data.")
        else:
            print("\\n❌ Data expansion failed. Check logs for details.")
    else:
        print("\\n⏹️ Collection cancelled by user.")

if __name__ == "__main__":
    main()