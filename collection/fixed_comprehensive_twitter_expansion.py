#!/usr/bin/env python3
"""
FIXED Comprehensive Twitter Data Expansion System
FIXED Chrome/Selenium issues with proper error handling and fallbacks

OBJECTIVE: Collect 25,000+ high-quality tweets from 70+ crypto influencers (2022-2025)
TARGET: 10x improvement in ML dataset size (178 → 1,720+ samples)

IMPROVEMENTS:
- Fixed Chrome driver configuration issues
- Enhanced error handling and recovery
- Multiple browser fallback options  
- Better rate limiting and human behavior simulation
- Comprehensive logging and progress tracking
"""

import os
import time
import random
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import sqlite3

# Enhanced Selenium imports with error handling
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Selenium not available: {e}")
    SELENIUM_AVAILABLE = False

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

@dataclass
class InfluencerProfile:
    username: str
    category: str
    priority: int  # 1=highest, 3=lowest
    description: str
    estimated_tweets_per_day: int

class FixedComprehensiveTwitterExpansion:
    def __init__(self):
        self.setup_logging()
        self.driver = None
        
        # Comprehensive influencer database - EXPANDED TO 85+ PROFILES
        self.influencers = self.build_comprehensive_influencer_list()
        
        # Collection parameters
        self.target_start_date = datetime(2022, 1, 1)
        self.target_end_date = datetime(2025, 8, 1)
        self.tweets_collected = 0
        self.target_tweets = 25000  # Ambitious target
        
        # Rate limiting and performance
        self.requests_made = 0
        self.last_request_time = time.time()
        self.base_delay = 2  # Base delay between requests
        self.profiles_processed = 0
        
        # Database setup
        self.db_path = "db/cryptopulse.db"
        self.init_database()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/fixed_twitter_expansion.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twitter_posts (
                    id TEXT PRIMARY KEY,
                    username TEXT,
                    content TEXT,
                    likes INTEGER,
                    retweets INTEGER,
                    replies INTEGER,
                    created_at REAL,
                    url TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()
        
    def build_comprehensive_influencer_list(self) -> List[InfluencerProfile]:
        """Build MASSIVE list of 85+ crypto influencers across all categories"""
        
        influencers = []
        
        # 🏛️ ETHEREUM FOUNDERS & CORE DEVELOPERS (Priority 1 - Must Have)
        ethereum_core = [
            ("VitalikButerin", "Ethereum Founder", 1, "Ethereum Creator - Most Important", 3),
            ("TimBeiko", "Ethereum Core Dev", 1, "EIP Editor & Core Dev", 4),
            ("evan_van_ness", "Ethereum Community", 1, "Week in Ethereum News", 7),
            ("drakefjustin", "Ethereum Research", 1, "Ethereum Foundation", 2),
            ("econoar", "Ethereum Core", 1, "Ethereum Core Developer", 3),
            ("dannyryan", "Ethereum Research", 1, "Ethereum 2.0 Researcher", 2),
            ("djrtwo", "Ethereum Research", 1, "Ethereum 2.0 Lead", 2),
            ("JustinDrake_", "Ethereum Research", 1, "Ethereum Researcher", 3),
            ("terencechain", "Ethereum Dev", 1, "Prysm Client Lead", 4),
            ("preston_vanloon", "Ethereum Dev", 1, "Prysm Co-founder", 2),
            ("lefterisjp", "Ethereum Dev", 1, "Rotki Founder", 3),
            ("nicksdjohnson", "Ethereum Dev", 1, "ENS Lead Developer", 4),
        ]
        
        # 🏦 DEFI PROTOCOL LEADERS (Priority 1 - High Volume)
        defi_leaders = [
            ("haydenzadams", "DeFi", 1, "Uniswap Founder", 5),
            ("bantg", "DeFi", 1, "Yearn Core Developer", 4),
            ("AndreCronjeTech", "DeFi", 1, "DeFi Architect", 3),
            ("divine_economy", "DeFi", 1, "DeFi Educator", 6),
            ("TheDeFiEdge", "DeFi", 1, "DeFi Strategy & Analysis", 5),
            ("DefiIgnas", "DeFi", 1, "DeFi Deep Dives", 8),
            ("sassal0x", "DeFi", 1, "DeFi Researcher", 4),
            ("ryandscott", "DeFi", 1, "DeFi Builder", 3),
            ("statelayer", "DeFi", 1, "Maker Protocol", 3),
            ("RuneKek", "DeFi", 1, "THORChain", 5),
            ("0xMaki", "DeFi", 1, "SushiSwap", 4),
            ("kaiynne", "DeFi", 1, "Curve Finance", 3),
        ]
        
        # 💰 CRYPTO VCs & INVESTORS (Priority 1 - Market Moving)
        crypto_vcs = [
            ("arjunblj", "VC", 1, "Variant Fund Partner", 2),
            ("ljin18", "VC", 1, "Variant Fund", 1),
            ("pet3rpan_", "VC", 1, "1kx Network", 3),
            ("austingriffith", "Builder/VC", 1, "BuidlGuidl", 5),
            ("CBVentures", "VC", 1, "Coinbase Ventures", 2),
            ("panteracapital", "VC", 1, "Pantera Capital", 1),
            ("a16zcrypto", "VC", 1, "Andreessen Horowitz", 2),
            ("polychain", "VC", 1, "Polychain Capital", 1),
            ("naval", "VC", 1, "AngelList Founder", 1),
            ("balajis", "VC", 1, "Former Coinbase CTO", 2),
            ("cdixon", "VC", 1, "a16z Partner", 1),
            ("novogratz", "VC", 1, "Galaxy Digital", 2),
        ]
        
        # 📊 TRADING & ANALYSIS EXPERTS (Priority 2 - High Volume)
        trading_experts = [
            ("CryptoCred", "Trading", 2, "Technical Analysis Expert", 10),
            ("pentosh1", "Trading", 2, "Crypto Analyst", 8),
            ("CryptoKaleo", "Trading", 2, "Swing Trader", 6),
            ("CryptoBirb", "Trading", 2, "Chart Analysis", 9),
            ("il_capo_of_crypto", "Trading", 2, "Market Analysis", 5),
            ("TechDev_52", "Trading", 2, "Technical Analysis", 4),
            ("CryptoMichNL", "Trading", 2, "Market Commentary", 6),
            ("AlexKruger", "Trading", 2, "Macro Analysis", 3),
            ("CryptoHamster", "Trading", 2, "DeFi Trading", 5),
            ("TraderKoz", "Trading", 2, "Derivatives Trading", 4),
            ("RektCapital", "Trading", 2, "Technical Analysis", 7),
            ("davthewave", "Trading", 2, "Wave Analysis", 3),
        ]
        
        # 📰 CRYPTO MEDIA & JOURNALISTS (Priority 2 - News & Updates)  
        crypto_media = [
            ("laurashin", "Media", 2, "Unchained Podcast Host", 4),
            ("nlw", "Media", 2, "The Breakdown Host", 5),
            ("ErikVoorhees", "Media", 2, "ShapeShift CEO", 3),
            ("nic__carter", "Media", 2, "Castle Island VC", 4),
            ("APompliano", "Media", 2, "Pomp Podcast", 6),
            ("DocumentingBTC", "Media", 2, "Bitcoin News Aggregator", 12),
            ("WuBlockchain", "Media", 2, "Crypto Journalist", 15),
            ("adamscochran", "Media", 2, "Crypto Analysis", 5),
            ("MessariCrypto", "Media", 2, "Crypto Research Platform", 8),
            ("coindesk", "Media", 2, "Crypto News Outlet", 15),
            ("Cointelegraph", "Media", 2, "Crypto News", 20),
            ("TheBlock__", "Media", 2, "The Block News", 12),
        ]
        
        # 🏢 EXCHANGE & CEX LEADERS (Priority 2 - Market Impact)
        exchange_leaders = [
            ("cz_binance", "CEX", 2, "Binance CEO", 4),
            ("brian_armstrong", "CEX", 2, "Coinbase CEO", 2),
            ("krakenfx", "CEX", 2, "Kraken Exchange", 5),
            ("kucoincom", "CEX", 2, "KuCoin Exchange", 4),
            ("OKEx", "CEX", 2, "OKX Exchange", 3),
            ("gate_io", "CEX", 2, "Gate.io Exchange", 4),
            ("HuobiGlobal", "CEX", 2, "Huobi Exchange", 3),
            ("cryptocom", "CEX", 2, "Crypto.com", 5),
            ("GeminiTrust", "CEX", 2, "Gemini Exchange", 3),
            ("Tyler", "CEX", 2, "Gemini Co-founder", 1),
            ("Cameron", "CEX", 2, "Gemini Co-founder", 1),
        ]
        
        # 🎨 NFT & WEB3 INFLUENCERS (Priority 2 - Cultural Impact)
        nft_web3 = [
            ("punk6529", "NFT", 2, "NFT Collector & Educator", 8),
            ("garyvee", "NFT", 2, "VeeFriends Creator", 5),
            ("dfinzer", "NFT", 2, "OpenSea CEO", 2),
            ("seedphrase", "Web3", 2, "Web3 Builder", 4),
            ("coopahtroopa", "Web3", 2, "Web3 Investor", 5),
            ("kylesamani", "Web3", 2, "Multicoin Capital", 3),
            ("punk4156", "NFT", 2, "NFT Collector", 6),
            ("beaniemaxi", "NFT", 2, "NFT Influencer", 7),
            ("franklinisbored", "NFT", 2, "BAYC Community", 5),
            ("j1mmyeth", "NFT", 2, "NFT Trader", 6),
        ]
        
        # 🔗 LAYER 2 & SCALING SOLUTIONS (Priority 2 - Technical)
        layer2_scaling = [
            ("arbitrum", "L2", 2, "Arbitrum Official", 3),
            ("optimismFND", "L2", 2, "Optimism Foundation", 4),
            ("0xPolygon", "L2", 2, "Polygon Official", 5),
            ("StarkWareLtd", "L2", 2, "StarkNet", 3),
            ("zksync", "L2", 2, "zkSync Official", 4),
            ("loopringorg", "L2", 2, "Loopring Protocol", 3),
            ("MetisDAO", "L2", 2, "Metis Network", 2),
            ("Immutable", "L2", 2, "Immutable X", 3),
            ("epolynya", "L2", 2, "L2 Researcher", 4),
        ]
        
        # 🌟 HIGH-PROFILE MENTIONS (Priority 3 - Occasional Crypto)
        high_profile = [
            ("elonmusk", "General", 3, "Tesla/SpaceX CEO", 10),
            ("michael_saylor", "General", 3, "MicroStrategy CEO", 3),
            ("RaoulGMI", "Macro", 3, "Real Vision CEO", 4),
            ("ODELL", "Bitcoin", 3, "Bitcoin Maximalist", 6),
            ("MartyBent", "Bitcoin", 3, "Bitcoin Podcast", 5),
            ("gladstein", "Bitcoin", 3, "Human Rights Foundation", 3),
            ("TraceMayer", "Bitcoin", 3, "Bitcoin Investor", 2),
            ("NickSzabo4", "Bitcoin", 3, "Smart Contracts Pioneer", 1),
        ]
        
        # Build comprehensive list
        all_categories = [
            ethereum_core, defi_leaders, crypto_vcs, trading_experts,
            crypto_media, exchange_leaders, nft_web3, layer2_scaling, high_profile
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
        
        # Sort by priority (1=highest) and then by tweet volume
        influencers.sort(key=lambda x: (x.priority, -x.estimated_tweets_per_day))
        
        self.logger.info(f"🎭 Built MASSIVE influencer list: {len(influencers)} profiles")
        print(f"🎭 COMPREHENSIVE INFLUENCER DATABASE: {len(influencers)} profiles")
        
        # Print category breakdown
        categories = {}
        for inf in influencers:
            if inf.category not in categories:
                categories[inf.category] = []
            categories[inf.category].append(inf)
        
        for cat, profiles in categories.items():
            total_tweets = sum(p.estimated_tweets_per_day for p in profiles)
            print(f"   📂 {cat}: {len(profiles)} accounts (~{total_tweets} tweets/day)")
        
        return influencers
    
    def setup_selenium_driver(self, headless=True):
        """FIXED Selenium setup using system chromium-driver and xvfb"""
        if not SELENIUM_AVAILABLE:
            self.logger.error("❌ Selenium not available")
            return False
        
        # Set up virtual display if running headless
        if headless and os.path.exists("/usr/bin/xvfb-run"):
            os.environ['DISPLAY'] = ':99'
            self.logger.info("🖥️ Using xvfb virtual display")
        
        chrome_options = Options()
        
        # OPTIMIZED Chrome arguments for system chromium-driver
        essential_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage", 
            "--disable-gpu",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-images",  # Faster loading
            "--disable-web-security",
            "--allow-running-insecure-content",
            "--ignore-certificate-errors",
            "--ignore-ssl-errors",
            "--ignore-certificate-errors-spki-list",
            "--window-size=1920,1080",
            "--remote-debugging-port=9222",  # Help with DevToolsActivePort issues
        ]
        
        # Add headless mode if requested
        if headless:
            essential_args.extend([
                "--headless",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--virtual-time-budget=10000"  # Help with headless timing
            ])
        
        for arg in essential_args:
            chrome_options.add_argument(arg)
        
        # Enhanced stealth options
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User agent for Linux system
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        
        # Use system chromium-driver paths
        chromium_drivers = [
            "/usr/bin/chromium-driver",
            "/usr/bin/chromedriver", 
            "/usr/lib/chromium-browser/chromedriver"
        ]
        
        # Try chromium browser binaries
        chromium_binaries = [
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium", 
            "/usr/bin/google-chrome"
        ]
        
        for chromium_binary in chromium_binaries:
            if os.path.exists(chromium_binary):
                chrome_options.binary_location = chromium_binary
                self.logger.info(f"📍 Using Chromium binary: {chromium_binary}")
                break
        
        # Try to create driver with system chromium-driver
        for attempt in range(2):
            try:
                self.logger.info(f"🔄 Chromium driver setup attempt {attempt + 1}/2")
                
                # Try system chromium-driver paths
                driver_service = None
                for driver_path in chromium_drivers:
                    if os.path.exists(driver_path):
                        self.logger.info(f"📍 Using ChromeDriver: {driver_path}")
                        driver_service = Service(driver_path)
                        break
                
                if not driver_service:
                    # Fallback to webdriver-manager
                    self.logger.info("📦 Falling back to WebDriver Manager...")
                    driver_service = Service(ChromeDriverManager().install())
                
                # Create driver
                self.driver = webdriver.Chrome(service=driver_service, options=chrome_options)
                
                # Test basic functionality
                self.driver.get("https://www.google.com")
                self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
                
                self.logger.info(f"✅ Chromium driver initialized successfully")
                print(f"✅ Chromium driver setup successful")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Driver setup attempt {attempt + 1} failed: {str(e)}")
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                
                # Try without headless on second attempt
                if attempt == 0 and headless:
                    self.logger.info("🔄 Retrying without headless mode")
                    headless = False
                    chrome_options.arguments = [arg for arg in chrome_options.arguments 
                                               if not arg.startswith("--headless") and not arg.startswith("--virtual-time")]
                
                time.sleep(2)
        
        self.logger.error("❌ All Chromium driver setup attempts failed")
        return False
    
    def human_delay(self, base_seconds=3, variance=2):
        """Enhanced human-like delays"""
        delay = base_seconds + random.uniform(-variance, variance)
        delay = max(0.5, delay)  # Minimum delay
        time.sleep(delay)
    
    def check_twitter_access(self):
        """Test Twitter access and login status"""
        try:
            print("🔍 Checking Twitter login status...")
            self.driver.get("https://twitter.com/home")  
            
            # Give more time for page load and potential redirects
            print("⏳ Waiting for Twitter page to load...")
            time.sleep(5)
            
            current_url = self.driver.current_url.lower()
            page_title = self.driver.title.lower()
            
            print(f"📍 Current URL: {current_url}")
            print(f"📄 Page title: {page_title}")
            
            # Check for login indicators
            if "home" in current_url or "timeline" in current_url:
                self.logger.info("✅ Already logged into Twitter")
                print("✅ Login detected - you're already logged in!")
                return True
            elif "login" in current_url or "signin" in current_url:
                self.logger.warning("⚠️ Twitter login page detected")
                print("🔐 Twitter login page detected - login required")
                return False
            else:
                # Ambiguous state - let user decide
                print(f"❓ Unclear login state. Current page: {current_url}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error checking Twitter access: {str(e)}")
            print(f"❌ Error checking Twitter: {str(e)}")
            return False
    
    def prompt_manual_login(self, extended_timeout=True):
        """Handle manual Twitter login with proper wait time"""
        print("\\n" + "="*80)
        print("🔐 TWITTER LOGIN REQUIRED")
        print("="*80)
        print("BROWSER WINDOW WILL OPEN - Please log in to Twitter there")
        print()
        print("STEPS:")
        print("1. Browser will open to Twitter login page")
        print("2. Enter your Twitter username/email and password")
        print("3. Complete any 2FA/verification if prompted")
        print("4. Wait until you see your Twitter home timeline")
        print("5. Come back here and press ENTER")
        print()
        print("⏰ You have PLENTY OF TIME - no rush!")
        print("="*80)
        
        # Navigate to Twitter login and WAIT
        try:
            print("\\n🌐 Opening Twitter login page...")
            self.driver.get("https://twitter.com/login")
            print("✅ Browser should now show Twitter login page")
            
            # Give user 20-25 seconds to see the page and start login
            print("\\n⏳ Giving you 25 seconds to start the login process...")
            for i in range(25, 0, -1):
                print(f"\\r   ⏰ {i} seconds remaining to start login...", end="", flush=True)
                time.sleep(1)
            print("\\n")
            
        except Exception as e:
            self.logger.error(f"Error navigating to Twitter login: {e}")
            print(f"❌ Error opening login page: {e}")
        
        # Now wait for user to complete login
        print("💡 COMPLETE YOUR LOGIN NOW")
        print("   - The browser should be showing Twitter login")
        print("   - Enter your credentials and complete 2FA if needed")
        print("   - Make sure you reach your Twitter home feed")
        print()
        
        # Simple input wait - no time pressure
        input("🔄 Press ENTER when you have successfully logged into Twitter: ")
        
        # Give a moment for any final page loads
        print("\\n⏳ Allowing page to fully load after login...")
        time.sleep(5)
        
        # Verify login worked
        print("🔍 Verifying your Twitter login...")
        
        # Try verification multiple times with pauses
        for attempt in range(3):
            if self.check_twitter_access():
                print("\\n✅ TWITTER LOGIN SUCCESSFUL!")
                print("🚀 Ready to start comprehensive collection!")
                return True
            else:
                if attempt < 2:
                    print(f"⚠️ Login verification failed, retrying in 5 seconds... ({attempt + 1}/3)")
                    time.sleep(5)
                else:
                    print("\\n❌ LOGIN VERIFICATION FAILED")
                    print("This might happen if:")
                    print("  - You're not fully logged in yet")
                    print("  - Twitter is asking for additional verification")
                    print("  - The page is still loading")
                    print()
                    
                    retry = input("Try login verification again? (y/N): ").lower().strip()
                    if retry == 'y':
                        print("🔄 Retrying verification...")
                        time.sleep(3)
                        return self.prompt_manual_login(extended_timeout=False)
                    else:
                        print("❌ Skipping login verification - proceeding anyway")
                        print("⚠️ Collection may fail if not properly logged in")
                        return False
        
        return False
    
    def scrape_user_tweets(self, username: str, target_count: int = 1000) -> List[Dict]:
        """COMPREHENSIVE tweet scraping from user profile"""
        tweets = []
        self.logger.info(f"🐦 Starting DEEP scrape of @{username} (target: {target_count} tweets)")
        
        try:
            # Navigate to profile
            profile_url = f"https://twitter.com/{username}"
            self.driver.get(profile_url)
            self.human_delay(5, 2)
            
            # Check if profile exists and is accessible
            page_source = self.driver.page_source.lower()
            if any(error in page_source for error in ["doesn't exist", "suspended", "protected"]):
                self.logger.warning(f"⚠️ Profile @{username} not accessible")
                return tweets
            
            # Initialize scroll tracking
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            no_new_content_count = 0
            scroll_count = 0
            max_scrolls = 100  # Allow deep scrolling for historical data
            
            seen_tweet_ids = set()
            
            while len(tweets) < target_count and scroll_count < max_scrolls and no_new_content_count < 10:
                # Find tweet elements with multiple selectors
                tweet_elements = []
                selectors = [
                    '[data-testid="tweet"]',
                    'article[data-testid="tweet"]', 
                    'div[data-testid="tweet"]',
                    '[role="article"]'
                ]
                
                for selector in selectors:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        tweet_elements = elements
                        break
                
                tweets_found_this_scroll = 0
                
                for element in tweet_elements:
                    if len(tweets) >= target_count:
                        break
                    
                    try:
                        # Extract tweet URL/ID
                        tweet_url = None
                        url_selectors = [
                            'a[href*="/status/"]',
                            'time a',
                            'a[role="link"][href*="/status/"]'
                        ]
                        
                        for url_sel in url_selectors:
                            try:
                                url_elem = element.find_element(By.CSS_SELECTOR, url_sel)
                                tweet_url = url_elem.get_attribute("href")
                                break
                            except:
                                continue
                        
                        if not tweet_url or "/status/" not in tweet_url:
                            continue
                        
                        tweet_id = tweet_url.split("/status/")[-1].split("?")[0]
                        
                        if tweet_id in seen_tweet_ids:
                            continue
                        seen_tweet_ids.add(tweet_id)
                        
                        # Extract tweet text
                        tweet_text = ""
                        text_selectors = [
                            '[data-testid="tweetText"]',
                            'div[lang]',
                            'span[lang]'
                        ]
                        
                        for text_sel in text_selectors:
                            try:
                                text_elem = element.find_element(By.CSS_SELECTOR, text_sel)
                                tweet_text = text_elem.text.strip()
                                if tweet_text:
                                    break
                            except:
                                continue
                        
                        if not tweet_text or len(tweet_text) < 5:
                            continue
                        
                        # Extract engagement metrics
                        def extract_count(element, testid):
                            try:
                                count_elem = element.find_element(By.CSS_SELECTOR, f'[data-testid="{testid}"]')
                                count_text = count_elem.text.strip()
                                if 'K' in count_text:
                                    return int(float(count_text.replace('K', '')) * 1000)
                                elif 'M' in count_text:
                                    return int(float(count_text.replace('M', '')) * 1000000)
                                else:
                                    return int(count_text) if count_text.isdigit() else 0
                            except:
                                return 0
                        
                        likes = extract_count(element, "like")
                        retweets = extract_count(element, "retweet") 
                        replies = extract_count(element, "reply")
                        
                        tweet_data = {
                            "id": tweet_id,
                            "username": username,
                            "content": tweet_text,
                            "likes": likes,
                            "retweets": retweets,
                            "replies": replies,
                            "created_at": datetime.now().timestamp(),  # Approximate
                            "url": tweet_url,
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        tweets.append(tweet_data)
                        tweets_found_this_scroll += 1
                        
                    except Exception as e:
                        self.logger.debug(f"Error extracting tweet from @{username}: {str(e)}")
                        continue
                
                # Log progress
                if tweets_found_this_scroll > 0:
                    self.logger.info(f"   📊 Found {tweets_found_this_scroll} new tweets from @{username} (total: {len(tweets)})")
                    no_new_content_count = 0
                else:
                    no_new_content_count += 1
                
                # Scroll down to load more content
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.human_delay(2, 1)
                
                # Check if page height changed
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    no_new_content_count += 1
                else:
                    last_height = new_height
                
                scroll_count += 1
                
                # Rate limiting
                self.respect_rate_limits()
            
            self.logger.info(f"✅ Collected {len(tweets)} tweets from @{username}")
            
        except Exception as e:
            self.logger.error(f"❌ Error scraping @{username}: {str(e)}")
        
        return tweets
    
    def respect_rate_limits(self):
        """Smart rate limiting with variable delays"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Variable delay based on time of day and request count
        if self.requests_made % 50 == 0:  # Longer break every 50 requests
            delay = random.uniform(10, 20)
            self.logger.info(f"💤 Extended break: {delay:.1f}s after {self.requests_made} requests")
        elif time_since_last < self.base_delay:
            delay = self.base_delay - time_since_last + random.uniform(0, 1)
        else:
            delay = random.uniform(0.5, 2)
        
        time.sleep(delay)
        self.last_request_time = time.time()
        self.requests_made += 1
    
    def save_tweets_to_database(self, tweets: List[Dict]) -> int:
        """Save collected tweets to database"""
        if not tweets:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        saved_count = 0
        
        try:
            cursor = conn.cursor()
            
            for tweet in tweets:
                try:
                    # Check for existing tweet
                    cursor.execute("SELECT id FROM twitter_posts WHERE id = ?", (tweet['id'],))
                    if cursor.fetchone():
                        continue  # Skip duplicates
                    
                    # Insert new tweet
                    cursor.execute("""
                        INSERT INTO twitter_posts 
                        (id, username, content, likes, retweets, replies, created_at, url, scraped_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        tweet['id'],
                        tweet['username'],
                        tweet['content'],
                        tweet.get('likes', 0),
                        tweet.get('retweets', 0),
                        tweet.get('replies', 0),
                        tweet.get('created_at', datetime.now().timestamp()),
                        tweet.get('url', ''),
                        tweet.get('scraped_at', datetime.now().isoformat())
                    ))
                    
                    saved_count += 1
                    self.tweets_collected += 1
                    
                except Exception as e:
                    self.logger.error(f"Error saving tweet {tweet.get('id', 'unknown')}: {str(e)}")
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
        finally:
            conn.close()
        
        if saved_count > 0:
            self.logger.info(f"💾 Saved {saved_count} new tweets to database")
        
        return saved_count
    
    def run_comprehensive_collection(self):
        """Execute COMPREHENSIVE Twitter data collection"""
        start_time = time.time()
        
        print("🚀 COMPREHENSIVE TWITTER DATA EXPANSION")
        print("="*80)
        print(f"🎯 Target: {self.target_tweets:,} tweets from {len(self.influencers)} influencers")
        print(f"📅 Timeframe Focus: 2022-2025 (historical + recent)")
        print(f"⏱️ Expected Duration: 4-8 hours for comprehensive collection")
        print("="*80)
        
        # Setup browser in VISIBLE mode for login
        print("🔧 Setting up Chrome browser in VISIBLE mode for login...")
        print("💡 Browser window will open - this is normal for Twitter login")
        if not self.setup_selenium_driver(headless=False):  # Always visible for login
            print("❌ Failed to setup browser. Cannot proceed.")
            return False
        
        # Check/setup Twitter access
        print("🔍 Checking Twitter access...")
        if not self.check_twitter_access():
            print("🔐 Manual Twitter login required...")
            if not self.prompt_manual_login():
                print("❌ Twitter access failed. Cannot proceed.")
                return False
        
        # Begin comprehensive collection
        print("\\n🎬 STARTING COMPREHENSIVE COLLECTION")
        print("="*60)
        
        # Process influencers in priority order
        for i, influencer in enumerate(self.influencers):
            if self.tweets_collected >= self.target_tweets:
                print(f"🎯 TARGET REACHED! Collected {self.tweets_collected:,} tweets")
                break
            
            # Calculate tweets to collect from this user based on priority
            if influencer.priority == 1:
                tweets_per_user = min(800, self.target_tweets // 30)  # Priority 1: Deep collection
            elif influencer.priority == 2:
                tweets_per_user = min(400, self.target_tweets // 50)  # Priority 2: Medium collection  
            else:
                tweets_per_user = min(200, self.target_tweets // 80)  # Priority 3: Light collection
            
            remaining_target = self.target_tweets - self.tweets_collected
            tweets_per_user = min(tweets_per_user, remaining_target)
            
            print(f"\\n📊 PROGRESS: {i+1}/{len(self.influencers)} | Collected: {self.tweets_collected:,}/{self.target_tweets:,}")
            print(f"🎭 PROCESSING: @{influencer.username} ({influencer.category})")
            print(f"   📈 Priority: {influencer.priority} | Target: {tweets_per_user} tweets")
            print(f"   📝 {influencer.description}")
            
            # Collect tweets from this user
            user_tweets = self.scrape_user_tweets(influencer.username, tweets_per_user)
            
            # Save tweets
            if user_tweets:
                saved_count = self.save_tweets_to_database(user_tweets)
                print(f"   ✅ SUCCESS: {saved_count} new tweets from @{influencer.username}")
            else:
                print(f"   ⚠️ NO TWEETS: @{influencer.username} (may be protected or suspended)")
            
            self.profiles_processed += 1
            
            # Progress milestone updates
            if self.profiles_processed % 5 == 0:
                elapsed_time = time.time() - start_time
                rate = self.tweets_collected / (elapsed_time / 3600) if elapsed_time > 0 else 0
                print(f"\\n📈 MILESTONE: {self.profiles_processed} profiles processed")
                print(f"   ⏱️ Time elapsed: {elapsed_time/3600:.1f} hours")
                print(f"   📊 Collection rate: {rate:.0f} tweets/hour")
                print(f"   🎯 Progress: {(self.tweets_collected/self.target_tweets)*100:.1f}%")
            
            # Smart breaks between profiles
            if influencer.priority == 1:
                break_time = random.uniform(8, 15)  # Longer breaks for high-priority
            else:
                break_time = random.uniform(3, 8)   # Shorter breaks for lower priority
            
            print(f"   💤 Break: {break_time:.1f}s before next profile...")
            time.sleep(break_time)
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\\n" + "="*80)
        print("🎉 COMPREHENSIVE COLLECTION COMPLETE!")
        print("="*80)
        print(f"⏱️ Total Duration: {total_time/3600:.2f} hours")
        print(f"👥 Profiles Processed: {self.profiles_processed}")
        print(f"📊 Total Tweets Collected: {self.tweets_collected:,}")
        print(f"🎯 Target Achievement: {(self.tweets_collected/self.target_tweets)*100:.1f}%")
        print(f"📈 Average Rate: {self.tweets_collected/(total_time/3600):.0f} tweets/hour")
        print("="*80)
        
        # Cleanup
        if self.driver:
            self.driver.quit()
        
        return True

def main():
    """Main execution function"""
    print("🚀 CRYPTOPULSE: FIXED COMPREHENSIVE TWITTER EXPANSION")
    print("="*70)
    print("🎯 MISSION: Collect 25,000+ tweets from 85+ crypto influencers")
    print("📅 TIMEFRAME: 2022-2025 historical + recent data")  
    print("⏱️ DURATION: 4-8 hours for complete collection")
    print("🛠️ IMPROVEMENTS: Fixed Chrome/Selenium issues")
    print("="*70)
    
    # Initialize expansion system
    expansion = FixedComprehensiveTwitterExpansion()
    
    # Show influencer overview
    categories = {}
    for influencer in expansion.influencers[:20]:  # Show first 20
        print(f"   {influencer.priority}. @{influencer.username:<20} {influencer.category:<12} {influencer.description}")
    print(f"   ... and {len(expansion.influencers)-20} more accounts")
    
    print("\\n⚠️ IMPORTANT:")
    print("   - This will open a browser window")
    print("   - Manual Twitter login may be required")
    print("   - Collection will run for several hours")
    print("   - Progress will be logged continuously")
    
    # Auto-proceed for comprehensive collection 
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        proceed = 'y'
        print("\\n🤖 AUTO-PROCEEDING with comprehensive collection...")
    else:
        try:
            proceed = input("\\nProceed with COMPREHENSIVE Twitter expansion? (y/N): ").lower().strip()
        except (EOFError, KeyboardInterrupt):
            proceed = 'y'  # Auto-proceed if no input available
            print("\\n🤖 No input detected - AUTO-PROCEEDING with collection...")
    
    if proceed == 'y':
        print("\\n🎬 STARTING COMPREHENSIVE COLLECTION...")
        success = expansion.run_comprehensive_collection()
        
        if success:
            print("\\n🎉 COMPREHENSIVE EXPANSION COMPLETED!")
            print("🚀 Your CryptoPulse system now has massive Twitter dataset!")
            print("📊 Ready to regenerate ML dataset with expanded data!")
        else:
            print("\\n❌ Collection failed. Check logs for details.")
            print("💡 Try running the debug script first: python archive/debug/debug_twitter.py")
    else:
        print("\\n⏹️ Collection cancelled by user.")
        print("💡 You can test Twitter access with: python src/twitter_scraper.py test")

if __name__ == "__main__":
    main()