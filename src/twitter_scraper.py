# src/twitter_scraper.py

"""
Twitter scraper using Selenium + Chrome profile, with duplicate skipping
and fields aligned to your schema.
"""
import os
import time
import logging
import random
import numpy as np
from datetime import datetime, timedelta

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

logging.basicConfig(
    filename="logs/twitter_scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TwitterScraper:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.queries = [
            q.strip()
            for q in os.getenv("TWITTER_QUERIES", "ethereum,ETH,$ETH").split(",")
            if q.strip()
        ]
        self.driver = None
        
        # EXHAUSTIVE crypto influencer profiles for maximum data collection
        self.crypto_profiles = [
            # Ethereum Core/Founders
            "VitalikButerin", "TimBeiko", "evan_van_ness", "drakefjustin", "econoar",
            
            # DeFi Leaders  
            "haydenzadams", "bantg", "divine_economy", "TheDeFiEdge", "DefiIgnas",
            "DefiKillTheBank", "DeFiTrinity", "sassal0x", "ryandscott", "DefiTech",
            
            # Crypto Analysts/Traders
            "ultrasoundmoney", "EthereumJoseph", "ethereumJoseph", "intocryptoverse",
            "CryptoCred", "TechDev_52", "CryptoKaleo", "CryptoNTez", "AltcoinSherpa",
            "PentoshiEth", "CryptoCapo_", "davthewave", "CryptoMichNL", "RektCapital",
            
            # News/Media
            "Trustnodes", "WatcherGuru", "CoinDesk", "Cointelegraph", "TheBlock__",
            "DecryptMedia", "CoinTelegraph", "CryptoSlate", "bitcoinmagazine", "CoinMarketCap",
            
            # VCs/Institutional  
            "naval", "balajis", "cdixon", "austinrief", "RyanSAdams", "AnthonyPompliano",
            "novogratz", "RaoulGMI", "mskvsk", "cburniske", "ljxie", "aliatiia_",
            
            # Ethereum L2s/Scaling
            "arbitrum", "optimismFND", "0xPolygon", "StarkWareLtd", "zksync", "loopringorg",
            "MetisDAO", "Immutable", "0xMaki", "kaiynne", "epolynya", "Galois_Capital",
            
            # NFT/Gaming
            "punk6529", "beaniemaxi", "pablostanley", "j1mmyeth", "seedphrase", "garyvee",
            "NFTGuyy", "franklinisbored", "cryptopathic", "0xminion", "dhof", "punk4156",
            
            # Memecoins/Culture
            "elonmusk", "justinsuntron", "CZ_Binance", "SBF_FTX", "krakenfx", "brian_armstrong",
            "Tyler", "Cameron", "APompliano", "woonomic", "WClementeIII", "DocumentingBTC",
            
            # Technical/Development
            "lefterisjp", "josephdelong", "danrobinson", "mrjasonchoi", "hosseeb", "spencernoon",
            "mattgcondon", "nicksdjohnson", "lemiscate", "pet3rpan_", "avsa", "alexmasmej",
            
            # International Crypto
            "woonomic", "WClementeIII", "100trillionUSD", "BitcoinMagazine", "DocumentingBTC",
            "APompliano", "gladstein", "danheld", "ODELL", "MartyBent", "matt_odell",
            
            # Additional High-Value Accounts
            "SteveLeebitcoin", "jimmysong", "lopp", "aantonop", "VinnyLingham", "TraceMayer",
            "TuurDemeester", "NickSzabo4", "francispouliot_", "saifedean", "nic__carter",
            "prestonjbyrne", "ErikVoorhees", "rogerkver", "officialmcafee", "Excellion"
        ]

    def setup_driver(self):
        options = Options()
        
        # Use dedicated profile for CryptoPulse Twitter automation
        profile_dir = os.path.join(os.getcwd(), "twitter_profile")
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument("--profile-directory=CryptoPulse")
        
        # ENHANCED Anti-detection measures with randomization
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ]
        selected_ua = random.choice(user_agents)
        options.add_argument(f"--user-agent={selected_ua}")
        logging.info(f"Using User-Agent: {selected_ua[:50]}...")
        
        # Random screen sizes to mimic different devices
        screen_sizes = [
            (1920, 1080),  # Full HD
            (1366, 768),   # Common laptop
            (1440, 900),   # MacBook Air
            (1536, 864),   # Scaled 1920x1080
            (1280, 720),   # HD
            (1600, 900),   # 16:9 widescreen
        ]
        selected_size = random.choice(screen_sizes)
        options.add_argument(f"--window-size={selected_size[0]},{selected_size[1]}")
        logging.info(f"Using screen size: {selected_size[0]}x{selected_size[1]}")
        
        # Advanced stealth options
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")  # Faster loading, reduces bandwidth
        
        # Randomize browser features to look more human
        if random.choice([True, False]):
            options.add_argument("--disable-web-security")
        if random.choice([True, False]):
            options.add_argument("--disable-features=TranslateUI")
        if random.choice([True, False]):
            options.add_argument("--disable-ipc-flooding-protection")
        
        # Performance and stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Random viewport variations
        viewport_variations = [
            "--force-device-scale-factor=1.0",
            "--force-device-scale-factor=1.25",
            "--force-device-scale-factor=1.5"
        ]
        options.add_argument(random.choice(viewport_variations))
        
        # Random language preferences
        languages = ["en-US,en", "en-GB,en", "en-CA,en"]
        options.add_argument(f"--lang={random.choice(languages)}")

        service = Service("/usr/bin/chromedriver") if os.path.exists("/usr/bin/chromedriver") else None
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # Dynamic implicit wait (3-7 seconds)
        wait_time = random.randint(3, 7)
        self.driver.implicitly_wait(wait_time)
        
        # Set realistic window size and position with small variations
        base_width, base_height = selected_size
        actual_width = base_width + random.randint(-50, 50)
        actual_height = base_height + random.randint(-30, 30)
        self.driver.set_window_size(actual_width, actual_height)
        
        # Random window position
        self.driver.set_window_position(random.randint(0, 100), random.randint(0, 100))
        
        # ADVANCED: Remove automation indicators and add human-like properties
        self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'permissions', {get: () => ({query: () => Promise.resolve({state: 'granted'})})});
        """)
        
        logging.info("Chrome driver initialized with ENHANCED anti-detection measures")

    def human_delay(self, base_seconds=2, variance=1):
        """EXTREME human-like delays that feel completely natural"""
        # Simulate different human states and behaviors
        human_states = [
            'focused',      # Quick, efficient scrolling
            'distracted',   # Long pauses, checking phone/other tabs
            'tired',        # Slower reactions, longer pauses
            'excited',      # Quick movements but occasional long reads
            'multitasking', # Irregular patterns, sudden stops
            'bored',        # Very long pauses, quick scrolls to find interesting content
        ]
        
        current_state = random.choice(human_states)
        
        if current_state == 'focused':
            delay = base_seconds * random.uniform(0.4, 0.8)
        elif current_state == 'distracted':
            if random.random() < 0.3:  # Sometimes very long pause (checking phone)
                delay = random.uniform(8, 20)
            else:
                delay = base_seconds * random.uniform(1.2, 2.5)
        elif current_state == 'tired':
            delay = base_seconds * random.uniform(1.5, 3.0)
            # Sometimes micro-sleeps (longer pauses)
            if random.random() < 0.2:
                delay += random.uniform(3, 8)
        elif current_state == 'excited':
            if random.random() < 0.7:  # Quick scrolling
                delay = base_seconds * random.uniform(0.3, 0.6)
            else:  # But stops to read interesting content
                delay = random.uniform(5, 12)
        elif current_state == 'multitasking':
            # Very erratic timing
            delay_options = [
                base_seconds * 0.2,  # Quick glance
                base_seconds * 4.0,  # Doing something else
                base_seconds * 0.8,  # Normal
                random.uniform(15, 30)  # Long distraction
            ]
            delay = random.choice(delay_options)
        else:  # bored
            if random.random() < 0.6:  # Quick scroll to find something interesting
                delay = base_seconds * random.uniform(0.2, 0.5)
            else:  # Long pause when something catches attention
                delay = random.uniform(6, 15)
        
        # Add natural micro-variations (humans aren't perfectly consistent)
        jitter = random.uniform(-0.1, 0.1) * delay
        delay += jitter
        delay = max(0.1, min(delay, 45))  # Reasonable bounds
        
        # Sometimes add typing/mouse movement sounds (micro-pauses)
        if random.random() < 0.4:
            time.sleep(random.uniform(0.05, 0.15))
        
        time.sleep(delay)
        
        # Very occasionally, simulate getting distracted mid-action
        if random.random() < 0.05:  # 5% chance
            time.sleep(random.uniform(2, 8))  # Brief distraction
    
    def human_scroll(self, scrolls=3):
        """ULTRA-REALISTIC human-like scrolling with natural behaviors"""
        for i in range(scrolls):
            # Simulate realistic human scroll patterns
            behavior = random.choices(
                ['normal_down', 'quick_down', 'scroll_up', 'big_scroll', 'hesitate', 'misclick_recovery'],
                weights=[40, 20, 15, 10, 10, 5]
            )[0]
            
            if behavior == 'normal_down':
                # Normal downward scroll
                scroll_amount = random.randint(250, 450)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
            elif behavior == 'quick_down':
                # Quick scroll (user is scanning)
                scroll_amount = random.randint(600, 900)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
            elif behavior == 'scroll_up':
                # Going back up to re-read something
                scroll_amount = random.randint(-300, -100)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(1, 3))  # Reading time
                # Then continue down
                scroll_amount = random.randint(200, 400)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
            elif behavior == 'big_scroll':
                # Impatient user - big scroll down
                scroll_amount = random.randint(1000, 1500)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
            elif behavior == 'hesitate':
                # User hesitates, small scroll then pause
                scroll_amount = random.randint(100, 200)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(2, 5))  # Thinking/reading
                
            elif behavior == 'misclick_recovery':
                # Simulate accidental scroll up then correction
                scroll_amount = random.randint(-150, -50)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.2, 0.5))  # Quick reaction
                scroll_amount = random.randint(300, 500)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            
            # Add realistic mouse movements during scrolling
            if random.random() < 0.3:  # 30% chance of mouse movement
                self.simulate_cursor_movement()
            
            # Variable delays between scrolls with natural patterns
            if i < scrolls - 1:  # Don't delay after last scroll
                delay_type = random.choices(['quick', 'normal', 'slow'], weights=[30, 50, 20])[0]
                if delay_type == 'quick':
                    time.sleep(random.uniform(0.3, 0.8))
                elif delay_type == 'normal':
                    time.sleep(random.uniform(0.8, 2.0))
                else:  # slow
                    time.sleep(random.uniform(2.0, 4.0))
    
    def simulate_cursor_movement(self):
        """Simulate realistic cursor movements during browsing"""
        try:
            actions = ActionChains(self.driver)
            
            # Get window dimensions
            window_size = self.driver.get_window_size()
            max_x = window_size['width'] - 50
            max_y = window_size['height'] - 50
            
            # Different cursor movement patterns
            movement_type = random.choice(['reading_hover', 'click_intent', 'wandering'])
            
            if movement_type == 'reading_hover':
                # Small movements like hovering over text while reading
                for _ in range(random.randint(2, 4)):
                    x_offset = random.randint(-20, 20)
                    y_offset = random.randint(-10, 30)
                    actions.move_by_offset(x_offset, y_offset)
                    actions.perform()
                    time.sleep(random.uniform(0.1, 0.3))
                    actions.reset_actions()
                    
            elif movement_type == 'click_intent':
                # Movement toward a potential click target then away
                target_x = random.randint(100, max_x)
                target_y = random.randint(200, max_y)
                actions.move_by_offset(target_x - 400, target_y - 300)
                actions.perform()
                time.sleep(random.uniform(0.5, 1.5))
                # Move away without clicking (changed mind)
                actions.move_by_offset(random.randint(-50, 50), random.randint(-30, 30))
                actions.perform()
                actions.reset_actions()
                
            else:  # wandering
                # Random cursor wandering
                x = random.randint(100, max_x)
                y = random.randint(100, max_y)
                actions.move_by_offset(x - 500, y - 400)
                actions.perform()
                actions.reset_actions()
                
        except Exception as e:
            logging.debug(f"Cursor movement simulation failed: {e}")
    
    def simulate_accidental_interactions(self):
        """Occasionally simulate accidental clicks or hovers"""
        if random.random() < 0.1:  # 10% chance of accidental interaction
            try:
                # Find a random element to accidentally interact with
                elements = self.driver.find_elements(By.CSS_SELECTOR, 'div, span, a')
                if elements:
                    element = random.choice(elements[:20])  # First 20 visible elements
                    
                    actions = ActionChains(self.driver)
                    
                    if random.random() < 0.7:  # Accidental hover
                        actions.move_to_element(element)
                        actions.perform()
                        time.sleep(random.uniform(0.1, 0.3))
                        # Move away quickly
                        actions.move_by_offset(random.randint(-50, 50), random.randint(-30, 30))
                        actions.perform()
                    else:  # Very brief accidental click (but cancel quickly)
                        actions.move_to_element(element)
                        actions.click_and_hold()
                        actions.perform()
                        time.sleep(random.uniform(0.05, 0.1))  # Very brief
                        actions.release()
                        actions.perform()
                        
                    actions.reset_actions()
            except Exception as e:
                logging.debug(f"Accidental interaction simulation failed: {e}")
    
    def random_mouse_movement(self):
        """Add random mouse movements to simulate human behavior"""
        try:
            actions = ActionChains(self.driver)
            
            # Get window size
            window_size = self.driver.get_window_size()
            max_x = window_size['width'] - 100
            max_y = window_size['height'] - 100
            
            # Random mouse movements (like reading or looking around)
            for _ in range(random.randint(1, 3)):
                x = random.randint(100, max_x)
                y = random.randint(100, max_y)
                actions.move_by_offset(x - 500, y - 400)  # Move relative to current position
                actions.perform()
                time.sleep(random.uniform(0.1, 0.5))
                actions.reset_actions()  # Reset for next movement
                
        except Exception as e:
            logging.debug(f"Mouse movement failed: {e}")
    
    def simulate_reading_behavior(self):
        """EXTREME reading behavior simulation - very human-like"""
        reading_behaviors = [
            'quick_scan',     # Just scanning through
            'deep_read',      # Found something interesting
            'selective_read', # Reading specific tweets
            'getting_bored',  # Starting to lose interest
            're_reading',     # Going back to read something again
        ]
        
        behavior = random.choice(reading_behaviors)
        
        if behavior == 'quick_scan':
            # Quick scanning, minimal reading
            if random.random() < 0.3:
                time.sleep(random.uniform(0.5, 2.0))
                
        elif behavior == 'deep_read':
            # Found something very interesting
            time.sleep(random.uniform(8, 20))  # Long reading time
            # Sometimes scroll back up to re-read
            if random.random() < 0.4:
                self.driver.execute_script("window.scrollBy(0, -150);")
                time.sleep(random.uniform(2, 5))
                self.driver.execute_script("window.scrollBy(0, 100);")
            # Move mouse like highlighting text or pointing
            if random.random() < 0.8:
                self.simulate_text_selection_behavior()
                
        elif behavior == 'selective_read':
            # Reading some tweets, skipping others
            for _ in range(random.randint(2, 4)):
                if random.random() < 0.6:  # Read this one
                    time.sleep(random.uniform(3, 8))
                else:  # Skip this one
                    time.sleep(random.uniform(0.2, 1.0))
                # Small scroll between tweets
                self.driver.execute_script(f"window.scrollBy(0, {random.randint(100, 200)});")
                
        elif behavior == 'getting_bored':
            # Progressively faster scrolling as interest wanes
            for i in range(random.randint(3, 6)):
                read_time = max(0.1, 4.0 - i * 0.8)  # Getting faster
                time.sleep(random.uniform(0.1, read_time))
                scroll_amount = min(300, 100 + i * 50)  # Bigger scrolls
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
        else:  # re_reading
            # Go back to something interesting
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(-400, -200)});")
            time.sleep(random.uniform(3, 8))  # Re-read
            self.driver.execute_script(f"window.scrollBy(0, {random.randint(300, 500)});")
        
        # Sometimes move mouse during any reading
        if random.random() < 0.7:
            self.simulate_cursor_movement()
    
    def simulate_text_selection_behavior(self):
        """Simulate highlighting/selecting text like a human reading"""
        try:
            # Find tweet text elements
            text_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[lang], span[lang]')
            if text_elements:
                element = random.choice(text_elements[:10])
                actions = ActionChains(self.driver)
                
                # Move to start of text
                actions.move_to_element(element)
                actions.perform()
                time.sleep(random.uniform(0.1, 0.3))
                
                # Simulate text selection (click and drag)
                if random.random() < 0.3:  # 30% chance of actually selecting
                    actions.click_and_hold(element)
                    actions.move_by_offset(random.randint(50, 150), random.randint(-5, 5))
                    time.sleep(random.uniform(0.2, 0.8))
                    actions.release()
                    actions.perform()
                    time.sleep(random.uniform(0.1, 0.5))
                    # Click elsewhere to deselect
                    actions.move_by_offset(random.randint(-100, 100), random.randint(50, 100))
                    actions.click()
                    actions.perform()
                
                actions.reset_actions()
        except Exception as e:
            logging.debug(f"Text selection simulation failed: {e}")
    
    def simulate_tab_switching_behavior(self):
        """Simulate occasionally switching tabs like a distracted human"""
        if random.random() < 0.1:  # 10% chance
            # Simulate Ctrl+Tab (switch tab) - but don't actually do it
            # Just add the delay that would happen
            time.sleep(random.uniform(2, 8))  # Time "away" from Twitter
            logging.debug("Simulated tab switching distraction")
    
    def check_for_blocks(self):
        """Check if we're blocked or need to solve captcha"""
        try:
            # More specific blocking indicators - avoid false positives
            page_text = self.driver.page_source.lower()
            
            # Check for actual blocking messages (more specific)
            specific_blocking_phrases = [
                "you are blocked",
                "account suspended",
                "rate limit exceeded", 
                "something went wrong. try reloading",
                "unusual traffic from your computer",
                "verify you are human"
            ]
            
            for phrase in specific_blocking_phrases:
                if phrase in page_text:
                    logging.warning(f"Potential block detected: {phrase}")
                    return True
                    
            # Check page title for blocking indicators
            page_title = self.driver.title.lower()
            if "blocked" in page_title or "suspended" in page_title:
                logging.warning(f"Block detected in page title: {page_title}")
                return True
                    
            # Check for captcha elements
            captcha_elements = [
                "captcha",
                "challenge",
                "verification"
            ]
            
            for element in captcha_elements:
                if self.driver.find_elements(By.CSS_SELECTOR, f"[class*='{element}']"):
                    logging.warning(f"Captcha/challenge detected: {element}")
                    return True
                    
            return False
            
        except Exception as e:
            logging.debug(f"Error checking for blocks: {e}")
            return False

    def check_login_status(self):
        """Check if already logged into Twitter"""
        try:
            self.driver.get("https://twitter.com/home")
            time.sleep(3)
            
            # If we can access home page, we're likely logged in
            if "home" in self.driver.current_url.lower():
                logging.info("Already logged into Twitter")
                return True
            else:
                logging.info("Not logged into Twitter")
                return False
        except Exception as e:
            logging.warning(f"Error checking login status: {e}")
            return False

    def manual_login_flow(self):
        """Handle login - either automatic (if saved) or manual"""
        print("🔍 Checking Twitter login status...")
        print("   (Giving you 10 seconds to manually log in if needed)")
        time.sleep(10)  # Give user time to log in manually if needed
        
        if self.check_login_status():
            print("✅ Already logged into Twitter! Continuing with data collection...")
            return
        
        print("🔐 Twitter login required...")
        self.driver.get("https://twitter.com/login")
        print("Please log into Twitter in the browser window, then press Enter here.")
        print("Your login will be saved for future automated runs.")
        input()
        logging.info("Continuing after manual login")

    def scrape_search_results(self, query, max_tweets=5000):
        logging.info(f"Scraping query: {query}")
        self.driver.get(f"https://twitter.com/search?q={query}&src=typed_query&f=live")
        time.sleep(5)

        results = []
        seen = set()
        scrolls = 0
        last_h = self.driver.execute_script("return document.body.scrollHeight")

        while len(results) < max_tweets and scrolls < 8:
            cards = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            for c in cards:
                try:
                    url = c.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]').get_attribute("href")
                    tid = url.rstrip("/").split("/")[-1]
                    if tid in seen or self.db.record_exists("twitter_posts", tid):
                        continue
                    seen.add(tid)

                    text = c.find_element(By.CSS_SELECTOR, 'div[lang]').text
                    user_elem = c.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                    username = user_elem.text.split("@")[-1].split()[0]

                    results.append({
                        "id": tid,
                        "username": username,
                        "content": text,
                        "likes": 0,
                        "retweets": 0,
                        "replies": 0,
                        "created_at": datetime.utcnow().timestamp(),
                        "url": url
                    })
                    if len(results) >= max_tweets:
                        break
                except Exception:
                    continue

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_h = self.driver.execute_script("return document.body.scrollHeight")
            scrolls += (new_h == last_h)
            last_h = new_h

        logging.info(f"Collected {len(results)} tweets for '{query}'")
        return results

    def scrape_profile_tweets(self, username, max_tweets=500):  # Increased for historical depth
        """Scrape tweets from a specific user profile with DEEP historical collection"""
        logging.info(f"Scraping profile: @{username} (targeting {max_tweets} tweets)")
        self.driver.get(f"https://twitter.com/{username}")
        self.human_delay(3, 1)  # Human-like delay after page load
        
        # Check if we're blocked before starting
        if self.check_for_blocks():
            logging.error(f"Blocked while accessing @{username}")
            return []

        results = []
        seen = set()
        scrolls = 0
        no_new_content_count = 0
        last_h = self.driver.execute_script("return document.body.scrollHeight")

        # Wait for page to fully load
        time.sleep(5)
        logging.info(f"Starting deep scroll collection for @{username}")
        
        # DEEP COLLECTION: More scrolls, more patience for historical data
        while len(results) < max_tweets and scrolls < 25 and no_new_content_count < 8:
            # Try multiple selectors for tweet cards
            cards = []
            selectors_to_try = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]', 
                'div[data-testid="tweet"]',
                '[role="article"]',
                'article'
            ]
            
            for selector in selectors_to_try:
                cards = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if cards:
                    logging.info(f"Found {len(cards)} tweet cards using selector: {selector}")
                    break
            
            if not cards:
                logging.warning(f"No tweet cards found for @{username}, forcing scroll to load content")
                # Force multiple scrolls to load more content
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                self.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(3)
                scrolls += 1
                continue
            
            tweets_found_this_round = 0
            for c in cards:
                try:
                    # Try multiple ways to find tweet links
                    url_elem = None
                    url_selectors = [
                        'a[href*="/status/"]',
                        'a[href*="/i/web/status/"]',
                        'time[datetime] a',
                        'a[href*="twitter.com"]'
                    ]
                    
                    for url_sel in url_selectors:
                        try:
                            url_elem = c.find_element(By.CSS_SELECTOR, url_sel)
                            break
                        except:
                            continue
                    
                    if not url_elem:
                        continue
                        
                    url = url_elem.get_attribute("href")
                    if "/status/" not in url:
                        continue
                        
                    tid = url.rstrip("/").split("/")[-1]
                    
                    if tid in seen or self.db.record_exists("twitter_posts", tid):
                        continue
                    seen.add(tid)

                    # Try multiple ways to find tweet text
                    text = ""
                    text_selectors = [
                        'div[lang]',
                        '[data-testid="tweetText"]',
                        'div[data-testid="tweetText"]',
                        'span[lang]'
                    ]
                    
                    for text_sel in text_selectors:
                        try:
                            text_elem = c.find_element(By.CSS_SELECTOR, text_sel)
                            text = text_elem.text
                            break
                        except:
                            continue
                    
                    if not text or len(text.strip()) < 3:
                        continue
                    
                    # Include ALL tweets from crypto profiles (don't filter by keywords)
                    # These are crypto influencers, so all their tweets are potentially relevant

                    # Try to get engagement metrics
                    likes = 0
                    retweets = 0
                    replies = 0
                    
                    try:
                        like_elem = c.find_element(By.CSS_SELECTOR, '[data-testid="like"] span')
                        likes = self.parse_engagement_count(like_elem.text)
                    except:
                        pass
                    
                    try:
                        rt_elem = c.find_element(By.CSS_SELECTOR, '[data-testid="retweet"] span')
                        retweets = self.parse_engagement_count(rt_elem.text)
                    except:
                        pass
                    
                    try:
                        reply_elem = c.find_element(By.CSS_SELECTOR, '[data-testid="reply"] span')
                        replies = self.parse_engagement_count(reply_elem.text)
                    except:
                        pass

                    results.append({
                        "id": tid,
                        "username": username,
                        "content": text,
                        "likes": likes,
                        "retweets": retweets,
                        "replies": replies,
                        "created_at": datetime.utcnow().timestamp(),
                        "url": url
                    })
                    
                    tweets_found_this_round += 1
                    
                    if len(results) >= max_tweets:
                        break
                        
                except Exception as e:
                    logging.debug(f"Error parsing tweet in profile {username}: {e}")
                    continue

            logging.info(f"Found {tweets_found_this_round} new tweets for @{username}, total: {len(results)}")

            # Check for blocking before continuing
            if self.check_for_blocks():
                logging.warning(f"Blocking detected while scraping @{username}, stopping")
                break

            # FORCE SCROLL DOWN to load historical tweets
            logging.info(f"Scrolling down to load more tweets for @{username}")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # Additional scroll to trigger infinite scroll
            self.driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(2)
            
            new_h = self.driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h:
                no_new_content_count += 1
                logging.debug(f"No new content loaded for @{username}, count: {no_new_content_count}")
            else:
                no_new_content_count = 0
                logging.info(f"Page height changed for @{username}, continuing scroll")
            
            scrolls += 1
            last_h = new_h

        logging.info(f"Collected {len(results)} crypto tweets from @{username}")
        return results

    def parse_engagement_count(self, count_text):
        """Parse engagement counts like '1.2K', '350', '2.1M' to integers"""
        if not count_text or count_text == '':
            return 0
        
        count_text = count_text.strip().upper()
        if 'K' in count_text:
            return int(float(count_text.replace('K', '')) * 1000)
        elif 'M' in count_text:
            return int(float(count_text.replace('M', '')) * 1000000)
        else:
            try:
                return int(count_text)
            except:
                return 0

    def scrape_time_range_search(self, query, start_date, end_date, max_tweets=1000):
        """Scrape tweets for a query within a specific time range"""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Build advanced search query
        search_query = f"{query} since:{start_str} until:{end_str} lang:en min_faves:5"
        logging.info(f"Historical search: {search_query}")
        
        self.driver.get(f"https://twitter.com/search?q={search_query}&src=typed_query&f=live")
        time.sleep(5)

        results = []
        seen = set()
        scrolls = 0
        last_h = self.driver.execute_script("return document.body.scrollHeight")

        while len(results) < max_tweets and scrolls < 8:
            cards = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            for c in cards:
                try:
                    url = c.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]').get_attribute("href")
                    tid = url.rstrip("/").split("/")[-1]
                    if tid in seen or self.db.record_exists("twitter_posts", tid):
                        continue
                    seen.add(tid)

                    text = c.find_element(By.CSS_SELECTOR, 'div[lang]').text
                    user_elem = c.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                    username = user_elem.text.split("@")[-1].split()[0]

                    results.append({
                        "id": tid,
                        "username": username,
                        "content": text,
                        "likes": 0,
                        "retweets": 0,
                        "replies": 0,
                        "created_at": datetime.utcnow().timestamp(),
                        "url": url
                    })
                    if len(results) >= max_tweets:
                        break
                except Exception:
                    continue

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_h = self.driver.execute_script("return document.body.scrollHeight")
            scrolls += (new_h == last_h)
            last_h = new_h

        logging.info(f"Collected {len(results)} tweets for time range {start_str} to {end_str}")
        return results

    def scrape_hybrid_historical(self, days_back=30, max_total=10000):
        """
        Hybrid approach: Combine profile scraping + historical searches
        Phase 1: Quality tweets from crypto influencers
        Phase 2: Volume tweets from historical searches
        """
        try:
            self.setup_driver()
            self.manual_login_flow()
            all_tweets = []
            
            print(f"🚀 Starting hybrid historical collection ({days_back} days back)")
            
            # PHASE 1: Profile-based collection (Quality)
            print("📊 Phase 1: Collecting from crypto influencer profiles...")
            profile_tweets = []
            for i, profile in enumerate(self.crypto_profiles[:10], 1):  # Limit to first 10 profiles
                print(f"  Scraping {i}/10: @{profile}")
                tweets = self.scrape_profile_tweets(profile, max_tweets=300)
                profile_tweets.extend(tweets)
                time.sleep(2)  # Be nice to Twitter
                
                if len(profile_tweets) >= max_total // 2:  # Use half quota for profiles
                    break
            
            all_tweets.extend(profile_tweets)
            print(f"✅ Phase 1 complete: {len(profile_tweets)} quality tweets collected")
            
            # PHASE 2: Historical time-range searches (Volume)
            print("📈 Phase 2: Historical searches for volume...")
            remaining_quota = max_total - len(all_tweets)
            
            if remaining_quota > 0:
                # Search in 1-week chunks going backwards
                end_date = datetime.now()
                search_tweets = []
                
                for week in range(0, days_back // 7):
                    start_date = end_date - timedelta(days=7)
                    
                    for query in ['ethereum', 'ETH', 'DeFi']:
                        if len(search_tweets) >= remaining_quota:
                            break
                            
                        print(f"  Searching '{query}' from {start_date.strftime('%m/%d')} to {end_date.strftime('%m/%d')}")
                        tweets = self.scrape_time_range_search(
                            query, 
                            start_date, 
                            end_date, 
                            max_tweets=remaining_quota // 6  # Distribute quota
                        )
                        search_tweets.extend(tweets)
                        time.sleep(3)
                    
                    end_date = start_date
                    if len(search_tweets) >= remaining_quota:
                        break
                
                all_tweets.extend(search_tweets)
                print(f"✅ Phase 2 complete: {len(search_tweets)} historical tweets collected")
            
            # Save all collected tweets
            if not all_tweets:
                print("❌ No new tweets collected")
                return 0
            
            df = pd.DataFrame(all_tweets).drop_duplicates(subset=["id"])
            inserted = self.db.insert_twitter_posts(df)
            
            print(f"🎯 HYBRID COLLECTION COMPLETE!")
            print(f"   📊 Profile tweets: {len(profile_tweets)}")
            print(f"   📈 Search tweets: {len(all_tweets) - len(profile_tweets)}")
            print(f"   💾 Total new tweets saved: {inserted}")
            
            logging.info(f"Hybrid collection: {inserted} tweets inserted from {len(all_tweets)} collected")
            return inserted
            
        except Exception as e:
            logging.error(f"Error in hybrid collection: {e}")
            print(f"❌ Error during collection: {e}")
            return 0
        finally:
            if self.driver:
                self.driver.quit()

    def scrape_crypto_profiles_exhaustive(self, max_per_profile=200):
        """EXHAUSTIVE profile collection - maximum data extraction"""
        try:
            self.setup_driver()
            self.manual_login_flow()
            all_tweets = []
            
            # Use ALL profiles but with smart batching
            total_profiles = len(self.crypto_profiles)
            batch_size = random.randint(8, 15)  # Variable batch sizes
            
            print(f"🚀 EXHAUSTIVE MODE: Collecting from {total_profiles} crypto profiles...")
            print(f"   Processing in batches of {batch_size} profiles")
            print(f"   Max {max_per_profile} tweets per profile")
            
            processed = 0
            for batch_start in range(0, min(25, total_profiles), batch_size):  # Reduced to 25 profiles for deeper collection
                batch_end = min(batch_start + batch_size, min(25, total_profiles))
                batch_profiles = self.crypto_profiles[batch_start:batch_end]
                
                print(f"\n📦 BATCH {batch_start//batch_size + 1}: Profiles {batch_start+1}-{batch_end}")
                
                for i, profile in enumerate(batch_profiles, 1):
                    print(f"  🔍 DEEP SCRAPING {batch_start + i}/{min(25, total_profiles)}: @{profile}")
                    print(f"    Targeting {max_per_profile} historical tweets...")
                    tweets = self.scrape_profile_tweets(profile, max_per_profile)
                    all_tweets.extend(tweets)
                    processed += 1
                    print(f"    ✅ Collected {len(tweets)} tweets from @{profile}")
                    
                    # Variable delays between profiles in batch
                    if i < len(batch_profiles):
                        delay = random.randint(5, 12)  # Longer delays for safety
                        print(f"    💤 Waiting {delay}s...")
                        time.sleep(delay)
                
                # Longer break between batches
                if batch_end < min(50, total_profiles):
                    batch_break = random.randint(15, 30)
                    print(f"🔄 Batch complete. Taking {batch_break}s break before next batch...")
                    time.sleep(batch_break)
            
            if not all_tweets:
                print("❌ No new tweets collected")
                return 0
            
            df = pd.DataFrame(all_tweets).drop_duplicates(subset=["id"])
            inserted = self.db.insert_twitter_posts(df)
            
            print(f"\n🎯 EXHAUSTIVE COLLECTION COMPLETE!")
            print(f"   📊 Profiles processed: {processed}")
            print(f"   📈 Total tweets collected: {len(all_tweets)}")
            print(f"   💾 New tweets saved: {inserted}")
            
            logging.info(f"Exhaustive profile collection: {inserted} tweets inserted from {processed} profiles")
            return inserted
            
        except Exception as e:
            logging.error(f"Error in exhaustive profile collection: {e}")
            print(f"❌ Error during collection: {e}")
            return 0
        finally:
            if self.driver:
                self.driver.quit()

    def scrape_crypto_profiles_only(self, max_per_profile=100):
        """Legacy method - redirects to exhaustive collection"""
        return self.scrape_crypto_profiles_exhaustive(max_per_profile)

    def test_twitter_access(self):
        """Test if we can access Twitter without getting blocked"""
        try:
            print("🧪 Testing Twitter access...")
            self.setup_driver()
            self.manual_login_flow()
            
            # Try accessing Vitalik's profile (most reliable test)
            self.driver.get("https://twitter.com/VitalikButerin")
            self.human_delay(3, 1)
            
            if self.check_for_blocks():
                print("❌ Access blocked - need to clear cookies/try later")
                return False
            
            # Try to find at least one tweet
            tweets = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            if len(tweets) > 0:
                print(f"✅ Access working - found {len(tweets)} tweets on page")
                return True
            else:
                print("⚠️ Access unclear - no tweets found")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            if self.driver:
                self.driver.quit()

    def scrape_all_queries(self, max_tweets_per_query=5000):
        try:
            self.setup_driver()
            self.manual_login_flow()
            all_tweets = []
            for q in self.queries:
                all_tweets.extend(self.scrape_search_results(q, max_tweets_per_query))
                time.sleep(2)

            if not all_tweets:
                logging.info("No new tweets collected.")
                return 0

            df = pd.DataFrame(all_tweets).drop_duplicates(subset=["id"])
            inserted = self.db.insert_twitter_posts(df)
            print(f"Total new tweets collected: {inserted}")
            logging.info(f"Inserted {inserted} new tweets")
            return inserted
        finally:
            if self.driver:
                self.driver.quit()

if __name__ == "__main__":
    import sys
    
    scraper = TwitterScraper()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test Twitter access
        scraper.test_twitter_access()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "hybrid":
        # Hybrid historical collection (REDUCED DEFAULTS)
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 15  # Reduced from 30
        max_tweets = int(sys.argv[3]) if len(sys.argv) > 3 else 1000  # Reduced from 5000
        print(f"Starting CONSERVATIVE hybrid collection: {days} days back, max {max_tweets} tweets")
        scraper.scrape_hybrid_historical(days_back=days, max_total=max_tweets)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "profiles":
        # Profile-only collection (CONSERVATIVE)
        max_per = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Reduced from 300
        print(f"Starting CONSERVATIVE profile collection: max {max_per} tweets per profile")
        scraper.scrape_crypto_profiles_only(max_per_profile=max_per)
    
    else:
        # Original search-based collection
        scraper.scrape_all_queries(max_tweets_per_query=1000)  # Reduced default
