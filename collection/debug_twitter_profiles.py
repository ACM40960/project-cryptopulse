#!/usr/bin/env python3
"""
Debug Twitter Profile Access Issues
Quick diagnostic to see what's happening with profile scraping
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def debug_twitter_profiles():
    """Debug specific Twitter profiles to understand accessibility issues"""
    
    print("🔍 DEBUGGING TWITTER PROFILE ACCESS")
    print("="*60)
    
    # Test with a mix of profiles - some definitely accessible
    test_profiles = [
        "VitalikButerin",    # Ethereum founder - should be accessible
        "elonmusk",          # Most followed account
        "ethereum",          # Official Ethereum account  
        "DefiIgnas",         # One that's showing as "not accessible"
        "evan_van_ness"      # Another showing as "not accessible"
    ]
    
    # Setup Chrome (same as the main script)
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Use system chromium driver
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # First check if we can access Twitter at all
        print("🌐 Testing basic Twitter access...")
        driver.get("https://twitter.com/home")
        time.sleep(3)
        
        current_url = driver.current_url
        page_title = driver.title
        print(f"📍 Twitter home URL: {current_url}")
        print(f"📄 Page title: {page_title}")
        
        # Test each profile
        for i, username in enumerate(test_profiles, 1):
            print(f"\\n{'='*40}")
            print(f"🧪 TEST {i}/{len(test_profiles)}: @{username}")
            print(f"{'='*40}")
            
            # Navigate to profile
            profile_url = f"https://twitter.com/{username}"
            print(f"🔗 Navigating to: {profile_url}")
            
            driver.get(profile_url)
            time.sleep(5)  # Wait for page load
            
            # Get current state
            current_url = driver.current_url
            page_title = driver.title
            page_source = driver.page_source.lower()
            
            print(f"📍 Final URL: {current_url}")
            print(f"📄 Page title: {page_title}")
            
            # Check for various indicators
            indicators = {
                "Profile in URL": username.lower() in current_url.lower(),
                "Username in title": username.lower() in page_title.lower(),
                "Contains 'tweet'": "tweet" in page_source,
                "Contains 'post'": "post" in page_source,
                "Shows suspended": "suspended" in page_source,
                "Shows protected": "protected" in page_source or "private" in page_source,
                "Login required": "log in" in page_source or "sign up" in page_source,
                "Rate limited": "rate limit" in page_source or "try again" in page_source,
                "Page not found": "doesn't exist" in page_source or "not found" in page_source
            }
            
            print("\\n🔍 PAGE ANALYSIS:")
            for indicator, present in indicators.items():
                status = "✅" if present else "❌"
                print(f"   {status} {indicator}")
            
            # Try to find tweet elements with various selectors
            tweet_selectors = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]', 
                'div[data-testid="tweet"]',
                '[role="article"]',
                'article',
                '[data-testid="cellInnerDiv"]'
            ]
            
            print("\\n📊 TWEET ELEMENT SEARCH:")
            tweets_found = 0
            working_selector = None
            
            for selector in tweet_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                count = len(elements)
                print(f"   {selector}: {count} elements")
                
                if count > 0 and tweets_found == 0:
                    tweets_found = count
                    working_selector = selector
            
            # If we found tweets, try to extract some data
            if tweets_found > 0:
                print(f"\\n✅ FOUND {tweets_found} TWEETS using: {working_selector}")
                
                try:
                    # Try to get first tweet's text
                    first_tweet = driver.find_element(By.CSS_SELECTOR, working_selector)
                    
                    # Try various text selectors
                    text_selectors = [
                        '[data-testid="tweetText"]',
                        'div[lang]',
                        'span[lang]'
                    ]
                    
                    tweet_text = ""
                    for text_sel in text_selectors:
                        try:
                            text_elem = first_tweet.find_element(By.CSS_SELECTOR, text_sel)
                            tweet_text = text_elem.text[:100]  # First 100 chars
                            print(f"   📝 Sample text ({text_sel}): {tweet_text}...")
                            break
                        except:
                            continue
                    
                    if not tweet_text:
                        print("   ⚠️ Could not extract tweet text")
                    
                except Exception as e:
                    print(f"   ❌ Error extracting tweet data: {e}")
            
            else:
                print("\\n❌ NO TWEETS FOUND")
                
                # Let's see what elements ARE on the page
                common_selectors = [
                    "div", "span", "article", "a", "button", "main", "section"
                ]
                
                print("\\n🔍 OTHER ELEMENTS ON PAGE:")
                for selector in common_selectors:
                    count = len(driver.find_elements(By.TAG_NAME, selector))
                    print(f"   {selector}: {count}")
            
            # Overall assessment
            if tweets_found > 0:
                print(f"\\n🎯 RESULT: ✅ @{username} is ACCESSIBLE ({tweets_found} tweets found)")
            elif "suspended" in page_source or "doesn't exist" in page_source:
                print(f"\\n🎯 RESULT: ❌ @{username} is SUSPENDED/DELETED")
            elif "protected" in page_source or "private" in page_source:
                print(f"\\n🎯 RESULT: 🔒 @{username} is PROTECTED/PRIVATE")
            elif "log in" in page_source:
                print(f"\\n🎯 RESULT: 🔐 @{username} REQUIRES LOGIN")
            else:
                print(f"\\n🎯 RESULT: ❓ @{username} STATUS UNCLEAR")
            
            # Take screenshot for debugging
            screenshot_path = f"logs/debug_{username}_profile.png"
            os.makedirs("logs", exist_ok=True)
            try:
                driver.save_screenshot(screenshot_path)
                print(f"📸 Screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"⚠️ Screenshot failed: {e}")
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
    
    finally:
        print("\\n🔄 Closing browser...")
        driver.quit()
        
    print("\\n🎯 DEBUGGING COMPLETE")
    print("Check screenshots in logs/ directory for visual confirmation")

if __name__ == "__main__":
    debug_twitter_profiles()