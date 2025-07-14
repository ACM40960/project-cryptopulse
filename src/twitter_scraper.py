# src/twitter_scraper.py

"""
Twitter scraper using Selenium + Chrome profile, with duplicate skipping
and fields aligned to your schema.
"""
import os
import time
import logging
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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

    def setup_driver(self):
        options = Options()
        user = os.getenv("USER")
        for p in (
            f"/home/{user}/.config/google-chrome/Default",
            f"/home/{user}/.config/chromium/Default",
        ):
            if os.path.isdir(p):
                options.add_argument(f"--user-data-dir={p}")
                break
        options.add_argument("--start-maximized")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service("/usr/bin/chromedriver") if os.path.exists("/usr/bin/chromedriver") else None
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.implicitly_wait(10)
        logging.info("Chrome driver initialized")

    def manual_login_flow(self):
        self.driver.get("https://twitter.com/login")
        print("Please log into Twitter in the browser window, then press Enter here.")
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
    TwitterScraper().scrape_all_queries(max_tweets_per_query=20000)
