# src/price_collector.py

"""
ETH price data collector using yfinance and CoinGecko.
Collects daily OHLCV data and stores with proper timestamps.
"""
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/price_collector.log'),
        logging.StreamHandler()
    ]
)

class PriceCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
    
    def collect_eth_prices_yfinance(self, days_back=365):
        """Collect ETH price data using yfinance."""
        try:
            logging.info(f"Collecting ETH prices for past {days_back} days via yfinance")
            
            # Get ETH-USD data
            eth = yf.Ticker("ETH-USD")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get historical data
            hist = eth.history(start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                logging.warning("No price data received from yfinance")
                return 0
            
            # Process and store data
            price_data = []
            for date, row in hist.iterrows():
                # Calculate 24h price change
                price_change_24h = 0
                if len(price_data) > 0:
                    prev_close = price_data[-1]['price_usd']
                    price_change_24h = ((row['Close'] - prev_close) / prev_close) * 100
                
                price_data.append({
                    'timestamp': date.timestamp(),
                    'price_usd': row['Close'],
                    'volume_24h': row['Volume'],
                    'market_cap': None,  # yfinance doesn't provide market cap
                    'price_change_24h': price_change_24h
                })
            
            # Insert into database
            inserted = self.insert_price_data(price_data)
            logging.info(f"Collected {len(price_data)} price points, {inserted} new inserted")
            return inserted
            
        except Exception as e:
            logging.error(f"Error collecting yfinance data: {e}")
            return 0
    
    def collect_eth_prices_coingecko(self, days_back=365):
        """Collect ETH price data using CoinGecko API (free tier)."""
        try:
            logging.info(f"Collecting ETH prices for past {days_back} days via CoinGecko")
            
            # CoinGecko API endpoint
            url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': min(days_back, 365),  # Free tier limit
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            price_data = []
            for i, (timestamp_ms, price) in enumerate(prices):
                timestamp = timestamp_ms / 1000  # Convert to seconds
                
                # Calculate 24h price change
                price_change_24h = 0
                if i > 0:
                    prev_price = prices[i-1][1]
                    price_change_24h = ((price - prev_price) / prev_price) * 100
                
                price_data.append({
                    'timestamp': timestamp,
                    'price_usd': price,
                    'volume_24h': volumes[i][1] if i < len(volumes) else 0,
                    'market_cap': market_caps[i][1] if i < len(market_caps) else 0,
                    'price_change_24h': price_change_24h
                })
            
            # Insert into database
            inserted = self.insert_price_data(price_data)
            logging.info(f"Collected {len(price_data)} price points, {inserted} new inserted")
            return inserted
            
        except Exception as e:
            logging.error(f"Error collecting CoinGecko data: {e}")
            return 0
    
    def insert_price_data(self, price_data):
        """Insert price data into database with duplicate checking."""
        import sqlite3
        
        new_count = 0
        conn = sqlite3.connect(self.db.db_path)
        try:
            cursor = conn.cursor()
            for data in price_data:
                # Check if timestamp already exists
                if cursor.execute(
                    "SELECT 1 FROM eth_prices WHERE timestamp = ? LIMIT 1",
                    (data['timestamp'],)
                ).fetchone():
                    continue
                
                cursor.execute("""
                    INSERT INTO eth_prices
                    (timestamp, price_usd, volume_24h, market_cap, price_change_24h)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    data['timestamp'],
                    data['price_usd'],
                    data['volume_24h'],
                    data['market_cap'],
                    data['price_change_24h']
                ))
                new_count += 1
            
            conn.commit()
        finally:
            conn.close()
        
        return new_count
    
    def collect_latest_price(self):
        """Collect just the latest ETH price (for daily cron job)."""
        return self.collect_eth_prices_coingecko(days_back=2)  # Get last 2 days
    
    def backfill_historical_prices(self, days_back=365):
        """Backfill historical price data."""
        # Try CoinGecko first (more comprehensive), fallback to yfinance
        inserted = self.collect_eth_prices_coingecko(days_back)
        if inserted == 0:
            logging.info("CoinGecko failed, trying yfinance...")
            inserted = self.collect_eth_prices_yfinance(days_back)
        return inserted

if __name__ == "__main__":
    collector = PriceCollector()
    
    # Backfill historical data
    count = collector.backfill_historical_prices(days_back=365)
    print(f"Total price data points collected: {count}")