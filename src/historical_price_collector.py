# src/historical_price_collector.py

"""
Historical ETH price data collector to fill gaps from 2017-2024.
Uses multiple APIs for comprehensive coverage.
"""
import os
import time
import logging
import requests
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/historical_price_collector.log'),
        logging.StreamHandler()
    ]
)

class HistoricalPriceCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        
        # Multiple price data sources for reliability
        self.price_sources = {
            'coingecko': {
                'base_url': 'https://api.coingecko.com/api/v3',
                'historical_endpoint': '/coins/ethereum/history',
                'market_chart_endpoint': '/coins/ethereum/market_chart',
                'rate_limit': 1.2  # seconds between requests
            },
            'coinapi': {
                'base_url': 'https://rest.coinapi.io/v1',
                'historical_endpoint': '/ohlcv/ETH/USD/history',
                'api_key': os.getenv('COINAPI_KEY'),  # Optional
                'rate_limit': 0.1
            },
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data/v2',
                'historical_endpoint': '/histoday',
                'rate_limit': 1.0
            }
        }
    
    def collect_coingecko_historical(self, start_date, end_date):
        """Collect historical price data from CoinGecko API."""
        logging.info(f"Collecting CoinGecko data from {start_date} to {end_date}")
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # CoinGecko market chart endpoint for price history
        url = f"{self.price_sources['coingecko']['base_url']}/coins/ethereum/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            price_data = []
            
            # Extract price data
            if 'prices' in data:
                prices = data['prices']
                market_caps = data.get('market_caps', [])
                volumes = data.get('total_volumes', [])
                
                for i, price_point in enumerate(prices):
                    timestamp = price_point[0] // 1000  # Convert to seconds
                    price = price_point[1]
                    
                    # Get corresponding market cap and volume
                    market_cap = market_caps[i][1] if i < len(market_caps) else None
                    volume = volumes[i][1] if i < len(volumes) else None
                    
                    price_data.append({
                        'timestamp': timestamp,
                        'price_usd': price,
                        'market_cap': market_cap,
                        'volume_24h': volume,
                        'price_change_24h': None  # Will calculate separately
                    })
            
            logging.info(f"CoinGecko: Collected {len(price_data)} price points")
            return price_data
            
        except Exception as e:
            logging.error(f"CoinGecko collection failed: {e}")
            return []
    
    def collect_cryptocompare_historical(self, start_date, end_date):
        """Collect historical price data from CryptoCompare API."""
        logging.info(f"Collecting CryptoCompare data from {start_date} to {end_date}")
        
        url = f"{self.price_sources['cryptocompare']['base_url']}/histoday"
        
        # Calculate days difference
        days_diff = (end_date - start_date).days
        limit = min(days_diff, 2000)  # CryptoCompare limit
        
        params = {
            'fsym': 'ETH',
            'tsym': 'USD',
            'limit': limit,
            'toTs': int(end_date.timestamp())
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            price_data = []
            
            if data.get('Response') == 'Success' and 'Data' in data:
                for day_data in data['Data']['Data']:
                    timestamp = day_data['time']
                    
                    # Skip if before start date
                    if timestamp < start_date.timestamp():
                        continue
                    
                    price_data.append({
                        'timestamp': timestamp,
                        'price_usd': day_data['close'],
                        'market_cap': None,
                        'volume_24h': day_data.get('volumeto'),
                        'price_change_24h': None
                    })
            
            logging.info(f"CryptoCompare: Collected {len(price_data)} price points")
            return price_data
            
        except Exception as e:
            logging.error(f"CryptoCompare collection failed: {e}")
            return []
    
    def calculate_price_changes(self, price_data):
        """Calculate 24h price changes."""
        if len(price_data) < 2:
            return price_data
        
        # Sort by timestamp
        price_data = sorted(price_data, key=lambda x: x['timestamp'])
        
        for i in range(1, len(price_data)):
            current_price = price_data[i]['price_usd']
            previous_price = price_data[i-1]['price_usd']
            
            if current_price and previous_price:
                change_pct = ((current_price - previous_price) / previous_price) * 100
                price_data[i]['price_change_24h'] = change_pct
            else:
                price_data[i]['price_change_24h'] = 0.0
        
        # First entry has no previous data
        if price_data:
            price_data[0]['price_change_24h'] = 0.0
        
        return price_data
    
    def collect_for_target_months(self, target_months):
        """Collect price data for specific months that need coverage."""
        logging.info(f"Collecting price data for {len(target_months)} target months")
        
        all_price_data = []
        
        for year_month in target_months:
            try:
                # Parse year-month
                year, month = map(int, year_month.split('-'))
                
                # Create start and end dates for the month
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                logging.info(f"Collecting price data for {year_month}")
                
                # Try CoinGecko first (most reliable)
                month_data = self.collect_coingecko_historical(start_date, end_date)
                
                if not month_data:
                    # Fallback to CryptoCompare
                    logging.info(f"CoinGecko failed for {year_month}, trying CryptoCompare")
                    month_data = self.collect_cryptocompare_historical(start_date, end_date)
                
                if month_data:
                    all_price_data.extend(month_data)
                    logging.info(f"✅ {year_month}: Collected {len(month_data)} price points")
                else:
                    logging.warning(f"❌ {year_month}: No price data collected")
                
                # Rate limiting
                time.sleep(self.price_sources['coingecko']['rate_limit'])
                
            except Exception as e:
                logging.error(f"Failed to collect price data for {year_month}: {e}")
                continue
        
        return all_price_data
    
    def save_price_data(self, price_data):
        """Save price data to database."""
        if not price_data:
            return 0
        
        # Calculate price changes
        price_data = self.calculate_price_changes(price_data)
        
        # Create DataFrame
        df = pd.DataFrame(price_data)
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
        
        # Insert into database
        saved_count = 0
        conn = sqlite3.connect(self.db.db_path)
        try:
            for _, row in df.iterrows():
                try:
                    # Check if timestamp already exists
                    existing = conn.execute(
                        "SELECT 1 FROM eth_prices WHERE timestamp = ?", 
                        (row['timestamp'],)
                    ).fetchone()
                    
                    if not existing:
                        conn.execute(
                            """INSERT INTO eth_prices 
                               (timestamp, price_usd, volume_24h, market_cap, price_change_24h)
                               VALUES (?, ?, ?, ?, ?)""",
                            (row['timestamp'], row['price_usd'], row['volume_24h'], 
                             row['market_cap'], row['price_change_24h'])
                        )
                        saved_count += 1
                        
                except Exception as e:
                    logging.debug(f"Failed to insert price data point: {e}")
                    continue
            
            conn.commit()
        finally:
            conn.close()
        logging.info(f"Saved {saved_count} new price data points")
        return saved_count
    
    def collect_historical_for_gaps(self):
        """Main method to collect historical price data for identified gaps."""
        logging.info("=== Starting Historical Price Data Collection ===")
        
        # Target months with good Reddit data but missing price data
        target_months = [
            '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
            '2018-01', '2018-02', '2018-03', '2018-04', '2018-11',
            '2019-09', '2019-10',
            '2020-09',
            '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', 
            '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
            '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07',
            '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
            '2023-01', '2023-03', '2023-04', '2023-08', '2023-09', '2023-10', '2023-11',
            '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'
        ]
        
        # Collect price data for target months
        price_data = self.collect_for_target_months(target_months)
        
        # Save to database
        saved_count = self.save_price_data(price_data)
        
        logging.info(f"=== Historical Price Collection Complete ===")
        logging.info(f"Total collected: {len(price_data)} price points")
        logging.info(f"Total saved: {saved_count} new entries")
        
        return saved_count

def main():
    """Main function for historical price collection."""
    collector = HistoricalPriceCollector()
    total = collector.collect_historical_for_gaps()
    print(f"Historical price collection complete. Collected {total} new price points.")

if __name__ == "__main__":
    main()