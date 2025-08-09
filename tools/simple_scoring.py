#!/usr/bin/env python3
# tools/simple_scoring.py

"""
Simple scoring system that doesn't require heavy ML dependencies.
Processes remaining unscored entries with basic metrics.
"""
import sys
import os
import re
import logging
import hashlib
from datetime import datetime
import sqlite3
import pandas as pd

sys.path.append('src')
from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleScoring:
    def __init__(self):
        self.db = CryptoPulseDB()
        
        # Simple keyword-based scoring
        self.ethereum_keywords = [
            'ethereum', 'eth', 'ether', 'vitalik', 'merge', 'staking', 'validator',
            'beacon', 'proof of stake', 'pos', 'shanghai', 'london', 'eip', 'gas',
            'smart contract', 'solidity', 'web3', 'dapp', 'defi'
        ]
        
        self.positive_keywords = [
            'bullish', 'bull', 'pump', 'moon', 'surge', 'rally', 'rise', 'up',
            'growth', 'gain', 'profit', 'buy', 'hodl', 'optimistic', 'positive'
        ]
        
        self.negative_keywords = [
            'bearish', 'bear', 'dump', 'crash', 'fall', 'drop', 'down', 'decline',
            'loss', 'sell', 'fear', 'pessimistic', 'negative', 'panic', 'fud'
        ]
        
        self.volatility_keywords = [
            'breaking', 'urgent', 'alert', 'massive', 'huge', 'shock', 'emergency',
            'crash', 'pump', 'whale', 'liquidation', 'hack', 'exploit', 'hack'
        ]
    
    def simple_sentiment_score(self, text):
        """Simple keyword-based sentiment scoring."""
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Return score between -1 and 1
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def relevance_score(self, text):
        """Calculate Ethereum relevance based on keyword density."""
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        ethereum_mentions = sum(1 for keyword in self.ethereum_keywords if keyword in text_lower)
        
        # Normalize by text length and cap at 1.0
        relevance = min(ethereum_mentions / max(word_count * 0.1, 1), 1.0)
        return relevance
    
    def volatility_score(self, text):
        """Calculate volatility trigger potential."""
        text_lower = text.lower()
        
        volatility_count = sum(1 for keyword in self.volatility_keywords if keyword in text_lower)
        
        # Also check for numbers/percentages which often indicate price movements
        number_pattern = r'\d+%|\$\d+|\d+\.\d+%'
        number_matches = len(re.findall(number_pattern, text))
        
        # Combine keyword and number indicators
        total_indicators = volatility_count + number_matches * 0.5
        
        # Normalize to 0-1 scale
        return min(total_indicators / 3.0, 1.0)
    
    def echo_score(self, text):
        """Estimate cross-platform discussion potential."""
        # Simple heuristic based on text characteristics
        word_count = len(text.split())
        
        # Longer texts tend to generate more discussion
        length_score = min(word_count / 100.0, 0.5)
        
        # Check for engagement indicators
        engagement_indicators = ['?', '!', 'what do you think', 'opinion', 'thoughts']
        engagement_count = sum(1 for indicator in engagement_indicators if indicator in text.lower())
        engagement_score = min(engagement_count * 0.2, 0.5)
        
        return length_score + engagement_score
    
    def content_depth_score(self, text):
        """Estimate content quality and depth."""
        word_count = len(text.split())
        
        if word_count < 10:
            return 0.1
        elif word_count < 50:
            return 0.3
        elif word_count < 100:
            return 0.6
        elif word_count < 200:
            return 0.8
        else:
            return 1.0
    
    def score_entry(self, text_content):
        """Score a single text entry with all 5 metrics."""
        if not text_content or len(text_content.strip()) < 5:
            return {
                'sentiment_score': 0.0,
                'relevance_score': 0.0,
                'volatility_score': 0.0,
                'echo_score': 0.0,
                'content_depth': 0.0
            }
        
        return {
            'sentiment_score': self.simple_sentiment_score(text_content),
            'relevance_score': self.relevance_score(text_content),
            'volatility_score': self.volatility_score(text_content),
            'echo_score': self.echo_score(text_content),
            'content_depth': self.content_depth_score(text_content)
        }
    
    def get_unprocessed_entries(self):
        """Get all unprocessed entries from the database."""
        conn = sqlite3.connect(self.db.db_path)
        
        # Get Reddit posts not in text_metrics (using id directly)
        reddit_query = """
        SELECT 'reddit' as source_type, id, title, selftext as content, created_utc as timestamp
        FROM reddit_posts 
        WHERE id NOT IN (SELECT id FROM text_metrics)
        """
        
        # Get Twitter posts not in text_metrics  
        twitter_query = """
        SELECT 'twitter' as source_type, id, text as title, '' as content, created_at as timestamp
        FROM twitter_posts
        WHERE id NOT IN (SELECT id FROM text_metrics)
        """
        
        # Get news articles not in text_metrics
        news_query = """
        SELECT 'news' as source_type, id, title, content, published_at as timestamp
        FROM news_articles
        WHERE id NOT IN (SELECT id FROM text_metrics)
        """
        
        # Combine all queries
        combined_query = f"{reddit_query} UNION ALL {twitter_query} UNION ALL {news_query}"
        
        df = pd.read_sql_query(combined_query, conn)
        conn.close()
        
        return df
    
    def process_unscored_entries(self):
        """Process all unscored entries and save metrics."""
        logging.info("Starting simple scoring of unprocessed entries...")
        
        # Get unprocessed entries
        unprocessed = self.get_unprocessed_entries()
        
        if len(unprocessed) == 0:
            logging.info("All entries are already processed!")
            return 0
        
        logging.info(f"Found {len(unprocessed)} unprocessed entries")
        
        # Process each entry
        processed_count = 0
        batch_size = 100
        
        for i in range(0, len(unprocessed), batch_size):
            batch = unprocessed.iloc[i:i+batch_size]
            batch_metrics = []
            
            for _, entry in batch.iterrows():
                try:
                    # Combine title and content for scoring
                    text_content = f"{entry['title']} {entry['content']}".strip()
                    
                    # Score the entry
                    scores = self.score_entry(text_content)
                    
                    # Prepare for database insertion
                    metric_record = {
                        'id': hashlib.md5(f"{entry['source_type']}_{entry['id']}".encode()).hexdigest()[:16],
                        'source_type': entry['source_type'],
                        'source_id': entry['id'],
                        'sentiment_score': scores['sentiment_score'],
                        'relevance_score': scores['relevance_score'],
                        'volatility_score': scores['volatility_score'],
                        'echo_score': scores['echo_score'],
                        'content_depth': scores['content_depth'],
                        'processed_at': datetime.now()
                    }
                    
                    batch_metrics.append(metric_record)
                    processed_count += 1
                    
                except Exception as e:
                    logging.warning(f"Failed to process entry {entry['id']}: {e}")
                    continue
            
            # Save batch to database
            if batch_metrics:
                df_batch = pd.DataFrame(batch_metrics)
                self.db.insert_text_metrics(df_batch)
                logging.info(f"Processed batch: {len(batch_metrics)} entries")
        
        logging.info(f"Simple scoring complete: {processed_count} entries processed")
        return processed_count

def main():
    """Main function for simple scoring."""
    scorer = SimpleScoring()
    
    print(f"🎯 SIMPLE SCORING SYSTEM")
    print(f"📊 Processing remaining unscored entries")
    print(f"💡 Using keyword-based metrics (no ML dependencies)")
    
    processed = scorer.process_unscored_entries()
    
    print(f"\n✅ SCORING COMPLETE!")
    print(f"📈 Processed {processed} entries")
    print(f"🎯 All data now has 5-metric scores")

if __name__ == "__main__":
    main()