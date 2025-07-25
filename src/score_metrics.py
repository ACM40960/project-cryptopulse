# src/score_metrics.py

"""
CryptoPulse 5-Metric Scoring System

Extracts the following metrics from text entries:
1. Sentiment Score (FinBERT) - Financial sentiment analysis
2. Relevance Score (ETH cosine similarity) - How relevant to Ethereum
3. Volatility Trigger (keyword/LLM flag) - Likelihood to cause price movement
4. Echo Score (cross-platform topic match) - Cross-platform discussion intensity
5. Content Depth (quality indicators) - Content quality and informativeness

Processes all unprocessed entries in the database and stores computed scores.
"""
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/score_metrics.log'),
        logging.StreamHandler()
    ]
)

class CryptoMetricsScorer:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # Ethereum reference corpus for relevance scoring
        self.eth_reference_texts = [
            "ethereum blockchain smart contracts decentralized finance DeFi",
            "ETH token cryptocurrency digital asset trading price",
            "ethereum virtual machine EVM gas fees transactions",
            "ethereum 2.0 proof of stake staking consensus",
            "decentralized applications dApps ethereum ecosystem"
        ]
        
        # Initialize models
        self._init_sentiment_model()
        self._init_relevance_model()
        self._init_volatility_keywords()
        
    def _init_sentiment_model(self):
        """Initialize FinBERT model for financial sentiment analysis."""
        try:
            logging.info("Loading FinBERT sentiment model...")
            model_name = "ProsusAI/finbert"
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            logging.info("✅ FinBERT model loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load FinBERT, using fallback: {e}")
            # Fallback to simpler sentiment model
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
    
    def _init_relevance_model(self):
        """Initialize TF-IDF vectorizer for relevance scoring."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Fit on Ethereum reference corpus
        self.tfidf_vectorizer.fit(self.eth_reference_texts)
        self.eth_reference_vectors = self.tfidf_vectorizer.transform(self.eth_reference_texts)
        
    def _init_volatility_keywords(self):
        """Initialize volatility trigger keywords and patterns."""
        self.volatility_keywords = {
            'price_action': ['pump', 'dump', 'moon', 'crash', 'surge', 'plummet', 'skyrocket', 'tank'],
            'market_events': ['breakout', 'breakdown', 'resistance', 'support', 'bull', 'bear', 'rally'],
            'emotions': ['FOMO', 'panic', 'euphoria', 'despair', 'greed', 'fear'],
            'technical': ['RSI', 'MACD', 'fibonacci', 'pattern', 'trend', 'volume'],
            'fundamental': ['adoption', 'regulation', 'ban', 'approval', 'institutional', 'whale'],
            'urgency': ['now', 'urgent', 'breaking', 'alert', 'immediately', 'critical']
        }
        
        # Compile regex patterns
        self.volatility_patterns = [
            r'\$\d+[kmb]?',  # Price targets ($100, $5k, $1m)
            r'\d+x',         # Multipliers (10x, 100x)
            r'\d+%',         # Percentages (50%, 200%)
            r'[🚀📈📉💎🔥]',  # Crypto emojis
            r'(TO|THE|DA)\s+(MOON|SUN|MARS)',  # "TO THE MOON"
        ]
    
    def calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment score using FinBERT.
        Returns: {'sentiment': 'positive'|'negative'|'neutral', 'confidence': float, 'score': float}
        """
        try:
            if not text or len(text.strip()) < 3:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'score': 0.0}
            
            # Truncate text to model limits
            text = text[:512]
            
            results = self.sentiment_model(text)[0]
            
            # Convert to standardized format
            if isinstance(results, list):
                sentiment_map = {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                               'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
                
                best_result = max(results, key=lambda x: x['score'])
                sentiment = sentiment_map.get(best_result['label'], 'neutral')
                confidence = best_result['score']
                
                # Convert to score: positive=1, neutral=0, negative=-1
                score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
                score = score_map[sentiment] * confidence
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'score': score
                }
            
        except Exception as e:
            logging.warning(f"Sentiment analysis failed: {e}")
            
        return {'sentiment': 'neutral', 'confidence': 0.0, 'score': 0.0}
    
    def calculate_relevance_score(self, text: str) -> float:
        """
        Calculate relevance to Ethereum using cosine similarity.
        Returns: float between 0 and 1
        """
        try:
            if not text or len(text.strip()) < 3:
                return 0.0
            
            # Vectorize the input text
            text_vector = self.tfidf_vectorizer.transform([text.lower()])
            
            # Calculate cosine similarity with ETH reference corpus
            similarities = cosine_similarity(text_vector, self.eth_reference_vectors)
            
            # Return max similarity score
            return float(np.max(similarities))
            
        except Exception as e:
            logging.warning(f"Relevance scoring failed: {e}")
            return 0.0
    
    def calculate_volatility_trigger(self, text: str) -> Dict[str, float]:
        """
        Calculate volatility trigger score based on keywords and patterns.
        Returns: {'score': float, 'triggers': List[str]}
        """
        try:
            if not text:
                return {'score': 0.0, 'triggers': []}
            
            text_lower = text.lower()
            triggers = []
            score = 0.0
            
            # Check keyword categories
            for category, keywords in self.volatility_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        triggers.append(f"{category}:{keyword}")
                        score += 0.1
            
            # Check regex patterns
            for pattern in self.volatility_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    triggers.extend([f"pattern:{match}" for match in matches[:3]])
                    score += len(matches) * 0.15
            
            # Normalize score (cap at 1.0)
            score = min(score, 1.0)
            
            return {'score': score, 'triggers': triggers[:10]}  # Limit triggers
            
        except Exception as e:
            logging.warning(f"Volatility trigger calculation failed: {e}")
            return {'score': 0.0, 'triggers': []}
    
    def calculate_content_depth(self, text: str, title: str = "", engagement: Dict = None) -> Dict[str, float]:
        """
        Calculate content depth/quality score.
        Returns: {'score': float, 'factors': Dict}
        """
        try:
            if not text:
                return {'score': 0.0, 'factors': {}}
            
            factors = {}
            score = 0.0
            
            # Text length factor (normalized)
            text_len = len(text.strip())
            factors['length'] = min(text_len / 500, 1.0)  # Normalize to 500 chars
            score += factors['length'] * 0.2
            
            # Word count factor
            word_count = len(text.split())
            factors['word_count'] = min(word_count / 100, 1.0)  # Normalize to 100 words
            score += factors['word_count'] * 0.2
            
            # Technical terms (crypto vocabulary)
            crypto_terms = ['blockchain', 'smart contract', 'defi', 'nft', 'dao', 'liquidity',
                          'yield', 'staking', 'mining', 'consensus', 'validator', 'gas', 'gwei']
            tech_score = sum(1 for term in crypto_terms if term in text.lower()) / len(crypto_terms)
            factors['technical_terms'] = tech_score
            score += tech_score * 0.2
            
            # URL/Link presence (indicates external references)
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            has_links = bool(re.search(url_pattern, text))
            factors['has_links'] = 1.0 if has_links else 0.0
            score += factors['has_links'] * 0.1
            
            # Engagement factor (if available)
            if engagement:
                # Normalize engagement (varies by platform)
                eng_score = 0.0
                if 'score' in engagement:  # Reddit
                    eng_score = min(engagement['score'] / 100, 1.0)
                elif 'likes' in engagement:  # Twitter
                    eng_score = min(engagement['likes'] / 1000, 1.0)
                
                factors['engagement'] = eng_score
                score += eng_score * 0.3
            
            return {'score': min(score, 1.0), 'factors': factors}
            
        except Exception as e:
            logging.warning(f"Content depth calculation failed: {e}")
            return {'score': 0.0, 'factors': {}}
    
    def calculate_echo_score(self, text: str, timestamp: float, window_hours: int = 24) -> float:
        """
        Calculate echo score - how much this content is being discussed across platforms.
        Returns: float between 0 and 1
        """
        try:
            # Get similar content from other platforms within time window
            start_time = timestamp - (window_hours * 3600)
            end_time = timestamp + (window_hours * 3600)
            
            # Query database for content in time window
            conn = sqlite3.connect(self.db.db_path)
            
            # Get all content from around the same time
            similar_content = []
            
            # Reddit posts
            reddit_query = """
                SELECT title || ' ' || content as text FROM reddit_posts 
                WHERE created_utc BETWEEN ? AND ?
            """
            reddit_results = conn.execute(reddit_query, (start_time, end_time)).fetchall()
            similar_content.extend([row[0] for row in reddit_results])
            
            # Twitter posts  
            twitter_query = """
                SELECT content as text FROM twitter_posts 
                WHERE created_at BETWEEN ? AND ?
            """
            twitter_results = conn.execute(twitter_query, (start_time, end_time)).fetchall()
            similar_content.extend([row[0] for row in twitter_results])
            
            # News articles
            news_query = """
                SELECT title || ' ' || content as text FROM news_articles 
                WHERE published_at BETWEEN ? AND ?
            """
            news_results = conn.execute(news_query, (start_time, end_time)).fetchall()
            similar_content.extend([row[0] for row in news_results])
            
            conn.close()
            
            if len(similar_content) < 2:
                return 0.0
            
            # Calculate similarity with other content
            all_texts = [text] + similar_content[:100]  # Limit for performance
            try:
                vectors = self.tfidf_vectorizer.transform(all_texts)
                similarities = cosine_similarity(vectors[0:1], vectors[1:])
                
                # Echo score is based on how many similar discussions exist
                high_similarity_count = np.sum(similarities > 0.3)  # Threshold for similarity
                echo_score = min(high_similarity_count / 10, 1.0)  # Normalize
                
                return float(echo_score)
                
            except:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Echo score calculation failed: {e}")
            return 0.0

    def process_entry(self, entry: Dict) -> Dict:
        """
        Process a single text entry and calculate all 5 metrics.
        """
        # Combine title and content
        text = f"{entry.get('title', '')} {entry.get('content', '')}".strip()
        
        if not text:
            return self._empty_scores(entry['id'])
        
        logging.info(f"Processing entry: {entry['id'][:8]}... ({len(text)} chars)")
        
        # Calculate all metrics
        sentiment = self.calculate_sentiment_score(text)
        relevance = self.calculate_relevance_score(text)
        volatility = self.calculate_volatility_trigger(text)
        
        # Prepare engagement data
        engagement = {}
        if 'score' in entry:  # Reddit
            engagement['score'] = entry['score']
        if 'likes' in entry:  # Twitter
            engagement['likes'] = entry['likes']
        
        content_depth = self.calculate_content_depth(text, entry.get('title', ''), engagement)
        
        # Calculate echo score (computationally expensive)
        timestamp = entry.get('created_utc') or entry.get('created_at') or entry.get('published_at')
        echo_score = self.calculate_echo_score(text, timestamp) if timestamp else 0.0
        
        return {
            'id': entry['id'],
            'sentiment_score': sentiment['score'],
            'sentiment_label': sentiment['sentiment'],
            'sentiment_confidence': sentiment['confidence'],
            'relevance_score': relevance,
            'volatility_score': volatility['score'],
            'volatility_triggers': ','.join(volatility['triggers'][:5]),  # Store top 5
            'echo_score': echo_score,
            'content_depth_score': content_depth['score'],
            'content_depth_factors': str(content_depth['factors']),
            'processed_at': datetime.now().timestamp()
        }
    
    def _empty_scores(self, entry_id: str = "") -> Dict:
        """Return empty scores for invalid entries."""
        return {
            'id': entry_id,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'sentiment_confidence': 0.0,
            'relevance_score': 0.0,
            'volatility_score': 0.0,
            'volatility_triggers': '',
            'echo_score': 0.0,
            'content_depth_score': 0.0,
            'content_depth_factors': '{}',
            'processed_at': datetime.now().timestamp()
        }

    def get_unprocessed_entries(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed entries from all tables."""
        conn = sqlite3.connect(self.db.db_path)
        entries = []
        
        try:
            # Reddit posts without scores
            reddit_query = """
                SELECT r.id, r.subreddit as source, r.title, r.content, r.score, 
                       r.num_comments, r.created_utc, r.url, 'reddit' as platform
                FROM reddit_posts r
                LEFT JOIN text_metrics m ON r.id = m.id
                WHERE m.id IS NULL
                LIMIT ?
            """
            reddit_results = conn.execute(reddit_query, (max(1, limit//3),)).fetchall()
            
            for row in reddit_results:
                entries.append({
                    'id': row[0], 'source': row[1], 'title': row[2], 'content': row[3],
                    'score': row[4], 'num_comments': row[5], 'created_utc': row[6],
                    'url': row[7], 'platform': row[8]
                })
            
            # Twitter posts without scores
            twitter_query = """
                SELECT t.id, t.username as source, '' as title, t.content, t.likes,
                       t.retweets, t.created_at, t.url, 'twitter' as platform
                FROM twitter_posts t
                LEFT JOIN text_metrics m ON t.id = m.id
                WHERE m.id IS NULL
                LIMIT ?
            """
            twitter_results = conn.execute(twitter_query, (max(1, limit//3),)).fetchall()
            
            for row in twitter_results:
                entries.append({
                    'id': row[0], 'source': row[1], 'title': row[2], 'content': row[3],
                    'likes': row[4], 'retweets': row[5], 'created_at': row[6],
                    'url': row[7], 'platform': row[8]
                })
            
            # News articles without scores
            news_query = """
                SELECT n.id, n.source, n.title, n.content, 0 as engagement,
                       0 as extra, n.published_at, n.url, 'news' as platform
                FROM news_articles n
                LEFT JOIN text_metrics m ON n.id = m.id
                WHERE m.id IS NULL
                LIMIT ?
            """
            news_results = conn.execute(news_query, (max(1, limit//3),)).fetchall()
            
            for row in news_results:
                entries.append({
                    'id': row[0], 'source': row[1], 'title': row[2], 'content': row[3],
                    'engagement': row[4], 'published_at': row[6],
                    'url': row[7], 'platform': row[8]
                })
                
        finally:
            conn.close()
        
        return entries[:limit]
    
    def save_metrics(self, metrics: List[Dict]) -> int:
        """Save calculated metrics to database."""
        if not metrics:
            return 0
            
        # Create metrics table if it doesn't exist
        conn = sqlite3.connect(self.db.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS text_metrics (
                    id TEXT PRIMARY KEY,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    sentiment_confidence REAL,
                    relevance_score REAL,
                    volatility_score REAL,
                    volatility_triggers TEXT,
                    echo_score REAL,
                    content_depth_score REAL,
                    content_depth_factors TEXT,
                    processed_at REAL
                )
            """)
            
            # Insert metrics
            for metric in metrics:
                conn.execute("""
                    INSERT OR REPLACE INTO text_metrics
                    (id, sentiment_score, sentiment_label, sentiment_confidence,
                     relevance_score, volatility_score, volatility_triggers,
                     echo_score, content_depth_score, content_depth_factors, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric['id'], metric['sentiment_score'], metric['sentiment_label'],
                    metric['sentiment_confidence'], metric['relevance_score'], 
                    metric['volatility_score'], metric['volatility_triggers'],
                    metric['echo_score'], metric['content_depth_score'],
                    metric['content_depth_factors'], metric['processed_at']
                ))
            
            conn.commit()
            return len(metrics)
            
        finally:
            conn.close()
    
    def process_batch(self, batch_size: int = 50) -> int:
        """Process a batch of unprocessed entries."""
        logging.info(f"Processing batch of {batch_size} entries...")
        
        # Get unprocessed entries
        entries = self.get_unprocessed_entries(batch_size)
        if not entries:
            logging.info("No unprocessed entries found")
            return 0
        
        logging.info(f"Found {len(entries)} unprocessed entries")
        
        # Process entries
        metrics = []
        for i, entry in enumerate(entries):
            try:
                metric = self.process_entry(entry)
                metrics.append(metric)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(entries)} entries")
                    
            except Exception as e:
                logging.error(f"Error processing entry {entry['id']}: {e}")
                continue
        
        # Save metrics
        saved_count = self.save_metrics(metrics)
        logging.info(f"Saved {saved_count} metric sets")
        
        return saved_count
    
    def process_all_unprocessed(self, batch_size: int = 50, max_batches: int = None):
        """Process all unprocessed entries in batches."""
        logging.info("=== Starting CryptoPulse Metrics Processing ===")
        
        total_processed = 0
        batch_count = 0
        
        while True:
            if max_batches and batch_count >= max_batches:
                break
                
            processed = self.process_batch(batch_size)
            if processed == 0:
                break
                
            total_processed += processed
            batch_count += 1
            
            logging.info(f"Batch {batch_count} complete. Total processed: {total_processed}")
        
        logging.info(f"=== Processing Complete ===")
        logging.info(f"Total entries processed: {total_processed}")
        logging.info(f"Batches processed: {batch_count}")
        
        return total_processed

def main():
    """Main function for running metrics processing."""
    scorer = CryptoMetricsScorer()
    
    # Process all unprocessed entries
    total_processed = scorer.process_all_unprocessed(batch_size=25, max_batches=10)
    print(f"Metrics processing complete. Processed {total_processed} entries.")

if __name__ == "__main__":
    main()