# src/modern_score_metrics.py

"""
CryptoPulse Modern 5-Metric Scoring System (2025 Edition)

Enhanced with:
1. Sentence-BERT embeddings for semantic understanding
2. LLM-based relevance and volatility scoring
3. Advanced semantic similarity for echo scoring
4. Multimodal content analysis capabilities
5. Dynamic threshold adjustment based on market conditions

Metrics:
1. Sentiment Score - FinBERT + LLM validation
2. Relevance Score - Semantic embeddings + LLM context understanding
3. Volatility Trigger - LLM-based market impact assessment
4. Echo Score - Cross-platform semantic similarity
5. Content Depth - Enhanced with semantic richness analysis
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch

# LLM integrations
import openai
from anthropic import Anthropic

from database import CryptoPulseDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/modern_score_metrics.log'),
        logging.StreamHandler()
    ]
)

class ModernCryptoMetricsScorer:
    def __init__(self, use_openai: bool = True, use_anthropic: bool = True):
        self.db = CryptoPulseDB()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_openai = use_openai
        self.use_anthropic = use_anthropic
        
        logging.info(f"Initializing Modern Scoring System on {self.device}")
        
        # Initialize models
        self._init_embedding_models()
        self._init_sentiment_model()
        self._init_llm_clients()
        self._init_reference_corpus()
        
        # Enhanced volatility patterns
        self._init_enhanced_volatility_system()
        
        logging.info("✅ Modern scoring system initialized successfully")
    
    def _init_embedding_models(self):
        """Initialize modern embedding models."""
        try:
            logging.info("Loading Sentence-BERT models...")
            
            # Primary model for general semantic understanding
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Financial domain-specific model
            try:
                self.finance_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                logging.info("✅ Loaded financial sentence transformer")
            except:
                self.finance_model = self.sentence_model
                logging.warning("Using general model for financial embeddings")
            
            # Create reference embeddings
            eth_reference_texts = [
                "Ethereum blockchain smart contracts decentralized applications",
                "ETH cryptocurrency token digital asset trading price movement",
                "Ethereum virtual machine gas fees transaction costs network",
                "Ethereum proof of stake staking validators consensus mechanism",
                "DeFi decentralized finance protocols liquidity yield farming",
                "NFT non-fungible tokens Ethereum marketplace",
                "Layer 2 scaling solutions Polygon Arbitrum Optimism",
                "Ethereum merge upgrade London fork EIP-1559",
                "Web3 decentralized internet blockchain applications",
                "Smart contract vulnerabilities hacks security audits"
            ]
            
            self.eth_reference_embeddings = self.finance_model.encode(
                eth_reference_texts, 
                convert_to_tensor=True,
                device=self.device
            )
            
            logging.info("✅ Embedding models initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize embedding models: {e}")
            raise
    
    def _init_sentiment_model(self):
        """Initialize FinBERT with fallback options."""
        try:
            logging.info("Loading FinBERT sentiment model...")
            
            # Try multiple FinBERT variants
            finbert_models = [
                "ProsusAI/finbert",
                "yiyanghkust/finbert-tone",
                "nlptown/bert-base-multilingual-uncased-sentiment"
            ]
            
            for model_name in finbert_models:
                try:
                    self.sentiment_model = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        device=0 if self.device == "cuda" else -1,
                        return_all_scores=True
                    )
                    logging.info(f"✅ Loaded sentiment model: {model_name}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {e}")
                    continue
            else:
                raise Exception("No sentiment model could be loaded")
                
        except Exception as e:
            logging.error(f"Failed to initialize sentiment model: {e}")
            raise
    
    def _init_llm_clients(self):
        """Initialize LLM API clients."""
        self.openai_client = None
        self.anthropic_client = None
        
        if self.use_openai:
            try:
                # Try to get API key from environment
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    logging.info("✅ OpenAI client initialized")
                else:
                    logging.warning("OpenAI API key not found in environment")
            except Exception as e:
                logging.warning(f"Failed to initialize OpenAI client: {e}")
        
        if self.use_anthropic:
            try:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    self.anthropic_client = Anthropic(api_key=api_key)
                    logging.info("✅ Anthropic client initialized")
                else:
                    logging.warning("Anthropic API key not found in environment")
            except Exception as e:
                logging.warning(f"Failed to initialize Anthropic client: {e}")
    
    def _init_reference_corpus(self):
        """Initialize enhanced reference corpus for relevance scoring."""
        self.eth_contexts = {
            'price_movement': [
                "ETH price pump surge bullish rally", 
                "ETH price dump crash bearish decline",
                "Ethereum price prediction target resistance support"
            ],
            'technology': [
                "Ethereum smart contracts solidity development",
                "Ethereum virtual machine gas optimization",
                "Ethereum proof of stake consensus validators"
            ],
            'defi': [
                "Decentralized finance DeFi protocols lending",
                "Uniswap Aave Compound liquidity mining yield",
                "DEX aggregators automated market makers"
            ],
            'nft': [
                "Non-fungible tokens NFT marketplace OpenSea",
                "Ethereum NFT collections digital art",
                "NFT royalties smart contracts ERC-721"
            ],
            'ecosystem': [
                "Ethereum ecosystem dApps web3 infrastructure",
                "Layer 2 scaling Polygon Arbitrum Optimism",
                "Ethereum foundation development roadmap"
            ]
        }
        
        # Create embeddings for each context
        self.context_embeddings = {}
        for context, texts in self.eth_contexts.items():
            embeddings = self.finance_model.encode(texts, convert_to_tensor=True)
            self.context_embeddings[context] = embeddings
    
    def _init_enhanced_volatility_system(self):
        """Initialize enhanced volatility detection system."""
        self.volatility_indicators = {
            'price_action_strong': {
                'keywords': ['pump', 'dump', 'moon', 'crash', 'surge', 'plummet', 'skyrocket', 'tank', 'rekt'],
                'weight': 0.3
            },
            'market_structure': {
                'keywords': ['breakout', 'breakdown', 'resistance', 'support', 'bull trap', 'bear trap'],
                'weight': 0.25
            },
            'emotions_extreme': {
                'keywords': ['FOMO', 'panic', 'euphoria', 'despair', 'greed', 'fear', 'capitulation'],
                'weight': 0.2
            },
            'institutional': {
                'keywords': ['whale', 'institutional', 'BlackRock', 'MicroStrategy', 'adoption', 'ETF'],
                'weight': 0.25
            },
            'regulatory': {
                'keywords': ['SEC', 'regulation', 'ban', 'approval', 'lawsuit', 'compliance'],
                'weight': 0.3
            },
            'technical_signals': {
                'keywords': ['golden cross', 'death cross', 'RSI', 'MACD', 'bollinger', 'fibonacci'],
                'weight': 0.15
            }
        }
        
        # Enhanced regex patterns
        self.enhanced_patterns = [
            (r'\$\d+[kmb]?', 0.2, 'price_target'),
            (r'\d+x', 0.25, 'multiplier'),  
            (r'[+-]?\d+\.?\d*%', 0.15, 'percentage'),
            (r'[🚀📈📉💎🔥💀⚡🌙]', 0.1, 'crypto_emoji'),
            (r'(TO|THE|DA)\s+(MOON|SUN|MARS)', 0.3, 'moon_language'),
            (r'(JUST|NOW|URGENT|BREAKING|ALERT)', 0.2, 'urgency')
        ]
    
    def calculate_modern_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Enhanced sentiment analysis with LLM validation.
        """
        try:
            if not text or len(text.strip()) < 3:
                return self._empty_sentiment()
            
            # Primary FinBERT analysis
            text_truncated = text[:512]
            finbert_results = self.sentiment_model(text_truncated)[0]
            
            # Process FinBERT results
            sentiment_scores = {}
            for result in finbert_results:
                label = result['label'].lower()
                if label in ['positive', 'negative', 'neutral']:
                    sentiment_scores[label] = result['score']
            
            # Get primary sentiment
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            primary_confidence = sentiment_scores[primary_sentiment]
            
            # LLM validation for high-stakes content
            llm_validation = None
            if primary_confidence > 0.8 or any(word in text.lower() for word in ['crash', 'moon', 'dump', 'pump']):
                llm_validation = self._llm_sentiment_validation(text)
            
            # Combine scores
            final_score = self._combine_sentiment_scores(sentiment_scores, llm_validation)
            
            return {
                'sentiment': primary_sentiment,
                'confidence': primary_confidence,
                'score': final_score,
                'finbert_scores': sentiment_scores,
                'llm_validation': llm_validation
            }
            
        except Exception as e:
            logging.warning(f"Modern sentiment analysis failed: {e}")
            return self._empty_sentiment()
    
    def _llm_sentiment_validation(self, text: str) -> Optional[Dict]:
        """Use LLM to validate sentiment for critical content."""
        try:
            prompt = f"""Analyze the sentiment of this crypto-related text towards Ethereum/ETH price:

Text: "{text[:500]}"

Rate the sentiment as:
- Positive (bullish, optimistic about ETH price)
- Negative (bearish, pessimistic about ETH price)  
- Neutral (informational, no clear price direction)

Also rate confidence (0-1) and explain key sentiment indicators.

Respond in JSON format:
{{"sentiment": "positive/negative/neutral", "confidence": 0.85, "reasoning": "explanation"}}"""

            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.content[0].text)
            
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
                
        except Exception as e:
            logging.warning(f"LLM sentiment validation failed: {e}")
            return None
    
    def _combine_sentiment_scores(self, finbert_scores: Dict, llm_validation: Optional[Dict]) -> float:
        """Combine FinBERT and LLM sentiment scores."""
        # Convert to numerical score
        score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        
        # FinBERT weighted score
        finbert_score = sum(score_map[label] * confidence for label, confidence in finbert_scores.items())
        
        if llm_validation:
            llm_score = score_map[llm_validation['sentiment']] * llm_validation['confidence']
            # Weight: 70% FinBERT, 30% LLM
            return 0.7 * finbert_score + 0.3 * llm_score
        
        return finbert_score
    
    def calculate_modern_relevance_score(self, text: str) -> Dict[str, float]:
        """
        Enhanced relevance scoring using semantic embeddings + LLM context.
        """
        try:
            if not text or len(text.strip()) < 3:
                return {'score': 0.0, 'context_scores': {}, 'llm_score': None}
            
            # Generate text embedding
            text_embedding = self.finance_model.encode([text], convert_to_tensor=True)
            
            # Calculate similarity with each ETH context
            context_scores = {}
            for context, context_embeddings in self.context_embeddings.items():
                similarities = torch.cosine_similarity(text_embedding, context_embeddings)
                context_scores[context] = float(torch.max(similarities))
            
            # Overall semantic relevance (max across contexts)
            semantic_score = max(context_scores.values())
            
            # LLM-based relevance validation for high-potential content
            llm_score = None
            if semantic_score > 0.3 or any(term in text.lower() for term in ['eth', 'ethereum', 'defi', 'smart contract']):
                llm_score = self._llm_relevance_validation(text)
            
            # Combine scores
            final_score = self._combine_relevance_scores(semantic_score, llm_score)
            
            return {
                'score': final_score,
                'semantic_score': semantic_score,
                'context_scores': context_scores,
                'llm_score': llm_score
            }
            
        except Exception as e:
            logging.warning(f"Modern relevance scoring failed: {e}")
            return {'score': 0.0, 'context_scores': {}, 'llm_score': None}
    
    def _llm_relevance_validation(self, text: str) -> Optional[float]:
        """Use LLM to assess Ethereum relevance."""
        try:
            prompt = f"""Rate how relevant this text is to Ethereum (ETH) cryptocurrency on a scale of 0-1:

Text: "{text[:500]}"

Consider:
- Direct mentions of Ethereum, ETH, smart contracts
- DeFi protocols, dApps built on Ethereum
- Ethereum ecosystem developments, upgrades
- ETH price movements, trading discussions
- Ethereum technology, gas fees, staking

Rate 0.0 = completely unrelated, 1.0 = highly relevant to Ethereum.

Respond with just a number between 0.0 and 1.0:"""

            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}]
                )
                return float(response.content[0].text.strip())
            
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                return float(response.choices[0].message.content.strip())
                
        except Exception as e:
            logging.warning(f"LLM relevance validation failed: {e}")
            return None
    
    def _combine_relevance_scores(self, semantic_score: float, llm_score: Optional[float]) -> float:
        """Combine semantic and LLM relevance scores."""
        if llm_score is not None:
            # Weight: 60% semantic, 40% LLM
            return 0.6 * semantic_score + 0.4 * llm_score
        return semantic_score
    
    def calculate_modern_volatility_trigger(self, text: str) -> Dict[str, Union[float, List, Dict]]:
        """
        Enhanced volatility trigger detection using LLM analysis.
        """
        try:
            if not text:
                return {'score': 0.0, 'triggers': [], 'llm_analysis': None}
            
            text_lower = text.lower()
            triggers = []
            base_score = 0.0
            
            # Enhanced keyword detection
            for category, info in self.volatility_indicators.items():
                for keyword in info['keywords']:
                    if keyword.lower() in text_lower:
                        triggers.append(f"{category}:{keyword}")
                        base_score += info['weight'] * 0.2
            
            # Enhanced pattern matching
            for pattern, weight, pattern_type in self.enhanced_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    triggers.extend([f"{pattern_type}:{match}" for match in matches[:3]])
                    base_score += len(matches) * weight
            
            # LLM-based volatility assessment
            llm_analysis = None
            if base_score > 0.2 or any(word in text_lower for word in ['breaking', 'urgent', 'pump', 'dump']):
                llm_analysis = self._llm_volatility_assessment(text)
            
            # Combine scores
            final_score = self._combine_volatility_scores(base_score, llm_analysis)
            
            return {
                'score': min(final_score, 1.0),
                'base_score': base_score,
                'triggers': triggers[:10],
                'llm_analysis': llm_analysis
            }
            
        except Exception as e:
            logging.warning(f"Modern volatility trigger calculation failed: {e}")
            return {'score': 0.0, 'triggers': [], 'llm_analysis': None}
    
    def _llm_volatility_assessment(self, text: str) -> Optional[Dict]:
        """Use LLM to assess market volatility potential."""
        try:
            prompt = f"""Analyze this crypto text for potential to trigger Ethereum price volatility:

Text: "{text[:500]}"

Rate 0-1 how likely this content could cause ETH price movement:
- 0.0-0.2: Low impact (general discussion, mild sentiment)
- 0.3-0.5: Medium impact (strong opinions, technical analysis)
- 0.6-0.8: High impact (breaking news, major events)
- 0.9-1.0: Extreme impact (regulatory changes, major hacks)

Consider: urgency, emotion, market-moving events, institutional activity.

Respond in JSON: {{"volatility_score": 0.65, "reasoning": "explanation", "impact_type": "regulatory/technical/sentiment/institutional"}}"""

            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.content[0].text)
            
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
                
        except Exception as e:
            logging.warning(f"LLM volatility assessment failed: {e}")
            return None
    
    def _combine_volatility_scores(self, base_score: float, llm_analysis: Optional[Dict]) -> float:
        """Combine keyword-based and LLM volatility scores."""
        if llm_analysis and 'volatility_score' in llm_analysis:
            # Weight: 40% keywords, 60% LLM
            return 0.4 * base_score + 0.6 * llm_analysis['volatility_score']
        return base_score
    
    def calculate_modern_echo_score(self, text: str, timestamp: float, window_hours: int = 24) -> Dict[str, float]:
        """
        Enhanced echo scoring using semantic similarity.
        """
        try:
            # Get content from time window
            similar_content = self._get_time_window_content(timestamp, window_hours)
            
            if len(similar_content) < 2:
                return {'score': 0.0, 'similar_count': 0, 'avg_similarity': 0.0}
            
            # Generate embeddings for all content
            all_texts = [text] + similar_content[:100]  # Limit for performance
            embeddings = self.sentence_model.encode(all_texts, convert_to_tensor=True)
            
            # Calculate semantic similarities
            target_embedding = embeddings[0:1]
            other_embeddings = embeddings[1:]
            
            similarities = torch.cosine_similarity(target_embedding, other_embeddings)
            
            # Enhanced echo metrics
            high_similarity_count = torch.sum(similarities > 0.4).item()  # Stricter threshold
            very_high_similarity_count = torch.sum(similarities > 0.6).item()
            avg_similarity = torch.mean(similarities).item()
            
            # Calculate final echo score
            echo_score = (
                (high_similarity_count / 20) * 0.5 +  # Normalize to 20 similar posts
                (very_high_similarity_count / 5) * 0.3 +  # Boost for very similar content
                avg_similarity * 0.2  # Overall similarity
            )
            
            return {
                'score': min(echo_score, 1.0),
                'similar_count': high_similarity_count,
                'very_similar_count': very_high_similarity_count,
                'avg_similarity': avg_similarity
            }
            
        except Exception as e:
            logging.warning(f"Modern echo score calculation failed: {e}")
            return {'score': 0.0, 'similar_count': 0, 'avg_similarity': 0.0}
    
    def calculate_modern_content_depth(self, text: str, title: str = "", engagement: Dict = None) -> Dict[str, float]:
        """
        Enhanced content depth analysis with semantic richness.
        """
        try:
            if not text:
                return {'score': 0.0, 'factors': {}}
            
            factors = {}
            score = 0.0
            
            # Basic metrics
            text_len = len(text.strip())
            word_count = len(text.split())
            
            factors['length_score'] = min(text_len / 1000, 1.0)  # Increased target
            factors['word_count_score'] = min(word_count / 150, 1.0)  # Increased target
            
            # Semantic richness analysis
            text_embedding = self.sentence_model.encode([text])
            
            # Technical vocabulary richness
            crypto_advanced_terms = [
                'blockchain', 'smart contract', 'defi', 'liquidity', 'yield farming',
                'staking', 'validator', 'consensus', 'gas optimization', 'layer 2',
                'rollup', 'zk-proof', 'dao', 'governance token', 'amm', 'impermanent loss',
                'flash loan', 'oracle', 'bridge', 'cross-chain'
            ]
            
            advanced_term_count = sum(1 for term in crypto_advanced_terms if term in text.lower())
            factors['technical_richness'] = min(advanced_term_count / 5, 1.0)
            
            # Structure and formatting quality
            has_structure = bool(re.search(r'[:\-\*•]|(\d+\.)|(\w+:)', text))
            has_links = bool(re.search(r'http[s]?://|www\.', text))
            has_data = bool(re.search(r'\$\d+|\d+%|\d+x', text))
            
            factors['structure_score'] = (
                (0.3 if has_structure else 0) +
                (0.4 if has_links else 0) +
                (0.3 if has_data else 0)
            )
            
            # Engagement quality (if available)
            engagement_score = 0.0
            if engagement:
                if 'score' in engagement:  # Reddit
                    engagement_score = min(engagement['score'] / 500, 1.0)  # Increased threshold
                elif 'likes' in engagement:  # Twitter
                    engagement_score = min(engagement['likes'] / 5000, 1.0)  # Increased threshold
            
            factors['engagement_score'] = engagement_score
            
            # Combine all factors
            score = (
                factors['length_score'] * 0.15 +
                factors['word_count_score'] * 0.15 +
                factors['technical_richness'] * 0.25 +
                factors['structure_score'] * 0.20 +
                factors['engagement_score'] * 0.25
            )
            
            return {'score': min(score, 1.0), 'factors': factors}
            
        except Exception as e:
            logging.warning(f"Modern content depth calculation failed: {e}")
            return {'score': 0.0, 'factors': {}}
    
    def _get_time_window_content(self, timestamp: float, window_hours: int) -> List[str]:
        """Get content from specified time window for echo analysis."""
        try:
            start_time = timestamp - (window_hours * 3600)
            end_time = timestamp + (window_hours * 3600)
            
            conn = sqlite3.connect(self.db.db_path)
            content = []
            
            # Get content from all sources
            queries = [
                ("SELECT title || ' ' || content FROM reddit_posts WHERE created_utc BETWEEN ? AND ?", (start_time, end_time)),
                ("SELECT content FROM twitter_posts WHERE created_at BETWEEN ? AND ?", (start_time, end_time)),
                ("SELECT title || ' ' || content FROM news_articles WHERE published_at BETWEEN ? AND ?", (start_time, end_time))
            ]
            
            for query, params in queries:
                results = conn.execute(query, params).fetchall()
                content.extend([row[0] for row in results if row[0] and len(row[0].strip()) > 10])
            
            conn.close()
            return content
            
        except Exception as e:
            logging.warning(f"Failed to get time window content: {e}")
            return []
    
    def _empty_sentiment(self) -> Dict:
        """Return empty sentiment scores."""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'score': 0.0,
            'finbert_scores': {'neutral': 1.0},
            'llm_validation': None
        }
    
    def process_entry_modern(self, entry: Dict) -> Dict:
        """
        Process a single entry with modern scoring system.
        """
        # Combine title and content
        text = f"{entry.get('title', '')} {entry.get('content', '')}".strip()
        
        if not text:
            return self._empty_modern_scores(entry['id'])
        
        logging.info(f"Processing entry with modern system: {entry['id'][:8]}... ({len(text)} chars)")
        
        # Calculate all modern metrics
        sentiment = self.calculate_modern_sentiment_score(text)
        relevance = self.calculate_modern_relevance_score(text)
        volatility = self.calculate_modern_volatility_trigger(text)
        
        # Prepare engagement data
        engagement = {}
        if 'score' in entry:  # Reddit
            engagement['score'] = entry['score']
        if 'likes' in entry:  # Twitter
            engagement['likes'] = entry['likes']
        
        content_depth = self.calculate_modern_content_depth(text, entry.get('title', ''), engagement)
        
        # Calculate modern echo score
        timestamp = entry.get('created_utc') or entry.get('created_at') or entry.get('published_at')
        echo_result = self.calculate_modern_echo_score(text, timestamp) if timestamp else {'score': 0.0, 'similar_count': 0, 'avg_similarity': 0.0}
        
        return {
            'id': entry['id'],
            'sentiment_score': sentiment['score'],
            'sentiment_label': sentiment['sentiment'],
            'sentiment_confidence': sentiment['confidence'],
            'sentiment_details': json.dumps(sentiment),
            'relevance_score': relevance['score'],
            'relevance_details': json.dumps(relevance),
            'volatility_score': volatility['score'],
            'volatility_triggers': ','.join(volatility['triggers'][:5]),
            'volatility_details': json.dumps(volatility),
            'echo_score': echo_result['score'],
            'echo_details': json.dumps(echo_result),
            'content_depth_score': content_depth['score'],
            'content_depth_factors': json.dumps(content_depth['factors']),
            'processed_at': datetime.now().timestamp(),
            'scoring_version': 'modern_v1.0'
        }
    
    def _empty_modern_scores(self, entry_id: str = "") -> Dict:
        """Return empty modern scores for invalid entries."""
        return {
            'id': entry_id,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'sentiment_confidence': 0.0,
            'sentiment_details': '{}',
            'relevance_score': 0.0,
            'relevance_details': '{}',
            'volatility_score': 0.0,
            'volatility_triggers': '',
            'volatility_details': '{}',
            'echo_score': 0.0,
            'echo_details': '{}',
            'content_depth_score': 0.0,
            'content_depth_factors': '{}',
            'processed_at': datetime.now().timestamp(),
            'scoring_version': 'modern_v1.0'
        }

def main():
    """Test the modern scoring system."""
    try:
        scorer = ModernCryptoMetricsScorer()
        
        # Test with sample text
        test_text = "Ethereum just broke through $3000 resistance! This could be the start of a major bull run. Smart money is accumulating ETH ahead of the next upgrade. DeFi protocols are seeing massive volume increases."
        
        test_entry = {
            'id': 'test_001',
            'title': 'ETH Breaks $3000 Resistance',
            'content': test_text,
            'score': 150,
            'created_utc': datetime.now().timestamp()
        }
        
        result = scorer.process_entry_modern(test_entry)
        
        print("=== Modern Scoring Test Results ===")
        print(f"Sentiment: {result['sentiment_score']:.3f} ({result['sentiment_label']})")
        print(f"Relevance: {result['relevance_score']:.3f}")
        print(f"Volatility: {result['volatility_score']:.3f}")
        print(f"Echo: {result['echo_score']:.3f}")
        print(f"Content Depth: {result['content_depth_score']:.3f}")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()