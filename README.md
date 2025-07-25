# CryptoPulse 🚀

A comprehensive cryptocurrency sentiment analysis and price prediction system that collects multi-source data to predict Ethereum price movements using advanced sentiment analysis.

## 📋 Overview

CryptoPulse aggregates sentiment data from multiple sources including Reddit, Twitter, news articles, and GitHub activity to predict Ethereum price movements. The system uses a sophisticated 5-metric scoring algorithm to analyze text content and correlate it with price data for machine learning model training.

## 🎯 Features

- **Multi-Source Data Collection**: Reddit, Twitter, News, GitHub, Forums
- **Advanced Sentiment Analysis**: 5-metric scoring system (sentiment, relevance, volatility, echo, content depth)
- **Price Data Integration**: Historical ETH price data correlation
- **Scalable Collection System**: Parallel processing and RSS feed integration
- **ML-Ready Dataset**: Processed data suitable for time-series prediction models
- **Comprehensive Documentation**: Detailed logging and analysis tools

## 🏗️ Architecture

```
CryptoPulse/
├── src/                          # Core modules
│   ├── database.py              # Database operations and schema
│   ├── reddit_scraper.py        # Reddit data collection
│   ├── twitter_scraper.py       # Twitter data collection  
│   ├── news_scraper.py          # News article collection
│   ├── price_collector.py       # ETH price data collection
│   ├── score_metrics.py         # 5-metric sentiment scoring
│   ├── forum_scraper.py         # Forum and discussion boards
│   └── historical_*.py          # Historical data collectors
├── logs/                        # Collection and processing logs
├── scripts/                     # Utility and automation scripts
├── massive_rss_campaign.py      # Comprehensive RSS collection
├── ultimate_collection_system.py # Advanced multi-source collector
└── final_dataset_status.py      # Dataset analysis and metrics
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install requests pandas sqlite3 feedparser beautifulsoup4 python-dotenv
```

### Basic Usage

1. **Initialize Database**:
```python
from src.database import CryptoPulseDB
db = CryptoPulseDB()
```

2. **Collect Price Data**:
```python
python src/price_collector.py
```

3. **Collect Sentiment Data**:
```python
python src/reddit_scraper.py
python src/news_scraper.py
```

4. **Process with Sentiment Scoring**:
```python
python src/score_metrics.py
```

5. **Run Comprehensive Collection**:
```python
python massive_rss_campaign.py
```

### Advanced Collection

For large-scale data collection:

```python
# Ultimate collection system with 100+ RSS feeds
python ultimate_collection_system.py

# Targeted collection for specific time periods
python strategic_2021_expansion.py

# Efficient gap filling for sparse periods
python efficient_gap_filler.py
```

## 📊 Dataset

### Current Statistics
- **Total Entries**: 14,000+ processed sentiment records
- **Time Range**: 2021-2024 (focus on 2022-2023)
- **Sources**: Reddit (9,200+), News (3,900+), Twitter (1,300+)
- **Price Points**: 2,200+ ETH price records
- **Quality Score**: 55% (academic research grade)

### Data Schema

**Sentiment Data Tables**:
- `reddit_posts`: Reddit discussions and comments
- `twitter_posts`: Twitter posts and engagement
- `news_articles`: News articles and analysis
- `text_metrics`: Processed 5-metric scores

**Price Data**:
- `eth_prices`: Historical Ethereum price data with timestamps

### 5-Metric Scoring System

Each text entry is scored across 5 dimensions:

1. **Sentiment Score** (-1 to 1): Positive/negative sentiment analysis
2. **Relevance Score** (0 to 1): Ethereum-specific relevance  
3. **Volatility Score** (0 to 1): Potential market impact
4. **Echo Score** (0 to 1): Viral/discussion potential
5. **Content Depth** (0 to 1): Information richness and quality

## 🛠️ Configuration

### Environment Variables

Create a `.env` file:

```env
# API Keys (optional but recommended)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
NEWS_API_KEY=your_news_api_key

# Database Configuration
DATABASE_PATH=data/cryptopulse.db

# Collection Settings
COLLECTION_START_DATE=2022-01-01
COLLECTION_END_DATE=2024-01-01
MAX_WORKERS=8
```

## 📈 Performance

### Collection Efficiency
- **RSS Feeds**: 1,000+ articles in 4 minutes
- **Parallel Processing**: 12 concurrent workers
- **Rate Limiting**: Respectful API usage
- **Deduplication**: Automatic duplicate removal

### Dataset Quality
- **Daily Coverage**: 4.5 entries/day average
- **Consistency**: 35% of days with 5+ entries  
- **Relevance**: High-quality filtering with keyword scoring
- **ML Readiness**: Suitable for academic research and model training

## 🤖 Machine Learning Integration

The processed dataset is designed for time-series prediction models:

```python
# Example model training setup
import pandas as pd
from src.database import CryptoPulseDB

# Load processed data
db = CryptoPulseDB()
sentiment_data = db.get_processed_metrics()
price_data = db.get_price_data()

# Features: 5-metric scores + temporal features
# Target: Price movement (next day/week/month)
```

### Recommended Models
- **LSTM Networks**: For time-series prediction
- **Random Forest**: For feature importance analysis  
- **Ridge Regression**: For baseline performance
- **Transformer Models**: For advanced sequence modeling

## 📊 Analysis Tools

### Dataset Analysis
```python
# Comprehensive dataset status
python final_dataset_status.py

# Data distribution analysis  
python data_distribution_analysis.py

# Quality metrics assessment
python enhanced_dataset_summary.py
```

### Visualization
- Daily sentiment trends
- Price correlation analysis
- Source contribution metrics
- Quality score distributions

## 🔧 Advanced Features

### Custom Collection Strategies
- **Targeted RSS Collection**: Focus on specific time periods
- **International Sources**: Multi-language crypto news
- **Historical Archives**: Wayback Machine integration
- **API Integration**: Professional news services

### Processing Pipeline
- **Text Cleaning**: Automated preprocessing
- **Sentiment Analysis**: Multi-model ensemble
- **Relevance Filtering**: Ethereum-specific content
- **Quality Scoring**: Comprehensive content evaluation

## 📝 Development

### Adding New Data Sources

1. Create collector in `src/`:
```python
class NewSourceCollector:
    def __init__(self):
        self.db = CryptoPulseDB()
    
    def collect_data(self):
        # Collection logic
        pass
```

2. Add to main collection pipeline
3. Update database schema if needed
4. Add tests and documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 🎯 Roadmap

### Short-term
- [ ] Real-time data collection pipeline
- [ ] Advanced ML model implementations
- [ ] Web dashboard for monitoring
- [ ] API for external access

### Long-term
- [ ] Multi-cryptocurrency support
- [ ] Advanced NLP models
- [ ] Automated trading integration
- [ ] Academic research collaboration

## 📋 Requirements

### Core Dependencies
```
requests>=2.25.0
pandas>=1.3.0
sqlite3 (built-in)
feedparser>=6.0.0
beautifulsoup4>=4.9.0
python-dotenv>=0.19.0
```

### Optional Dependencies
```
scikit-learn>=1.0.0  # For ML models
matplotlib>=3.5.0    # For visualization
jupyter>=1.0.0       # For analysis notebooks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Data sources: Reddit, Twitter, CoinTelegraph, CoinDesk, Ethereum Foundation
- APIs: CryptoCompare, CoinGecko, Google News
- Community: Ethereum developers and crypto analysis community

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review logs in `/logs` for troubleshooting

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Cryptocurrency trading involves significant risk. Always do your own research and never invest more than you can afford to lose.

---

*Built with ❤️ for the crypto community*