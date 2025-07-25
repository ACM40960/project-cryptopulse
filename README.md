# CryptoPulse

Cryptocurrency sentiment analysis and price prediction system that collects social media data to predict Ethereum price movements.

## Status: Work in Progress 🚧

This project is currently under active development. The core data collection and sentiment analysis systems are functional, but the ML models are still being developed.

## What's Working

- **Multi-source data collection** from Reddit, Twitter, news sources
- **Sentiment scoring system** with 5 metrics (sentiment, relevance, volatility, echo, depth)
- **Historical price data** integration
- **RSS feed collection** from 100+ crypto news sources
- **Database** with 14K+ processed entries

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run basic collection:
```bash
python src/reddit_scraper.py
python src/news_scraper.py
python src/price_collector.py
```

3. Process sentiment scores:
```bash
python src/score_metrics.py
```

## Project Structure

```
├── src/                    # Core scrapers and database
├── collection/             # Large-scale collection systems  
├── analysis/               # Dataset analysis tools
├── scripts/                # Automation and daily tasks
├── tools/                  # Testing and utilities
├── data/                   # Database and processed data
└── logs/                   # Collection logs
```

## Current Dataset

- **14K+ entries** across Reddit, Twitter, and news sources
- **2022-2023 focus** for optimal crypto market coverage
- **4.5 entries/day** average with sentiment scoring
- **Academic research quality** suitable for initial ML training

## Next Steps

- [ ] Complete ML model implementation
- [ ] Real-time prediction system
- [ ] Web interface for monitoring
- [ ] Expanded data sources

## Requirements

- Python 3.8+
- SQLite database
- Optional: Reddit/Twitter API keys for enhanced collection

---

*Note: This is educational/research project. Not financial advice.*