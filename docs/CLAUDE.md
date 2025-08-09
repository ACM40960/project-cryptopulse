# CryptoPulse - Claude Session Context
**Last Updated**: August 7, 2025  
**Project Status**: 🚧 **UNDER RE-EVALUATION & STATISTICAL REFINEMENT**

## 🎯 **PROJECT OVERVIEW**
CryptoPulse is a **cryptocurrency price prediction system** that explores the impact of social media sentiment analysis on Ethereum price movements. Initial findings showed high accuracy, but further statistical analysis revealed limitations due to dataset size. The project is now focused on robust methodology and transparent reporting of model performance under data constraints.

### **🏆 KEY ACHIEVEMENTS:**
- **15,959 total data entries** collected and processed
- **14,851 entries** with modern AI sentiment scoring
- **8 trained ML models** (including a new simple, robust model)
- **Automated data collection** running every 6 hours via cron jobs
- **Complete directory cleanup** completed August 1, 2025
- **Identified and addressed statistical limitations** of small dataset

## 📁 **CURRENT DIRECTORY STRUCTURE**

### **🔧 CORE PRODUCTION FILES:**
```
src/
├── reddit_scraper.py              # Active Reddit data collection
├── twitter_scraper.py             # Active Twitter data collection  
├── news_scraper.py                # Active news data collection
├── price_collector.py             # ETH price data collection
├── modern_score_metrics.py        # AI-powered sentiment scoring (Sentence-BERT)
├── ml_model_trainer.py            # Production ML training
├── database.py                    # Database operations (SQLite)
├── comprehensive_model_comparison.py # Model evaluation framework
├── baseline_model_trainer.py      # Baseline model comparison
└── [4 other core files]           # Essential utilities

analysis/
├── visualizations/                 # Visualization scripts and plots
│   ├── create_clean_visualizations.py
│   ├── visualize_model_predictions.py
│   ├── create_prediction_visualizations.py
│   ├── plots/ (10 PNG files + CSV)
│   └── README.md
├── data_distribution_analysis.py
├── dataset_sufficiency_analysis.py
└── final_dataset_status.py
```

### **📊 PRODUCTION DATA & MODELS:**
```
data/
├── simplified_ml_dataset.csv      # FINAL PRODUCTION DATASET (178 samples)
├── simplified_ml_dataset_info.json
└── simplified_ml_dataset_feature_scores.csv

models/
├── LightGBM_direction_1d.joblib   # Best performing complex model
├── LogisticRegression_direction_1d.joblib # New simple model
├── baseline/                      # Baseline comparison models
└── advanced/                      # Advanced models (LSTM, etc.)

db/
└── cryptopulse.db                 # Main production database
```

### **🤖 AUTOMATION SYSTEM:**
```
scripts/
├── daily_collection.py           # Automated data collection (cron job)
└── daily_scoring.py             # Automated sentiment scoring (cron job)
```

### **📚 COMPREHENSIVE DOCUMENTATION (5,985+ lines):**
```
docs/
├── README.md                     # Documentation index
├── technical/                    # Implementation details
│   ├── COMPREHENSIVE_TECHNICAL_DOCUMENTATION.md
│   ├── DETAILED_FILE_INVENTORY.md
│   └── MODERN_SCORING_UPGRADE_SUMMARY.md
├── models/                       # Model documentation
│   ├── FINAL_MODEL_TESTING_REPORT.md
│   ├── COMPREHENSIVE_MODEL_PERFORMANCE_REPORT.md
│   └── CRYPTOBERT_INTEGRATION_REPORT.md
├── operations/                   # System operations
│   ├── PROJECT_STATUS.md
│   ├── COLLECTION_DOCUMENTATION.md
│   └── OPTIMAL_PERIOD_PROCESSING_PLAN.md
└── analysis/                     # Analysis & assessment
    ├── CRITICAL_ASSESSMENT_AND_QUESTIONS.md
    ├── COMPREHENSIVE_PROJECT_DOCUMENTATION.md
    └── CRYPTOPULSE_COMPREHENSIVE_REPORT.md
```

## 🤖 **CURRENT SYSTEM STATUS**

### **✅ OPERATIONAL COMPONENTS:**
1. **Data Collection**: Automated every 6 hours (Reddit, Twitter, News)
2. **Price Updates**: Hourly ETH price monitoring via price_collector.py
3. **Sentiment Processing**: Modern AI scoring using Sentence-BERT embeddings
4. **ML Pipeline**: 8 trained models, undergoing re-evaluation for robustness
5. **Database**: SQLite with optimized schema and 15,959+ entries

### **🎯 BEST PERFORMING MODEL (Under Re-evaluation):**
- **Algorithm**: LightGBM with sentiment-enhanced features
- **Accuracy**: 75% for 1-day ETH price direction prediction (on small test set)
- **Features**: 12 selected features combining sentiment + technical analysis
- **Status**: Performance needs validation on larger datasets; prone to overfitting

### **💡 NEW SIMPLE MODEL (Logistic Regression):**
- **Algorithm**: Logistic Regression with robust features
- **Accuracy**: 33.3% for 1-day ETH price direction prediction
- **Features**: 7 selected robust features (excluding content length)
- **Status**: Provides a more statistically sound baseline for comparison, less prone to overfitting.

### **🔄 CRON AUTOMATION:**
```bash
# Data collection every 6 hours
0 */6 * * * cd /home/thej/Desktop/CryptoPulse && python3 scripts/daily_collection.py

# Sentiment scoring 30 minutes after collection  
30 */6 * * * cd /home/thej/Desktop/CryptoPulse && python3 scripts/daily_scoring.py

# ETH price updates every hour
0 * * * * cd /home/thej/Desktop/CryptoPulse && python3 src/price_collector.py
```

## 🔧 **TECHNICAL IMPLEMENTATION DETAILS**

### **🧠 AI/ML STACK:**
- **Sentiment Analysis**: FinBERT, CryptoBERT integration
- **Modern Embeddings**: Sentence-BERT (all-MiniLM-L6-v2), MPNet
- **ML Algorithms**: LightGBM, RandomForest, XGBoost, LSTM Neural Networks
- **Feature Engineering**: 12-30 engineered features with statistical selection

### **📊 DATA SOURCES:**
- **Reddit**: 10,081+ posts from 43+ crypto subreddits
- **Twitter**: 1,731+ posts from crypto influencers and discussions  
- **News**: 4,147+ articles from 15+ major crypto news sources
- **Price Data**: 2,293+ ETH price points with complete 2017-2025 history

### **🔍 MODERN SCORING SYSTEM:**
1. **Sentiment Score**: Enhanced FinBERT with fallback models
2. **Relevance Score**: Sentence-BERT semantic similarity (+32% improvement)
3. **Volatility Trigger**: Advanced pattern detection (+92% improvement)
4. **Echo Score**: Semantic cross-platform correlation analysis  
5. **Content Depth**: Technical vocabulary + engagement analysis (+6% improvement)

## 📋 **IMPORTANT FILES TO KNOW**

### **📖 START HERE FOR NEW SESSIONS:**
1. **`README.md`** - Project overview and quick start
2. **`docs/README.md`** - Complete documentation index
3. **`docs/operations/PROJECT_STATUS.md`** - Current system status
4. **`docs/technical/COMPREHENSIVE_TECHNICAL_DOCUMENTATION.md`** - Complete technical details

### **🔧 KEY OPERATIONAL FILES:**
- **`src/database.py`** - Database operations and schema
- **`scripts/daily_collection.py`** - Main automation script
- **`data/simplified_ml_dataset.csv`** - Final production dataset
- **`models/LightGBM_direction_1d.joblib`** - Best performing model

### **⚙️ CONFIGURATION:**
- **`requirements.txt`** - Python dependencies
- **`Shared Doc.txt`** - Original project roadmap and design decisions
- **`.env`** - Environment variables (API keys, etc.)

## 🗂️ **RECENT CHANGES (August 1, 2025 - August 7, 2025)**

### **✅ DIRECTORY CLEANUP COMPLETED (August 1, 2025):**
- **Organized 5,985+ lines of documentation** into structured docs/ directory
- **Streamlined src/ directory** to 13 core production files
- **Archived 37+ experimental files** in organized archive/ structure
- **Maintained all operational functionality** with clean organization
- **Created comprehensive documentation index** for easy navigation

### **📊 VISUALIZATION SUITE ADDED (August 7, 2025):**
- **Created analysis/visualizations/** directory with 3 visualization scripts
- **Generated 2 comprehensive plots** showing model predictions and dataset analysis (overall and directional accuracy).
- **Key insight**: Confirmed that current 36-sample test set is too small for statistical significance and leads to misleading accuracy metrics.

### **🔄 DATA COLLECTION ATTEMPTS & ROLLBACK (August 7, 2025):**
- **Reddit Scraping Attempt**: Ran `reddit_scraper.py` for 2 years of historical data. Collected 672 new posts, but identified an issue where the `days_back` parameter was not correctly applied (still scraped for 180 days).
- **News Scraping Attempt (Direct)**: Ran `news_scraper.py` for 100 articles per source. Collected only 18 new articles due to rate limiting and 404 errors on many sources.
- **News Scraping Attempt (RSS)**: Ran `massive_rss_campaign.py`. Collected 387 new articles, primarily from Google News searches. Identified that many direct RSS feeds were empty or problematic.
- **Scoring Attempt & Rollback**: Attempted to process new data using `modern_score_metrics.py`. Encountered CUDA errors and then an `AttributeError` due to incorrect usage of the `ModernCryptoMetricsScorer` class. All changes related to this attempt (including database modifications and script changes) were rolled back to ensure project stability.

### **📁 ARCHIVE ORGANIZATION:**
```
archive/
├── experimental/     # 37+ collection experiments and advanced models
├── debug/           # 4 debug utilities  
├── analysis/        # 6 analysis scripts
└── testing/         # 4 test files
```

## 🚀 **NEXT STEPS CAPABILITY**

With current clean organization, the system supports:
1. **Real-time predictions** - Models ready for deployment
2. **System monitoring** - Comprehensive logging and status tracking
3. **Easy maintenance** - Well-documented, organized codebase
4. **Future enhancements** - Clean structure ready for new features
5. **Academic validation** - Complete methodology documentation

## ⚠️ **CRITICAL REMINDERS**

### **🔧 OPERATIONAL:**
- System requires continuous internet connection for data collection
- Cron jobs need to remain active for automated operation
- Database is SQLite-based for simplicity and portability
- All timestamps stored in UTC for consistency

### **📊 DATA & MODELS:**
- **Final dataset**: 178 samples (Feb-July 2025) in `simplified_ml_dataset.csv` (CRITICAL LIMITATION)
- **Best complex model**: LightGBM with 75% accuracy for 1-day ETH predictions (likely overfit)
- **New simple model**: Logistic Regression with 33.3% accuracy (more robust, less overfit)
- **Data processing**: 14,851 entries processed with modern AI metrics
- **Collection status**: 15,959 total entries across all sources
- **WARNING**: Small dataset size significantly impacts statistical validity and generalizability.

### **📚 DOCUMENTATION:**
- **Complete coverage**: Every file, model, and process documented
- **5,985+ lines** of comprehensive documentation organized by category
- **Critical assessments**: Methodology validation and statistical analysis included
- **Usage instructions**: Setup, operation, and maintenance procedures documented

---

**🎯 FOR NEW SESSIONS:** Start with `README.md` → `docs/README.md` → `docs/operations/PROJECT_STATUS.md` for complete context.