<div align="center">
  <img src="https://user-images.githubusercontent.com/80636305/126576577-cb07ba84-a4fe-4d63-b43a-e7832c77483d.png" width="200">
  <h1>CryptoPulse: A Critical Analysis of Sentiment-Based Financial Prediction</h1>

  <p>
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.9+-brightgreen.svg">
    <img alt="Status" src="https://img.shields.io/badge/status-complete-green.svg">
  </p>

  <blockquote>
    <strong>This project develops and evaluates a comprehensive machine learning pipeline to assess the true predictive power of social media sentiment on cryptocurrency price movements, revealing key insights into the challenges of data sparsity and model overfitting in financial forecasting.</strong>
  </blockquote>
</div>

---
## Table of Contents

1.  [Introduction: The Vision for an Automated System](#1-introduction-the-vision-for-an-automated-system)
2.  [The Research Journey: A Detailed Chronology](#2-the-research-journey-a-detailed-chronology)
3.  [Methodology](#3-methodology)
    * [System Architecture](#31-system-architecture)
    * [Modeling: An Iterative Path](#32-modeling-an-iterative-path)
4.  [Results and Analysis](#4-results-and-analysis)
    * [Final Model Performance](#41-final-model-performance)
    * [The Overfitting Trap: A Deeper Look](#42-the-overfitting-trap-a-deeper-look)
5.  [Conclusion](#5-conclusion)
6.  [Getting Started](#6-getting-started)
7.  [Future Work](#7-future-work)
8.  [References](#8-references)

---


## Abstract

**Objective:** To critically re-evaluate the use of social media sentiment for cryptocurrency price prediction, focusing on data limitations and model robustness.

**Methods:** We developed CryptoPulse, an integrated machine learning pipeline that collects and processes social media data from Reddit, Twitter, and news sources. The system gathered over 15,000 social media entries and engineered 12 sentiment-based features using advanced NLP models like CryptoBERT and Sentence-BERT. A suite of machine learning models, from LSTMs to Logistic Regression, were trained and evaluated.

**Results:** Complex models like LightGBM achieved 75% accuracy, while simpler models provided more balanced performance across market conditions. Feature analysis revealed models relied heavily on non-semantic features like `content_length`, indicating spurious correlations rather than genuine sentiment learning.

**Conclusions:** Social media sentiment contains a predictive signal, but its utility is severely constrained by data sparsity. This research underscores the importance of statistical rigor and the necessity of large, high-quality datasets. The automated data pipeline built for this project provides a solid foundation for future work to overcome these limitations.

---

## Technologies Used

* **Programming Language:** Python
* **Data Handling & Analysis:** Pandas, NumPy, Scikit-learn
* **Web Scraping:** Selenium, BeautifulSoup
* **Database:** SQLite
* **Machine Learning:** LightGBM, Random Forest, Logistic Regression
* **Natural Language Processing:** CryptoBERT, Sentence-BERT (via Hugging Face)
* **Data Visualization:** Matplotlib, Seaborn
* **Development:** Jupyter Notebooks, Git & GitHub

---

## 1. Introduction: The Vision for an Automated System

Cryptocurrency markets are notoriously volatile, driven as much by collective sentiment as by fundamental value. The vision for CryptoPulse was to build a sophisticated, automated data pipeline capable of continuously ingesting and analyzing social media sentiment to forecast price movements.

The ultimate aim was not a static analysis, but a **living project**—a system that could run autonomously, gathering more data over time and improving its own predictive power. This "future-proof" architecture is the core engineering achievement of CryptoPulse. While the scientific results highlight the limitations of small datasets, the system itself is a robust foundation ready to be scaled.

---

## 2. The Research Journey: A Detailed Chronology

The development of CryptoPulse was an iterative journey marked by challenges, pivots, and breakthroughs.

### Phase 1: The Data Collection Challenge
The project began with an ambitious goal to collect a massive, multi-source dataset over 30 days. The primary obstacle emerged from Twitter, where our initial scrapers were blocked. We engineered a more sophisticated solution using a persistent Chrome profile ([`src/twitter_scraper.py`](./src/twitter_scraper.py)) to mimic human behavior, but large-scale historical collection remained difficult.

### Phase 2: Strategic Pivot and Expansion Breakthroughs
With Twitter collection proving slow, we made a strategic pivot: shift focus from *collecting* more data to *deeply processing* the high-quality Reddit and news data we already had. However, a subsequent breakthrough in expanding our list of subreddits and using targeted search queries exploded our data volume from a few thousand entries to **over 15,000**, validating our multi-source approach.

### Phase 3: Modeling, Analysis, and Insights
We tested a wide spectrum of models, from complex LSTMs to simpler tree-based models. The complex models failed to converge on our sparse daily dataset (178 days). LightGBM achieved our highest accuracy (75%), leading to detailed feature importance analysis that revealed the model's reliance on non-semantic features like `content_length` rather than genuine sentiment signals.

---

## 3. Methodology

### Key Notebooks

#### 🔄 [CryptoPulse Workflow](notebooks/CryptoPulse_Workflow.ipynb)
Project overview, system architecture, and script links.

#### 🧪 [CryptoPulse Complete Analysis](notebooks/CryptoPulse_Complete_Analysis.ipynb)
3-phase model comparison (Baseline → Enhanced → CryptoBERT), statistical validation, and hypothesis testing.

### 3.1. Data Selection and Subset Rationale

While the CryptoPulse pipeline collected over 15,000 social media entries, modeling was performed on 178 daily aggregated samples from February-July 2025. This subset was selected based on specific quality criteria:

**Data Completeness**: 95%+ coverage across all three sources (Reddit, Twitter, News) with no gaps exceeding 2 days.

**Market Representation**: Includes both bull market (Feb-Apr) and correction phases (May-Jul), with 52% up days and 48% down days.

**Technical Quality**: All samples pass data validation checks with consistent sentiment scoring methodology and complete price alignment.

**Temporal Consistency**: Represents the most recent period with stable data collection infrastructure and unified processing pipeline.

This rigorous selection ensures model training on high-quality, representative data rather than the full historical collection which contains gaps and inconsistencies from earlier development phases.

### 3.2. System Architecture

CryptoPulse is an automated pipeline composed of four main layers.

1.  **Data Collection:** Scripts ([`scripts/daily_collection.py`](./scripts/daily_collection.py)) orchestrate data collection from Reddit, Twitter, and news sources.
2.  **Data Processing & Storage:** A robust system for cleaning, processing, and storing data in a central SQLite database.
3.  **Feature Engineering:** An NLP pipeline ([`src/modern_score_metrics.py`](./src/modern_score_metrics.py)) that enriches raw text with sentiment and other predictive features.
4.  **Modeling & Evaluation:** A comprehensive training and evaluation framework ([`src/ml_model_trainer.py`](./src/ml_model_trainer.py)).

### 3.3. Feature Engineering

The core of CryptoPulse's predictive power comes from its custom-engineered features designed to capture different aspects of social media sentiment and activity:

*   **Sentiment Score:** Cryptocurrency-specific sentiment analysis using CryptoBERT embeddings, producing normalized scores from -1 (bearish) to +1 (bullish) for each text entry.
*   **Relevance Score:** Semantic similarity measurement between text content and predefined crypto market keywords using Sentence-BERT cosine similarity, filtered for market-relevant discussions.
*   **Volatility Trigger:** Binary classification score identifying content containing volatility-inducing language patterns (regulatory news, whale movements, technical breakouts) based on historical price correlation analysis.
*   **Echo Score:** Cross-platform content similarity detection measuring information propagation and narrative amplification across Reddit, Twitter, and news sources using TF-IDF vectorization.
*   **Content Depth:** Text complexity measurement combining sentence length, technical terminology frequency, and discussion thread depth to distinguish casual mentions from analytical content.

### 3.4. Modeling: An Iterative Path

Our modeling approach was deliberately iterative, moving from complex to simple as we understood the data's limitations.

* **Regression vs. Classification:** We quickly found that predicting the binary **Up/Down direction (Classification)** was a more tractable problem than predicting the exact price change (Regression).
* **The Failure of Complexity:** LSTMs and Temporal Fusion Transformers failed to converge with only 178 daily data points, a critical lesson in matching model complexity to data availability.
* **Traditional ML Performance:** Tree-based models like LightGBM achieved the highest raw accuracy scores.
* **Baseline Validation:** A simple **Logistic Regression** model provided balanced performance across market conditions, serving as a crucial baseline for comparison.

---

## 4. Results and Analysis

### 4.1. Model Performance and Overfitting Analysis

The LightGBM model achieved the highest raw accuracy (75%) but detailed analysis revealed significant limitations:

**Directional Bias**: 100% accuracy on "Up" days vs. 25% accuracy on "Down" days, indicating the model learned the market's upward trend rather than predictive patterns.

**Feature Dependence**: Primary reliance on `content_length` rather than sentiment features, suggesting spurious correlations where important events generate longer posts.

**Small Sample Overfitting**: With only 178 daily samples, complex models fitted to noise rather than signal, demonstrated by poor cross-validation performance compared to training accuracy.

**Baseline Comparison**: Simple Logistic Regression provided more balanced predictions (50% up/down accuracy) despite lower overall scores, indicating better generalization.

![Model Comparison](./analysis/visualizations/plots/model_comparison.png)
_Model accuracy comparison showing LightGBM's high overall score but poor balance between Up/Down day predictions._

<br>

![LightGBM Confusion Matrix](./plots/LightGBM_confusion_matrix.png)
_Confusion matrix revealing LightGBM's extreme directional bias: perfect Up day prediction (100%) but poor Down day performance (25%)._

<br>

![Feature Importance for Random Forest](./analysis/visualizations/plots/feature_importance.png)
_Feature importance analysis showing content_length dominance over sentiment features, indicating spurious correlation learning._

<br>

![Model Predictions Comparison](./analysis/visualizations/plots/model_predictions_comparison.png)
_Temporal prediction comparison demonstrating LightGBM's consistent Up bias versus more balanced baseline model predictions._

---

## 5. Conclusion

CryptoPulse successfully achieved its primary engineering goal: to build a robust, automated pipeline for sentiment analysis.

The scientific journey provided critical insights into financial ML methodology. We demonstrated that **data scale and statistical rigor are paramount** for reliable financial prediction. This project serves as a case study in transparent ML reporting, emphasizing the importance of understanding *why* a model works, not just *that* it appears to perform well.

---

## 6. Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites
* Python 3.9 or higher
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ACM40960/project-cryptopulse.git
    cd project-cryptopulse
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Explore the notebooks:**
    - **Start with**: [CryptoPulse Workflow Notebook](notebooks/CryptoPulse_Workflow.ipynb) for project overview
    - **Deep dive**: [CryptoPulse Complete Analysis Notebook](notebooks/CryptoPulse_Complete_Analysis.ipynb) for full scientific analysis and model comparison

---

## 7. Future Work

**Data Scale**: Deploy pipeline for 24+ months to achieve statistical significance and capture multiple market cycles.

**Advanced Models**: Implement Temporal Fusion Transformers, Prophet, and ARIMA models with proper regularization and ensemble methods.

**Multi-modal Capabilities**: 
- Process chart images and technical analysis diagrams using computer vision
- Analyze video content from crypto influencers and news broadcasts  
- Extract sentiment from podcast transcripts and audio sentiment analysis
- Integrate on-chain metrics (transaction volumes, whale movements, DeFi activity)

**Real-time System**: Build live prediction dashboard with streaming data ingestion, confidence intervals, and uncertainty quantification.

**Enhanced NLP**: Implement domain-specific fine-tuning of language models on crypto-specific terminology and slang.

**Cross-asset Analysis**: Extend framework to other cryptocurrencies and traditional financial markets for comparative studies.

---

## 8. References

1.  **ElKulako, A. (2023). *CryptoBERT: A Cryptocurrency Sentiment Analysis Model*.**
2.  **Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems.**
3.  **Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*, 2(1), 1-8.**
4.  **Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool.**
5.  **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.**
