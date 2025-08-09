# CryptoPulse: A Comprehensive Analysis of Sentiment-Based Financial Prediction

---

## 1. Project Overview

CryptoPulse is a full-stack data science project that builds and evaluates a pipeline for predicting cryptocurrency price movements (specifically for Ethereum) using sentiment analysis from social media and news sources. 

The core thesis of this project is not to present a model with impossibly high accuracy, but rather to conduct a **critical re-evaluation of the entire methodology**. It demonstrates that while sentiment can be a predictive feature, its effectiveness is severely limited by real-world data sparsity and the high risk of model overfitting. The project serves as a case study in the importance of methodological rigor, model simplicity, and honest evaluation in the face of noisy, limited data.

**The key takeaway is that in financial machine learning, a model with lower but robustly-validated accuracy is far more valuable than a complex model with high but misleading metrics.**

---

## 2. Interactive Walkthrough

For a complete, hands-on walkthrough of the entire project, from data collection to final analysis, please see the main Jupyter Notebook:

➡️ **[CryptoPulse_Workflow.ipynb](./notebooks/CryptoPulse_Workflow.ipynb)**

---

## 3. Project Structure

The repository is organized into the following directories:

```
├── data/          # Contains the SQLite database and other data files (e.g., CSVs). This directory is ignored by Git.
├── docs/          # All project documentation, including this README, the scientific poster content, and other supplementary documents.
├── models/        # Saved machine learning models. This directory is ignored by Git.
├── notebooks/     # Jupyter Notebooks for analysis, experimentation, and the main project workflow.
├── reports/       # Generated reports, such as HTML or PDF versions of analyses. This directory is ignored by Git.
├── scripts/       # Automation scripts, such as shell scripts for setting up cron jobs for daily data collection.
├── src/           # All Python source code for the project.
│   ├── collection/  # Scripts for large-scale and experimental data collection.
│   ├── analysis/    # Scripts for data analysis and visualization.
│   └── ...          # Core scripts for scraping, processing, and modeling.
├── .gitignore     # Specifies intentionally untracked files to ignore (e.g., data, models, venv).
├── README.md      # This file.
└── requirements.txt # A list of all Python packages required to run the project.
```

---

## 4. The Complete Workflow

The project pipeline is executed in a sequence of steps, each handled by one or more scripts in the `src/` directory.

### Step 1: Data Collection
Raw data is collected from multiple sources to provide a diverse dataset:
- **Reddit:** Fetches posts from relevant subreddits (e.g., r/ethereum, r/cryptocurrency) using the `praw` library. See `src/reddit_scraper.py`.
- **News & RSS:** Scrapes articles from over 100 crypto news sources using the `feedparser` and `newspaper3k` libraries. See `collection/massive_rss_campaign.py`.
- **Twitter:** Scrapes tweets from influential crypto accounts using `selenium`. This is the most fragile part of the pipeline due to Twitter's anti-scraping measures. See `src/twitter_scraper.py`.
- **Price Data:** Collects historical price data for Ethereum (ETH) using the `yfinance` library. See `src/price_collector.py`.

### Step 2: Feature Engineering & Scoring
Once collected, the raw data is processed to create meaningful features for the machine learning models:
- **Sentiment Analysis:** Text data from Reddit, Twitter, and news is analyzed to determine sentiment (positive, negative, neutral). This project uses pre-trained NLP models like FinBERT. 
- **Custom Metrics:** In addition to simple sentiment, we engineered more nuanced metrics like `relevance_score`, `volatility_score`, and `echo_score` (measuring how much a story is being repeated across sources). See `src/score_metrics.py`.
- **Dataset Creation:** All data is aggregated by day and merged with the price data to create a final, time-series dataset ready for modeling. See `src/simplified_ml_dataset.py`.

### Step 3: Model Training
We train several models to compare their performance and robustness:
- **Baseline Model:** A simple model that uses only historical price data (technical indicators) to predict future price movements.
- **Simple Model (Logistic Regression):** A robust, interpretable model that serves as our "honest" baseline. It is less prone to overfitting on our small dataset. See `src/simple_model_trainer.py`.
- **Complex Model (LightGBM):** A powerful gradient boosting model that is highly effective but requires large amounts of data. On our limited dataset, this model is used to demonstrate the dangers of overfitting. See `src/ml_model_trainer.py`.

### Step 4: Analysis & Evaluation
The final step is to critically evaluate the models:
- **Performance Comparison:** We compare the models on standard metrics like Accuracy, Precision, Recall, and F1-Score.
- **Overfitting Diagnosis:** We show how the complex model achieves high accuracy by learning spurious correlations in the training data, leading to poor generalization on the test set. We specifically analyze its heavily biased predictions for "Up" vs. "Down" days.
- **Feature Importance:** We analyze which features are most predictive for each model, highlighting how the complex model relies on unreliable features like `content_length`. See `src/generate_plots.py`.

---

## 5. Techniques & Technologies

This project utilizes a wide range of data science technologies:

- **Programming Language:** Python 3
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM
- **Natural Language Processing (NLP):** Transformers (Hugging Face), FinBERT, NLTK
- **Web Scraping:** Selenium, Praw (for Reddit), Newspaper3k, Feedparser
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Database:** SQLite

---

## 6. Further Documentation

For a deeper dive into specific aspects of the project, please refer to the documents in the `docs/` folder:

- **[Scientific Poster Content](./docs/SCIENTIFIC_POSTER_COMPREHENSIVE_DOCUMENT.md):** A detailed summary of the project's methodology and findings, suitable for a poster presentation.
- **[Email to Project Guide](./docs/EMAIL_TO_PROJECT_GUIDE.md):** A candid assessment of the project's statistical challenges and limitations.

---

## 7. How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd CryptoPulse
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Explore the main workflow:**
    The best way to understand the project is to run the `CryptoPulse_Workflow.ipynb` notebook in the `notebooks/` directory.

4.  **Run individual scripts (optional):**
    You can also run the individual scripts in the `src/` directory.
    ```bash
    python src/reddit_scraper.py
    python src/ml_model_trainer.py
    ```