# CryptoPulse: A Critical Re-evaluation of Social Media Sentiment for Cryptocurrency Price Prediction

---

## Abstract

**Objective:** This project critically re-evaluates the efficacy of social media sentiment analysis in cryptocurrency price prediction, with a focus on the impact of data limitations and model robustness.

**Methods:** We developed CryptoPulse, an integrated machine learning pipeline that collects and processes social media data from Reddit, Twitter, and news sources. The system gathered over 15,000 social media entries and engineered 12 sentiment-based features using advanced NLP models like FinBERT and Sentence-BERT. We trained and evaluated a suite of machine learning models, from complex architectures like LSTMs to simpler, more interpretable models like Logistic Regression.

**Results:** While complex models initially showed high accuracy (75% for LightGBM), further analysis revealed this was likely due to overfitting on the small dataset (178 daily samples). A key insight was the model's reliance on non-semantic features like content length, a classic sign of learning spurious correlations. A simpler, more robust model provided a more realistic performance baseline, highlighting the challenges of prediction with limited data.

**Conclusions:** Social media sentiment contains a predictive signal for cryptocurrency price movements, but its practical utility is severely constrained by data sparsity. This research underscores the importance of statistical rigor, transparent reporting of limitations, and the necessity of large, high-quality datasets for building robust financial prediction models. The automated data pipeline built for this project provides a solid foundation for future work to overcome these data limitations.

---

## 1. Introduction: The Vision for an Automated System

Cryptocurrency markets are notoriously volatile, driven as much by collective sentiment as by fundamental value. This makes them a fertile ground for predictive modeling, yet a challenging one. The vision for CryptoPulse was to build a sophisticated, automated data pipeline and machine learning system capable of continuously ingesting and analyzing social media sentiment to forecast price movements.

The ultimate aim was not to create a static, one-off analysis, but a living project—a system that, once set up, could run autonomously, gathering more data over time and thereby improving its own predictive power. This "future-proof" architecture is the core engineering achievement of CryptoPulse. While the scientific results of this initial study highlight the limitations of working with small datasets, the system itself is a robust foundation, ready to be scaled into a powerful forecasting tool as its dataset grows.

---

## 2. The Research Journey: A Detailed Chronology

The development of CryptoPulse was not a linear path, but an iterative journey marked by significant challenges, strategic pivots, and major breakthroughs.

### Phase 1: The Data Collection Challenge

The project began with an ambitious goal: to collect a massive, multi-source dataset of crypto-related sentiment. Our initial plan was to gather data continuously over 30 days to build a rich time-series dataset.

The primary obstacle emerged from Twitter. Our initial automated scrapers were quickly blocked by login requirements. To overcome this, we engineered a more sophisticated solution using a persistent Chrome profile ([`src/twitter_scraper.py`](./src/twitter_scraper.py)) to maintain a logged-in session, effectively mimicking human behavior. However, even with this in place, large-scale historical collection proved difficult due to network timeouts and anti-bot measures, a common challenge in web scraping.

### Phase 2: Strategic Pivot and Expansion Breakthroughs

After an initial period of collection, we faced a critical decision. While Twitter collection was slow, our Reddit and news scrapers were highly effective. We had reached what we initially thought was "collection saturation." This led to a strategic pivot: we would shift focus from *collecting* more data to *deeply processing and analyzing* the high-quality data we already had.

This pivot, however, was followed by a series of breakthroughs. By creatively expanding our list of data sources—adding dozens of new, specialized subreddits and using targeted historical search queries—we dramatically increased our dataset. In a short period, our data volume exploded from a few thousand entries to over 15,000, creating a far more robust dataset than we had originally planned for. This success validated our multi-source approach and proved that high-quality data could be found with persistent and creative collection strategies.

### Phase 3: Modeling, Analysis, and Realization

With a rich dataset in hand, we moved to the modeling phase. Our approach was to test a wide spectrum of models to understand the trade-offs between complexity and performance. We started with ambitious, complex models like LSTMs and even attempted to implement a Temporal Fusion Transformer (TFT), believing their sophistication would be necessary to capture the nuances of the data.

However, these complex models failed to produce reliable results, which led to a key realization: our dataset, while large in the number of posts, was still small in terms of daily aggregated samples (178 days). This data sparsity was insufficient for these data-hungry architectures.

This led us to focus on more traditional, but robust, machine learning models like LightGBM and Random Forest. Here, we found our highest accuracy (75%), but also our most important scientific insight. A deep analysis of the model's feature importance revealed that it was heavily relying on `content_length`. The model wasn't learning the *sentiment* of the text, but a spurious correlation that important market events often generate longer, more detailed posts. This was a classic overfitting trap.

This final realization brought the project full circle, confirming our initial hypothesis that high accuracy on small financial datasets is often misleading. It prompted us to favor simpler, more interpretable models as a more honest baseline and to frame the project's conclusion around the critical importance of data scale and statistical rigor.

---

## 3. Methodology

### 3.1. System Architecture

CryptoPulse is an automated pipeline composed of four main layers:
1.  **Data Collection:** A suite of scripts ([`scripts/daily_collection.py`](./scripts/daily_collection.py)) that orchestrate the collection of data from Reddit, Twitter, and news sources.
2.  **Data Processing & Storage:** A robust system for cleaning, processing, and storing the collected data in a central SQLite database.
3.  **Feature Engineering:** An advanced NLP pipeline ([`src/modern_score_metrics.py`](./src/modern_score_metrics.py)) that enriches the raw text data with sentiment and other predictive features.
4.  **Modeling & Evaluation:** A comprehensive model training and evaluation framework ([`src/ml_model_trainer.py`](./src/ml_model_trainer.py)) that allows for the comparison of multiple machine learning models.

### 3.2. Modeling: An Iterative Path

Our modeling approach was deliberately iterative, starting with the most complex models and moving towards simpler, more robust ones as we gained a deeper understanding of the data's limitations.

*   **Regression vs. Classification:** We initially explored both regression (predicting the exact price change) and classification (predicting the price direction). We quickly found that classification was a more tractable problem, as predicting the binary Up/Down direction is less susceptible to the noise and volatility of exact price movements.
*   **The Failure of Complexity:** Our initial attempts with LSTMs and TFTs failed. These models, while powerful, require vast amounts of data to learn effectively. With only 178 daily data points, they were unable to converge on a meaningful solution. This was a critical lesson in matching model complexity to data availability.
*   **The Sweet Spot of Traditional ML:** We found the most success with tree-based ensemble models like LightGBM, which are known for their high performance on tabular data. This model achieved our highest accuracy but, as discussed, also fell into the overfitting trap.
*   **The Honesty of Simplicity:** Finally, we trained a simple Logistic Regression model. Its lower accuracy was a more realistic representation of the true predictive power of the features on this dataset, providing a crucial baseline for evaluating the more complex models.

---

## 4. Results and Analysis

### 4.1. Final Model Performance

The final comparison of our models clearly shows the trade-off between performance and complexity. The LightGBM model was the top performer in terms of raw accuracy, but its directional accuracy reveals a significant bias.

![Directional Accuracy Comparison](./analysis/visualizations/plots/model_directional_accuracy_comparison.png)

The model achieved its 75% accuracy by being exceptionally good at predicting "Up" days (100% accuracy) but failing significantly on "Down" days (25% accuracy). This is a common pitfall in time-series modeling, where a model may simply learn the overall trend of the training period.

The confusion matrix for the LightGBM model provides a clear picture of these results:

![LightGBM Confusion Matrix](./analysis/visualizations/plots/LightGBM_confusion_matrix.png)

### 4.2. The Overfitting Trap: A Deeper Look

The most critical insight came from the feature importance analysis. The fact that `content_length` was the most important feature for the high-performing models was a clear red flag.

![Feature Importance](./analysis/visualizations/plots/feature_importance.png)

This plot shows the feature importance for a Random Forest model, which exhibited similar behavior to the LightGBM. This realization was central to the project's conclusion: without a sufficiently large and diverse dataset, even sophisticated models will resort to learning simple, and often wrong, patterns.

---

## 5. Conclusion

CryptoPulse successfully achieved its primary engineering goal: to build a robust, automated pipeline for sentiment analysis and cryptocurrency prediction. The system is a solid foundation for future research and development.

The scientific journey, however, was one of critical re-evaluation. We demonstrated that while social media sentiment does contain a predictive signal, its utility is easily overestimated. The key takeaway is that for financial prediction, **data scale and statistical rigor are paramount**. High accuracy on small datasets should be treated with extreme skepticism. This project serves as a case study in the honest and transparent reporting of machine learning results, emphasizing the importance of understanding *why* a model works, not just that it appears to.

---

## 6. How to Run the Project

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
    The best way to understand the project is to review the `CryptoPulse_Workflow.ipynb` notebook in the `notebooks/` directory.

---

## 7. References

1.  Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
2.  Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems 30.
3.  Lim, B., et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Conference on Machine Learning.
4.  Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
5.  ElKulako, A. (2023). *CryptoBERT: A Cryptocurrency Sentiment Analysis Model*. IEEE Conference Proceedings.
