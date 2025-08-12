# CryptoPulse: Master Briefing Document for Scientific Poster

---

## 1. INSTRUCTIONS FOR AI / DESIGNER

This document contains all the necessary content, figures, and design guidelines to create a scientific poster for the CryptoPulse project. The goal is to produce a professional, academic poster suitable for a university-level mathematical modeling course.

**Key Mandates:**
- The poster must be created using LaTeX.
- The design should be simple, clean, and professional. Avoid overly complex themes that can cause compilation errors.
- The content should be laid out in a clear, logical three-column format.
- All text and figures specified in this document must be included.

---

## 2. POSTER CONTENT

### 2.1. Title, Author, and Affiliation

- **Title:** CryptoPulse: A Critical Analysis of Sentiment-Based Financial Prediction Under Data Sparsity
- **Author:** Thej Ratheesh
- **Affiliation:** University College Dublin, School of Mathematics and Statistics

### 2.2. Abstract

This project presents CryptoPulse, an automated pipeline for cryptocurrency price prediction using social media sentiment. The core of this work is not the pursuit of unattainable accuracy, but a rigorous, critical re-evaluation of the entire methodology. We demonstrate that while sentiment is a predictive feature, its effectiveness is severely constrained by real-world data sparsity, leading to a high risk of model overfitting. By comparing a simple, robust Logistic Regression model with a complex, but overfit, LightGBM model, we reveal the dangers of relying on misleading accuracy metrics. Our key finding is that in financial machine learning with limited data, methodological rigor and the establishment of a robust baseline are more valuable than inflated performance claims. This work presents a complete, automated system ready for long-term data collection and provides a crucial, honest assessment of the challenges and limitations of sentiment-based financial forecasting.

### 2.3. Methodology

Our project follows a four-stage pipeline, designed for automation and continuous improvement.

**1. Data Collection**
A suite of custom scrapers collects data from diverse sources:
- **Reddit:** Posts and comments from relevant subreddits (e.g., r/ethereum) using PRAW.
- **News:** Articles from over 100 crypto news outlets via RSS feeds using `feedparser`.
- **Twitter:** Tweets from crypto influencers using `selenium`.
- **Price Data:** Historical ETH/USD price data from Yahoo Finance.

**2. Feature Engineering**
The raw data is processed to create a feature set for our models:
- **Sentiment Analysis:** Text data is scored using pre-trained NLP models (FinBERT) to determine sentiment.
- **Custom Metrics:** We engineer more nuanced features, including `volatility_score` (sentiment variance) and `echo_score` (story repetition across sources).
- **Aggregation:** All features are aggregated into a daily time-series dataset.

**3. Model Training & Evaluation**
We train and evaluate a suite of models to test our hypothesis:
- **Baseline:** A simple model using only technical price indicators.
- **Simple Model (Logistic Regression):** A robust, interpretable model to serve as an honest baseline.
- **Complex Model (LightGBM):** A powerful gradient boosting model used to demonstrate the effects of overfitting.
- **Evaluation:** Models are evaluated on a held-out test set using Accuracy, Precision, Recall, F1-Score, and, most importantly, **Directional Accuracy**.

### 2.4. Results

Our results demonstrate the critical trade-off between model complexity and robustness on a limited dataset. While the complex model achieved a higher headline accuracy, a deeper analysis reveals it is overfit and unreliable.

**1. Misleading Overall Accuracy**
The LightGBM model produced a seemingly impressive overall accuracy of **75%**, significantly outperforming the simpler Logistic Regression model (**33.3%**).

**2. Unreliable Directional Predictions**
A breakdown of the directional accuracy reveals the LightGBM model's weakness. It achieved **100% accuracy on "Up" days** but only **25% on "Down" days**, indicating a heavy bias towards predicting the majority class.

**3. Spurious Feature Importance**
Analysis of the feature importances for the LightGBM model shows that the most predictive features were `content_length_max` and `content_length_mean` - not true sentiment signals.

### 2.5. Conclusion

The seemingly poor performance of our models is, in fact, the central finding of this project. Our work demonstrates that for financial prediction tasks with noisy and limited data, a simple model that produces a realistic baseline is more valuable than a complex model that overfits to produce misleadingly high accuracy. The primary contribution of this project is not a "black box" that claims to predict the future, but a rigorous and honest methodology for evaluating the true potential of sentiment analysis. We have successfully built an automated pipeline and established a reliable baseline, proving the soundness of our approach and paving the way for future success.

### 2.6. Future Work

The CryptoPulse system is fully automated and designed for continuous operation. The clear path forward is to:
1.  **Run the automated data collection pipeline** over an extended period to build a large and robust dataset (targeting 1,000+ daily samples).
2.  **Retrain the current models** on the new, larger dataset.
3.  **Explore Advanced Model Architectures:** Once a substantial dataset is available, we will leverage more sophisticated time-series models:
    -   **Statistical Baselines:** ARIMA/SARIMA to establish a strong, traditional econometric baseline.
    -   **Recurrent Neural Networks:** LSTMs and GRUs to capture temporal dependencies.
    -   **State-of-the-Art Transformers:** Temporal Fusion Transformers (TFT) for their high performance on complex time-series data.
    -   **Hybrid Models (ARIMA+LSTM):** To combine the strengths of statistical and deep learning approaches.
4.  **Implement Enhanced Validation:** As the dataset grows, we will implement more rigorous validation techniques, such as walk-forward validation.

---

## 3. VISUAL ELEMENTS

### 3.1. Required Figures

The poster must include the following four figures, laid out in a logical sequence to tell the project's story.

1.  **Figure 1: Overall Model Performance**
    -   **File:** `analysis/visualizations/plots/model_comparison.png`
    -   **Caption:** Comparison of key metrics for the simple (Logistic Regression) and complex (LightGBM) models.

2.  **Figure 2: Directional Accuracy Breakdown**
    -   **File:** `analysis/visualizations/plots/directional_accuracy_comparison.png`
    -   **Caption:** Directional accuracy on Up vs. Down days, revealing the bias of the LightGBM model.

3.  **Figure 3: Confusion Matrix**
    -   **File:** `analysis/visualizations/plots/confusion_matrix.png`
    -   **Caption:** The confusion matrix for the LightGBM model, providing clear, quantitative evidence of its predictive bias.

4.  **Figure 4: Feature Importance**
    -   **File:** `analysis/visualizations/plots/feature_importance.png`
    -   **Caption:** Feature importance for the LightGBM model, highlighting its reliance on spurious features like `content_length`.

### 3.2. Workflow Diagram

The poster should also include a workflow diagram in the Methodology section. The diagram should be generated using TikZ in LaTeX for a professional look and should illustrate the four main stages: Data Collection, Feature Engineering, Model Training, and Analysis.

---

## 4. DESIGN GUIDELINES

-   **Tool:** LaTeX
-   **Theme:** Use the standard, default `beamer` theme. Do not use complex external themes like `metropolis` to ensure high compatibility and avoid errors.
-   **Layout:** A clean, three-column layout.
-   **Color Scheme:** The background should be a light, neutral color (e.g., a very light grey or off-white) to look less plain than the default white. Section headers should use a single, professional accent color (e.g., a dark blue).
-   **Typography:** Use standard LaTeX fonts. Ensure font sizes are large enough to be readable on a poster.
