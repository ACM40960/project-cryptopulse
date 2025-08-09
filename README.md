# CryptoPulse: A Critical Re-evaluation of Sentiment-Based Financial Prediction

This project is a comprehensive system for cryptocurrency sentiment analysis and price prediction. More importantly, it serves as a critical re-evaluation of the effectiveness of social media sentiment for financial forecasting, especially when dealing with limited and noisy real-world data.

The core of this project is a robust data pipeline that automatically collects, processes, and analyzes data from multiple sources. While the current dataset is small, the system is designed to run continuously, allowing the dataset to grow over time and enabling the training of increasingly accurate and robust models.

## 🚀 Automated Data Pipeline

This project is built as a self-improving system. The data collection scripts in `src/` and the automation setup in `scripts/` are designed to:

1.  **Continuously Collect Data:** Automatically gather new data from Reddit, news sources, and Twitter on a daily basis.
2.  **Process and Score:** New data is automatically processed and scored for sentiment and other engineered features.
3.  **Grow the Dataset:** The machine learning dataset grows larger and more robust every day.
4.  **Improve Models:** As the dataset expands, the predictive models can be retrained to improve their accuracy and generalizability, overcoming the initial limitations of data sparsity.

This automated workflow means that the project is not just a static analysis, but a living system designed for long-term improvement and research.

## Project Structure

```
├── data/          # Raw and processed data (ignored by git)
├── docs/          # Project documentation
├── models/        # Trained models (ignored by git)
├── notebooks/     # Jupyter notebooks, including the main workflow
├── reports/       # Generated reports (ignored by git)
├── scripts/       # Automation scripts (e.g., cron jobs)
├── src/           # All Python source code
├── .gitignore     # Specifies intentionally untracked files to ignore
├── README.md      # This file
└── requirements.txt # Project dependencies
```

## The Core Investigation: Data Sparsity and Model Robustness

Our initial analysis with complex models like LightGBM yielded high but misleading accuracy scores (e.g., 75%). Further investigation revealed that this was a classic case of overfitting on a small dataset (178 daily samples).

The key findings of this project are:
-   **Data Sparsity is the main challenge:** Small datasets fundamentally limit the reliability of complex financial prediction models.
-   **Simple Models Provide Honest Baselines:** A simple Logistic Regression model, while less "accurate", provides a more realistic and robust baseline for performance.
-   **Feature Engineering is Critical:** Identifying and removing spurious features (like `content_length`) is crucial for building reliable models.

The `notebooks/CryptoPulse_Workflow.ipynb` notebook provides a complete walkthrough of this investigation.

## Getting Started

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Explore the entire workflow in `notebooks/CryptoPulse_Workflow.ipynb`.

## Future Work

The primary future work is to let the automated data pipeline run to collect a large and robust dataset (ideally 1000+ daily samples). Once a larger dataset is available, the following steps can be taken:

-   Retrain the complex models (LightGBM, XGBoost) and evaluate their performance on the new data.
-   Implement more advanced validation techniques like walk-forward validation.
-   Expand the system to include more data sources (e.g., on-chain data).
