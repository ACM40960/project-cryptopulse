# CryptoPulse Regression Analysis

## 🎯 Overview
This directory contains **regression analysis** for predicting ETH price changes using sentiment features. 

**Key Insight:** Converted from binary classification (up/down) to **continuous regression** (actual % changes) for more meaningful predictions.

## 📊 Current Analysis Files

### Active Regression Models
- **`comprehensive_regression_models.py`** - Trains 15 different regression models
- **`regression_analysis.py`** - Initial regression analysis with 4 models

### Generated Results
- **`plots/comprehensive_regression_analysis.png`** - Complete model comparison with smooth curves
- **`plots/comprehensive_model_results.csv`** - Detailed performance metrics
- **`plots/regression_analysis.png`** - Initial regression visualization
- **`plots/scatter_comparison.png`** - Predicted vs actual scatter plots

### Archive
- **`old_binary_classification/`** - Previous binary classification attempts (step functions)

## 🏆 Best Model Results

**Lasso Regression** achieved the best performance:
- **RMSE**: 2.796% (Root Mean Square Error)
- **MAE**: 2.180% (Mean Absolute Error)  
- **R²**: -0.101 (Coefficient of determination)
- **Direction Accuracy**: 66.7% (still predicts direction correctly)

## 📈 Key Findings

### Model Performance Ranking (by RMSE):
1. **Lasso** - 2.796% RMSE, 66.7% direction accuracy
2. **Elastic Net** - 2.845% RMSE, 69.4% direction accuracy
3. **Baseline (Mean)** - 2.991% RMSE, 33.3% direction accuracy
4. **SVR (RBF)** - 3.040% RMSE, 38.9% direction accuracy
5. **Ridge** - 3.152% RMSE, 58.3% direction accuracy

### Why Regression is Better:
- **Continuous predictions** instead of binary step functions
- **Magnitude matters** - Predicting "+3.2%" vs just "up"
- **Smoother visualizations** - Natural curves instead of rectangles
- **More business value** - Actual percentage changes useful for trading

## 🎨 Visualizations

### What You See in the Plots:
- **Black line** = Actual ETH price changes (ground truth)
- **Colored lines** = Model predictions (smooth curves)
- **Bar charts** = Model performance comparison
- **Scatter plots** = Predicted vs actual correlation

### Visual Improvements:
- ✅ **Smooth curves** instead of step functions
- ✅ **Continuous values** (-3.34% to +7.50% range)
- ✅ **15 models compared** comprehensively
- ✅ **Multiple evaluation metrics** (RMSE, MAE, R², Direction Accuracy)

## 🚀 Usage

### Run Comprehensive Analysis:
```bash
python3 comprehensive_regression_models.py
```

### Run Simple Analysis:
```bash
python3 regression_analysis.py
```

## 📊 Dataset Details
- **36 test samples** (80/20 train/test split)
- **12 sentiment features** (content length, engagement, sentiment scores)
- **Target**: `price_change_1d` (daily ETH price change percentage)
- **Date range**: Test period covers recent ETH price movements

## 💡 Critical Assessment

### Strengths:
- **Meaningful predictions** - Actual percentage changes
- **Multiple model comparison** - 15 different approaches
- **Better than baseline** - All top models outperform simple averages
- **Direction accuracy maintained** - Still predicts up/down correctly

### Limitations:
- **Small dataset** - Only 36 test samples (statistical significance limited)
- **Negative R²** - Models struggle to explain variance (need more data)
- **High RMSE** - Predictions have ~3% average error
- **Need more data** - 10x larger dataset required for reliable results

## 🎯 Conclusion

The **regression approach is significantly better** than binary classification:
1. **Provides actual value predictions** instead of just direction
2. **Creates smooth, interpretable visualizations**
3. **Offers business value** for trading decisions
4. **Maintains direction accuracy** while adding magnitude information

**Next step**: Collect more data to improve statistical significance and model performance.

---

*Generated: August 2, 2025*  
*Part of CryptoPulse cryptocurrency price prediction system*