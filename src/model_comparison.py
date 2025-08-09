#!/usr/bin/env python3
# src/model_comparison.py

"""
Model Comparison for CryptoPulse
Compares baseline models (price only) vs sentiment-enhanced models
to demonstrate the value of our advanced text metrics.
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_comparison.log'),
        logging.StreamHandler()
    ]
)

class ModelComparison:
    def __init__(self):
        self.baseline_results = None
        self.sentiment_results = None
        
    def load_results(self):
        """Load results from both baseline and sentiment models."""
        logging.info("📊 Loading model results for comparison...")
        
        try:
            # Load baseline results
            with open('models/baseline/baseline_results_summary_direction_1d.json', 'r') as f:
                self.baseline_results = json.load(f)
            logging.info("   ✅ Loaded baseline results")
        except FileNotFoundError:
            logging.error("   ❌ Baseline results not found. Run baseline training first.")
            return False
        
        try:
            # Load sentiment-enhanced results
            with open('models/results_summary_direction_1d.json', 'r') as f:
                self.sentiment_results = json.load(f)
            logging.info("   ✅ Loaded sentiment-enhanced results")
        except FileNotFoundError:
            logging.error("   ❌ Sentiment-enhanced results not found. Run sentiment training first.")
            return False
        
        return True
    
    def create_comparison_table(self):
        """Create comprehensive comparison table."""
        logging.info("\n🔍 BASELINE vs SENTIMENT-ENHANCED MODEL COMPARISON")
        logging.info("=" * 80)
        
        comparison_data = []
        
        # Process baseline models
        for model_name, results in self.baseline_results.items():
            comparison_data.append({
                'Model_Type': 'Baseline',
                'Algorithm': model_name.replace('Baseline_', ''),
                'Features': 'Price + Technical Only',
                'Feature_Count': results.get('feature_count', 30),
                'Test_Accuracy': results['test_accuracy'],
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'Up_Days_Acc': results['up_accuracy'],
                'Down_Days_Acc': results['down_accuracy'],
                'F1_Score': results['f1_score']
            })
        
        # Process sentiment-enhanced models
        for model_name, results in self.sentiment_results.items():
            comparison_data.append({
                'Model_Type': 'Sentiment-Enhanced',
                'Algorithm': model_name,
                'Features': 'Sentiment + Price + Technical',
                'Feature_Count': 12,  # From our sentiment dataset
                'Test_Accuracy': results['test_accuracy'],
                'CV_Mean': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'Up_Days_Acc': results['up_accuracy'],
                'Down_Days_Acc': results['down_accuracy'],
                'F1_Score': results['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by test accuracy
        comparison_df = comparison_df.sort_values(['Test_Accuracy', 'CV_Mean'], ascending=[False, False])
        
        logging.info("\nComplete Model Comparison:")
        logging.info(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df
    
    def analyze_performance_gains(self, comparison_df):
        """Analyze performance improvements from sentiment features."""
        logging.info("\n📈 PERFORMANCE ANALYSIS")
        logging.info("=" * 50)
        
        # Get best models from each category
        baseline_models = comparison_df[comparison_df['Model_Type'] == 'Baseline']
        sentiment_models = comparison_df[comparison_df['Model_Type'] == 'Sentiment-Enhanced']
        
        if len(baseline_models) == 0 or len(sentiment_models) == 0:
            logging.error("Missing baseline or sentiment models for comparison")
            return
        
        best_baseline = baseline_models.iloc[0]
        best_sentiment = sentiment_models.iloc[0]
        
        # Calculate improvements
        accuracy_improvement = best_sentiment['Test_Accuracy'] - best_baseline['Test_Accuracy']
        cv_improvement = best_sentiment['CV_Mean'] - best_baseline['CV_Mean']
        f1_improvement = best_sentiment['F1_Score'] - best_baseline['F1_Score']
        
        logging.info(f"Best Baseline Model: {best_baseline['Algorithm']}")
        logging.info(f"   Test Accuracy: {best_baseline['Test_Accuracy']:.3f}")
        logging.info(f"   CV Score: {best_baseline['CV_Mean']:.3f} ± {best_baseline['CV_Std']:.3f}")
        logging.info(f"   Features: {best_baseline['Feature_Count']} ({best_baseline['Features']})")
        
        logging.info(f"\nBest Sentiment-Enhanced Model: {best_sentiment['Algorithm']}")
        logging.info(f"   Test Accuracy: {best_sentiment['Test_Accuracy']:.3f}")
        logging.info(f"   CV Score: {best_sentiment['CV_Mean']:.3f} ± {best_sentiment['CV_Std']:.3f}")
        logging.info(f"   Features: {best_sentiment['Feature_Count']} ({best_sentiment['Features']})")
        
        logging.info(f"\n🎯 IMPROVEMENT FROM SENTIMENT FEATURES:")
        logging.info(f"   Accuracy Gain: {accuracy_improvement:+.3f} ({accuracy_improvement/best_baseline['Test_Accuracy']*100:+.1f}%)")
        logging.info(f"   CV Score Gain: {cv_improvement:+.3f}")
        logging.info(f"   F1-Score Gain: {f1_improvement:+.3f}")
        
        # Determine if sentiment features provide value
        if accuracy_improvement > 0.02:  # 2% improvement threshold
            logging.info("\n✅ CONCLUSION: Sentiment features provide SIGNIFICANT improvement!")
        elif accuracy_improvement > 0:
            logging.info("\n📊 CONCLUSION: Sentiment features provide modest improvement")
        else:
            logging.info("\n⚠️  CONCLUSION: Sentiment features do not improve performance")
        
        return {
            'best_baseline': best_baseline.to_dict(),
            'best_sentiment': best_sentiment.to_dict(),
            'improvements': {
                'accuracy': accuracy_improvement,
                'cv_mean': cv_improvement,
                'f1_score': f1_improvement
            }
        }
    
    def algorithm_comparison(self, comparison_df):
        """Compare how different algorithms perform with different feature sets."""
        logging.info("\n🔬 ALGORITHM-SPECIFIC ANALYSIS")
        logging.info("=" * 50)
        
        algorithms = ['RandomForest', 'LightGBM', 'XGBoost']
        
        for algorithm in algorithms:
            baseline_row = comparison_df[
                (comparison_df['Algorithm'] == algorithm) & 
                (comparison_df['Model_Type'] == 'Baseline')
            ]
            sentiment_row = comparison_df[
                (comparison_df['Algorithm'] == algorithm) & 
                (comparison_df['Model_Type'] == 'Sentiment-Enhanced')
            ]
            
            if len(baseline_row) > 0 and len(sentiment_row) > 0:
                baseline_acc = baseline_row.iloc[0]['Test_Accuracy']
                sentiment_acc = sentiment_row.iloc[0]['Test_Accuracy']
                improvement = sentiment_acc - baseline_acc
                
                logging.info(f"\n{algorithm}:")
                logging.info(f"   Baseline: {baseline_acc:.3f}")
                logging.info(f"   Sentiment: {sentiment_acc:.3f}")
                logging.info(f"   Improvement: {improvement:+.3f} ({improvement/baseline_acc*100:+.1f}%)")
    
    def feature_importance_analysis(self):
        """Analyze which types of features are most important."""
        logging.info("\n📋 FEATURE IMPORTANCE ANALYSIS")
        logging.info("=" * 50)
        
        try:
            # Load feature importance from best models
            baseline_features = pd.read_csv('models/baseline/Baseline_RandomForest_direction_1d_features.csv')
            sentiment_features = pd.read_csv('models/LightGBM_direction_1d_features.csv')
            
            logging.info("Top Baseline Features (Price/Technical):")
            for idx, row in baseline_features.head(5).iterrows():
                logging.info(f"   {row['feature']}: {row['importance']:.4f}")
            
            logging.info("\nTop Sentiment-Enhanced Features:")
            for idx, row in sentiment_features.head(5).iterrows():
                feature_type = "Sentiment" if any(x in row['feature'].lower() for x in ['relevance', 'volatility', 'echo', 'content']) else "Other"
                logging.info(f"   {row['feature']} ({feature_type}): {row['importance']:.4f}")
            
        except FileNotFoundError:
            logging.warning("Feature importance files not found")
    
    def save_comparison_report(self, comparison_df, analysis_results):
        """Save detailed comparison report."""
        report = {
            'comparison_date': datetime.now().isoformat(),
            'summary': {
                'baseline_best_model': analysis_results['best_baseline']['Algorithm'],
                'baseline_best_accuracy': analysis_results['best_baseline']['Test_Accuracy'],
                'sentiment_best_model': analysis_results['best_sentiment']['Algorithm'],
                'sentiment_best_accuracy': analysis_results['best_sentiment']['Test_Accuracy'],
                'accuracy_improvement': analysis_results['improvements']['accuracy'],
                'cv_improvement': analysis_results['improvements']['cv_mean'],
                'f1_improvement': analysis_results['improvements']['f1_score']
            },
            'detailed_comparison': comparison_df.to_dict('records')
        }
        
        with open('models/model_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"\n💾 Comparison report saved to models/model_comparison_report.json")

def main():
    """Main comparison function."""
    # Initialize comparison
    comparator = ModelComparison()
    
    # Load results
    if not comparator.load_results():
        logging.error("Cannot proceed without both baseline and sentiment results")
        return
    
    # Create comparison table
    comparison_df = comparator.create_comparison_table()
    
    # Analyze performance gains
    analysis_results = comparator.analyze_performance_gains(comparison_df)
    
    # Algorithm-specific comparison
    comparator.algorithm_comparison(comparison_df)
    
    # Feature importance analysis
    comparator.feature_importance_analysis()
    
    # Save report
    if analysis_results:
        comparator.save_comparison_report(comparison_df, analysis_results)
    
    logging.info(f"\n🎉 Model comparison complete!")

if __name__ == "__main__":
    main()