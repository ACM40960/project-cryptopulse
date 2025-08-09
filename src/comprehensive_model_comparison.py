#!/usr/bin/env python3
# src/comprehensive_model_comparison.py

"""
Comprehensive Model Comparison for CryptoPulse
Compares baseline, sentiment-enhanced, and advanced deep learning models
to provide complete performance analysis and recommendations.
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_comparison.log'),
        logging.StreamHandler()
    ]
)

class ComprehensiveModelComparison:
    def __init__(self):
        self.baseline_results = None
        self.sentiment_results = None
        self.advanced_results = None
        
    def load_all_results(self):
        """Load results from all model categories."""
        logging.info("📊 Loading comprehensive model results...")
        
        success = True
        
        try:
            # Load baseline results
            with open('models/baseline/baseline_results_summary_direction_1d.json', 'r') as f:
                self.baseline_results = json.load(f)
            logging.info("   ✅ Loaded baseline results")
        except FileNotFoundError:
            logging.error("   ❌ Baseline results not found")
            success = False
        
        try:
            # Load sentiment-enhanced results
            with open('models/results_summary_direction_1d.json', 'r') as f:
                self.sentiment_results = json.load(f)
            logging.info("   ✅ Loaded sentiment-enhanced results")
        except FileNotFoundError:
            logging.error("   ❌ Sentiment-enhanced results not found")
            success = False
        
        try:
            # Load advanced results
            with open('models/advanced/advanced_results_summary_direction_1d.json', 'r') as f:
                self.advanced_results = json.load(f)
            logging.info("   ✅ Loaded advanced deep learning results")
        except FileNotFoundError:
            logging.error("   ❌ Advanced results not found")
            success = False
        
        return success
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison table."""
        logging.info("\\n🔍 COMPREHENSIVE MODEL COMPARISON")
        logging.info("=" * 100)
        
        comparison_data = []
        
        # Process baseline models
        if self.baseline_results:
            for model_name, results in self.baseline_results.items():
                comparison_data.append({
                    'Category': 'Baseline',
                    'Algorithm': model_name.replace('Baseline_', ''),
                    'Features': 'Price + Technical (15 features)',
                    'Feature_Count': results.get('feature_count', 15),
                    'Test_Accuracy': results['test_accuracy'],
                    'CV_Mean': results.get('cv_mean', 0),
                    'CV_Std': results.get('cv_std', 0),
                    'Up_Days_Acc': results['up_accuracy'],
                    'Down_Days_Acc': results['down_accuracy'],
                    'F1_Score': results['f1_score'],
                    'Model_Type': 'Traditional ML'
                })
        
        # Process sentiment-enhanced models
        if self.sentiment_results:
            for model_name, results in self.sentiment_results.items():
                comparison_data.append({
                    'Category': 'Sentiment-Enhanced',
                    'Algorithm': model_name,
                    'Features': 'Sentiment + Price + Technical (12 features)',
                    'Feature_Count': 12,
                    'Test_Accuracy': results['test_accuracy'],
                    'CV_Mean': results.get('cv_mean', 0),
                    'CV_Std': results.get('cv_std', 0),
                    'Up_Days_Acc': results['up_accuracy'],
                    'Down_Days_Acc': results['down_accuracy'],
                    'F1_Score': results['f1_score'],
                    'Model_Type': 'Traditional ML + Sentiment'
                })
        
        # Process advanced models
        if self.advanced_results:
            for model_name, results in self.advanced_results.items():
                comparison_data.append({
                    'Category': 'Advanced Deep Learning',
                    'Algorithm': model_name,
                    'Features': 'Sequential Sentiment + Price (12 features)',
                    'Feature_Count': 12,
                    'Test_Accuracy': results['test_accuracy'],
                    'CV_Mean': 0,  # Not available for deep learning models
                    'CV_Std': 0,
                    'Up_Days_Acc': results['up_accuracy'],
                    'Down_Days_Acc': results['down_accuracy'],
                    'F1_Score': results['f1_score'],
                    'Model_Type': 'Deep Learning'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by test accuracy
        comparison_df = comparison_df.sort_values(['Test_Accuracy', 'F1_Score'], ascending=[False, False])
        
        logging.info("\\nComplete Model Performance Comparison:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        logging.info(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df
    
    def analyze_category_performance(self, comparison_df):
        """Analyze performance by model category."""
        logging.info("\\n📈 CATEGORY PERFORMANCE ANALYSIS")
        logging.info("=" * 80)
        
        # Group by category and get best performers
        category_analysis = []
        
        for category in comparison_df['Category'].unique():
            category_data = comparison_df[comparison_df['Category'] == category]
            best_model = category_data.iloc[0]
            
            category_analysis.append({
                'Category': category,
                'Best_Algorithm': best_model['Algorithm'],
                'Best_Accuracy': best_model['Test_Accuracy'],
                'Avg_Accuracy': category_data['Test_Accuracy'].mean(),
                'Accuracy_Std': category_data['Test_Accuracy'].std(),
                'Best_F1': best_model['F1_Score'],
                'Avg_F1': category_data['F1_Score'].mean(),
                'Model_Count': len(category_data)
            })
        
        category_df = pd.DataFrame(category_analysis).sort_values('Best_Accuracy', ascending=False)
        
        logging.info("\\nCategory Performance Summary:")
        logging.info(category_df.to_string(index=False, float_format='%.3f'))
        
        # Calculate improvements
        if len(category_df) >= 2:
            best_category = category_df.iloc[0]
            baseline_category = category_df[category_df['Category'] == 'Baseline']
            
            if len(baseline_category) > 0:
                baseline_acc = baseline_category.iloc[0]['Best_Accuracy']
                best_acc = best_category['Best_Accuracy']
                improvement = best_acc - baseline_acc
                
                logging.info(f"\\n🎯 OVERALL IMPROVEMENT ANALYSIS:")
                logging.info(f"   Best Overall Model: {best_category['Best_Algorithm']} ({best_category['Category']})")
                logging.info(f"   Best Accuracy: {best_acc:.3f}")
                logging.info(f"   Baseline Accuracy: {baseline_acc:.3f}")
                logging.info(f"   Absolute Improvement: +{improvement:.3f}")
                logging.info(f"   Relative Improvement: +{improvement/baseline_acc*100:.1f}%")
        
        return category_df
    
    def feature_importance_comparison(self):
        """Compare feature importance across different model types."""
        logging.info("\\n📋 FEATURE IMPORTANCE COMPARISON")
        logging.info("=" * 80)
        
        try:
            # Load feature importance from best models in each category
            baseline_features = pd.read_csv('models/baseline/Baseline_LightGBM_direction_1d_features.csv')
            sentiment_features = pd.read_csv('models/LightGBM_direction_1d_features.csv')
            
            logging.info("\\nTop 5 Baseline Features (Price/Technical Only):")
            for idx, row in baseline_features.head(5).iterrows():
                logging.info(f"   {row['feature']}: {row['importance']:.4f}")
            
            logging.info("\\nTop 5 Sentiment-Enhanced Features:")
            for idx, row in sentiment_features.head(5).iterrows():
                feature_type = self.classify_feature_type(row['feature'])
                logging.info(f"   {row['feature']} ({feature_type}): {row['importance']:.4f}")
            
            # Count feature types in sentiment model
            sentiment_feature_types = {'Sentiment': 0, 'Content': 0, 'Engagement': 0, 'Other': 0}
            
            for _, row in sentiment_features.iterrows():
                feature_type = self.classify_feature_type(row['feature'])
                if feature_type in sentiment_feature_types:
                    sentiment_feature_types[feature_type] += 1
            
            logging.info("\\nFeature Type Distribution in Best Sentiment Model:")
            for feature_type, count in sentiment_feature_types.items():
                logging.info(f"   {feature_type}: {count} features")
        
        except FileNotFoundError:
            logging.warning("Feature importance files not found")
    
    def classify_feature_type(self, feature_name):
        """Classify feature type based on name."""
        feature_lower = feature_name.lower()
        
        if any(x in feature_lower for x in ['relevance', 'volatility_score', 'echo']):
            return 'Sentiment'
        elif any(x in feature_lower for x in ['content_length', 'num_comments']):
            return 'Content'
        elif any(x in feature_lower for x in ['engagement']):
            return 'Engagement'
        else:
            return 'Other'
    
    def model_complexity_analysis(self, comparison_df):
        """Analyze model complexity vs performance trade-offs."""
        logging.info("\\n⚖️  MODEL COMPLEXITY vs PERFORMANCE ANALYSIS")
        logging.info("=" * 80)
        
        # Define complexity scores
        complexity_map = {
            'RandomForest': 2,
            'LightGBM': 2,
            'XGBoost': 2,
            'LSTM': 4,
            'Prophet': 3
        }
        
        # Add complexity scores
        comparison_df['Complexity'] = comparison_df['Algorithm'].map(complexity_map)
        
        # Calculate efficiency (accuracy per complexity unit)
        comparison_df['Efficiency'] = comparison_df['Test_Accuracy'] / comparison_df['Complexity']
        
        # Get top performers by efficiency
        efficiency_ranking = comparison_df.nlargest(5, 'Efficiency')[['Algorithm', 'Category', 'Test_Accuracy', 'Complexity', 'Efficiency']]
        
        logging.info("\\nTop 5 Most Efficient Models (Accuracy/Complexity):")
        logging.info(efficiency_ranking.to_string(index=False, float_format='%.3f'))
        
        # Complexity vs accuracy analysis
        simple_models = comparison_df[comparison_df['Complexity'] <= 2]
        complex_models = comparison_df[comparison_df['Complexity'] > 2]
        
        if len(simple_models) > 0 and len(complex_models) > 0:
            simple_avg = simple_models['Test_Accuracy'].mean()
            complex_avg = complex_models['Test_Accuracy'].mean()
            
            logging.info(f"\\nComplexity Analysis:")
            logging.info(f"   Simple Models (Complexity ≤ 2): {simple_avg:.3f} average accuracy")
            logging.info(f"   Complex Models (Complexity > 2): {complex_avg:.3f} average accuracy")
            logging.info(f"   Complexity Premium: {complex_avg - simple_avg:+.3f} accuracy points")
    
    def generate_recommendations(self, comparison_df, category_df):
        """Generate model recommendations based on analysis."""
        logging.info("\\n🎯 MODEL RECOMMENDATIONS")
        logging.info("=" * 80)
        
        # Best overall model
        best_overall = comparison_df.iloc[0]
        
        # Best in each category
        baseline_models = comparison_df[comparison_df['Category'] == 'Baseline']
        sentiment_models = comparison_df[comparison_df['Category'] == 'Sentiment-Enhanced']
        advanced_models = comparison_df[comparison_df['Category'] == 'Advanced Deep Learning']
        
        best_baseline = baseline_models.iloc[0] if len(baseline_models) > 0 else None
        best_sentiment = sentiment_models.iloc[0] if len(sentiment_models) > 0 else None
        best_advanced = advanced_models.iloc[0] if len(advanced_models) > 0 else None
        
        logging.info("\\n🏆 RECOMMENDED MODELS BY USE CASE:")
        
        logging.info("\\n1. PRODUCTION DEPLOYMENT (Best Overall Performance):")
        logging.info(f"   Model: {best_overall['Algorithm']} ({best_overall['Category']})")
        logging.info(f"   Accuracy: {best_overall['Test_Accuracy']:.3f}")
        logging.info(f"   F1-Score: {best_overall['F1_Score']:.3f}")
        logging.info(f"   Features: {best_overall['Features']}")
        
        if best_sentiment is not None:
            logging.info("\\n2. SENTIMENT-DRIVEN TRADING (Best Sentiment Model):")
            logging.info(f"   Model: {best_sentiment['Algorithm']} ({best_sentiment['Category']})")
            logging.info(f"   Accuracy: {best_sentiment['Test_Accuracy']:.3f}")
            logging.info(f"   F1-Score: {best_sentiment['F1_Score']:.3f}")
            logging.info(f"   Advantage: Incorporates social sentiment signals")
        
        if best_baseline is not None:
            logging.info("\\n3. SIMPLE/FAST DEPLOYMENT (Best Baseline):")
            logging.info(f"   Model: {best_baseline['Algorithm']} ({best_baseline['Category']})")
            logging.info(f"   Accuracy: {best_baseline['Test_Accuracy']:.3f}")
            logging.info(f"   F1-Score: {best_baseline['F1_Score']:.3f}")
            logging.info(f"   Advantage: Minimal feature requirements, fast inference")
        
        # Model selection criteria
        logging.info("\\n📋 MODEL SELECTION CRITERIA:")
        logging.info("   • High Accuracy Required: Choose Sentiment-Enhanced LightGBM")
        logging.info("   • Fast Inference Required: Choose Baseline models")
        logging.info("   • Interpretability Required: Choose RandomForest models")
        logging.info("   • Sequential Patterns Important: Choose LSTM")
        logging.info("   • Resource Constrained: Choose simplest baseline model")
    
    def save_comprehensive_report(self, comparison_df, category_df):
        """Save comprehensive comparison report."""
        report = {
            'comparison_date': datetime.now().isoformat(),
            'summary': {
                'total_models_compared': len(comparison_df),
                'categories': list(comparison_df['Category'].unique()),
                'best_overall_model': comparison_df.iloc[0]['Algorithm'],
                'best_overall_category': comparison_df.iloc[0]['Category'],
                'best_overall_accuracy': comparison_df.iloc[0]['Test_Accuracy'],
                'accuracy_range': {
                    'min': comparison_df['Test_Accuracy'].min(),
                    'max': comparison_df['Test_Accuracy'].max(),
                    'mean': comparison_df['Test_Accuracy'].mean(),
                    'std': comparison_df['Test_Accuracy'].std()
                }
            },
            'detailed_comparison': comparison_df.to_dict('records'),
            'category_analysis': category_df.to_dict('records')
        }
        
        with open('models/comprehensive_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info("\\n💾 Comprehensive comparison report saved to models/comprehensive_comparison_report.json")

def main():
    """Main comprehensive comparison function."""
    # Initialize comparison
    comparator = ComprehensiveModelComparison()
    
    # Load all results
    if not comparator.load_all_results():
        logging.error("Cannot proceed without all model results")
        return
    
    # Create comprehensive comparison
    comparison_df = comparator.create_comprehensive_comparison()
    
    # Analyze category performance
    category_df = comparator.analyze_category_performance(comparison_df)
    
    # Feature importance comparison
    comparator.feature_importance_comparison()
    
    # Model complexity analysis
    comparator.model_complexity_analysis(comparison_df)
    
    # Generate recommendations
    comparator.generate_recommendations(comparison_df, category_df)
    
    # Save comprehensive report
    comparator.save_comprehensive_report(comparison_df, category_df)
    
    logging.info("\\n🎉 Comprehensive model comparison complete!")

if __name__ == "__main__":
    main()