#!/usr/bin/env python3
"""
Phase 3: CryptoBERT-Enhanced Models

This module implements ultimate cryptocurrency price prediction models using:
- Price data (price_usd)
- Technical indicators (price_ma_7, price_volatility)
- Text-derived sentiment features (12 features)
- CryptoBERT embeddings (768-dimensional contextual embeddings)

Hypothesis: CryptoBERT embeddings provide domain-specific semantic understanding
that significantly improves prediction performance beyond traditional sentiment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
import os
import sqlite3
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

class CryptoBERTModelTrainer:
    def __init__(self, data_path, models_dir, db_path):
        self.data_path = data_path
        self.models_dir = models_dir
        self.db_path = db_path
        self.results = {}
        
        # Create CryptoBERT directory
        self.cryptobert_dir = os.path.join(models_dir, 'cryptobert_phase3')
        os.makedirs(self.cryptobert_dir, exist_ok=True)
        
        # Initialize CryptoBERT model
        print("🤖 Loading CryptoBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('ElKulako/cryptobert')
        self.model = AutoModel.from_pretrained('ElKulako/cryptobert')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"✅ CryptoBERT loaded on {self.device}")
        
    def get_cryptobert_embedding(self, text):
        """Generate CryptoBERT embedding for a single text"""
        if not text or len(text.strip()) == 0:
            return np.zeros(768)  # Return zero vector for empty text
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                  max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embedding.flatten()
        except Exception as e:
            print(f"⚠️ Error processing text: {str(e)[:100]}...")
            return np.zeros(768)
    
    def generate_daily_cryptobert_features(self):
        """Generate CryptoBERT embeddings for each day in the dataset"""
        print("🔗 Generating daily CryptoBERT embeddings...")
        
        # Load ML dataset to get dates
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        daily_embeddings = {}
        
        for idx, row in df.iterrows():
            date = row['date'].strftime('%Y-%m-%d')
            print(f"  📅 Processing {date} ({idx+1}/{len(df)})...")
            
            # Get all text content for this date
            reddit_query = """
            SELECT content FROM reddit_posts 
            WHERE date(created_utc) = ? AND length(content) > 10
            LIMIT 10
            """
            
            twitter_query = """
            SELECT content FROM twitter_posts 
            WHERE date(created_at) = ? AND length(content) > 10
            LIMIT 10
            """
            
            news_query = """
            SELECT content FROM news_articles 
            WHERE date(published_at) = ? AND length(content) > 50
            LIMIT 5
            """
            
            # Collect texts
            all_texts = []
            
            # Reddit texts
            reddit_texts = pd.read_sql_query(reddit_query, conn, params=[date])
            all_texts.extend(reddit_texts['content'].tolist())
            
            # Twitter texts
            twitter_texts = pd.read_sql_query(twitter_query, conn, params=[date])
            all_texts.extend(twitter_texts['content'].tolist())
            
            # News texts
            news_texts = pd.read_sql_query(news_query, conn, params=[date])
            all_texts.extend(news_texts['content'].tolist())
            
            if all_texts:
                # Combine texts (limit total length)
                combined_text = ' '.join(all_texts)[:2000]  # Limit to 2000 chars
                
                # Generate embedding
                embedding = self.get_cryptobert_embedding(combined_text)
                daily_embeddings[date] = embedding
                
                print(f"    ✅ Generated embedding from {len(all_texts)} texts")
            else:
                # Use zero embedding for days with no text
                daily_embeddings[date] = np.zeros(768)
                print(f"    ⚠️ No text found, using zero embedding")
        
        conn.close()
        
        # Convert to DataFrame
        embedding_df = pd.DataFrame.from_dict(daily_embeddings, orient='index')
        embedding_df.columns = [f'cryptobert_{i}' for i in range(768)]
        embedding_df.index.name = 'date'
        embedding_df.reset_index(inplace=True)
        embedding_df['date'] = pd.to_datetime(embedding_df['date'])
        
        # Save embeddings
        embeddings_path = os.path.join(self.cryptobert_dir, 'daily_cryptobert_embeddings.csv')
        embedding_df.to_csv(embeddings_path, index=False)
        
        print(f"💾 Saved embeddings to {embeddings_path}")
        return embedding_df
    
    def reduce_embeddings_with_pca(self, embeddings_df, n_components=50):
        """Reduce CryptoBERT embedding dimensionality with PCA"""
        print(f"🔄 Reducing embeddings from 768 to {n_components} dimensions with PCA...")
        
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('cryptobert_')]
        embedding_matrix = embeddings_df[embedding_cols].values
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embedding_matrix)
        
        # Create DataFrame with reduced embeddings
        reduced_df = embeddings_df[['date']].copy()
        for i in range(n_components):
            reduced_df[f'cryptobert_pca_{i}'] = reduced_embeddings[:, i]
        
        # Save PCA model and reduced embeddings
        pca_path = os.path.join(self.cryptobert_dir, 'cryptobert_pca_model.joblib')
        joblib.dump(pca, pca_path)
        
        reduced_path = os.path.join(self.cryptobert_dir, 'daily_cryptobert_pca_embeddings.csv')
        reduced_df.to_csv(reduced_path, index=False)
        
        print(f"✅ PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"💾 Saved reduced embeddings to {reduced_path}")
        
        return reduced_df, pca
    
    def load_data_with_cryptobert(self):
        """Load dataset and add CryptoBERT features"""
        print("📊 Loading dataset and integrating CryptoBERT features...")
        
        # Load base dataset
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Check if embeddings exist, otherwise generate them
        embeddings_path = os.path.join(self.cryptobert_dir, 'daily_cryptobert_pca_embeddings.csv')
        
        if os.path.exists(embeddings_path):
            print("📂 Loading existing CryptoBERT embeddings...")
            embeddings_df = pd.read_csv(embeddings_path)
            embeddings_df['date'] = pd.to_datetime(embeddings_df['date'])
        else:
            print("🔗 Generating new CryptoBERT embeddings...")
            full_embeddings = self.generate_daily_cryptobert_features()
            embeddings_df, pca = self.reduce_embeddings_with_pca(full_embeddings)
        
        # Merge embeddings with main dataset
        df_with_embeddings = df.merge(embeddings_df, on='date', how='left')
        
        # Define feature groups
        baseline_features = ['price_usd', 'price_ma_7', 'price_volatility']
        text_features = [
            'content_length_max', 'content_length_mean', 'num_comments_sum',
            'volatility_score_reddit', 'volatility_score_mean', 'relevance_score_max',
            'echo_score_mean', 'engagement_sum', 'echo_score_reddit',
            'echo_score_max', 'volatility_score_max', 'engagement_mean'
        ]
        cryptobert_features = [col for col in embeddings_df.columns if col.startswith('cryptobert_pca_')]
        
        all_features = baseline_features + text_features + cryptobert_features
        
        # Remove rows with missing features
        df_clean = df_with_embeddings.dropna(subset=all_features)
        
        print(f"✅ Dataset loaded: {len(df_clean)} samples")
        print(f"🔧 Feature breakdown:")
        print(f"   📊 Baseline features: {len(baseline_features)}")
        print(f"   📝 Text features: {len(text_features)}")
        print(f"   🤖 CryptoBERT features: {len(cryptobert_features)}")
        print(f"   📈 Total features: {len(all_features)}")
        
        # Features (X) - ALL FEATURES INCLUDING CRYPTOBERT
        X = df_clean[all_features].copy()
        
        # Targets (y)
        targets = {
            'direction_1d': df_clean['direction_1d'].copy(),
            'direction_3d': df_clean['direction_3d'].copy(),
            'direction_7d': df_clean['direction_7d'].copy(),
            'price_change_1d': df_clean['price_change_1d'].copy(),
            'price_change_3d': df_clean['price_change_3d'].copy(),
            'price_change_7d': df_clean['price_change_7d'].copy()
        }
        
        # Remove samples with missing targets
        valid_indices = df_clean.dropna(subset=list(targets.keys())).index
        X = X.loc[valid_indices]
        targets = {k: v.loc[valid_indices] for k, v in targets.items()}
        
        print(f"📈 Final dataset: {len(X)} samples with {len(all_features)} features")
        return X, targets, all_features, baseline_features, text_features, cryptobert_features
    
    def train_classification_models(self, X, y, target_name, features):
        """Train classification models"""
        print(f"\n🎯 Training CLASSIFICATION models for {target_name}")
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔄 Training {model_name}...")
            
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            model.fit(X, y)
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            results[model_name] = {
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'train_accuracy': float(accuracy),
                'feature_importance': {k: float(v) for k, v in zip(features, model.feature_importances_)}
            }
            
            model_path = os.path.join(self.cryptobert_dir, f'{model_name}_{target_name}.joblib')
            joblib.dump(model, model_path)
            
            print(f"    ✅ CV Accuracy: {cv_scores.mean():.3f} ±{cv_scores.std():.3f}")
        
        return results
    
    def train_regression_models(self, X, y, target_name, features):
        """Train regression models"""
        print(f"\n📈 Training REGRESSION models for {target_name}")
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔄 Training {model_name}...")
            
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            model.fit(X, y)
            predictions = model.predict(X)
            
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            results[model_name] = {
                'cv_mae_mean': float(-cv_scores.mean()),
                'cv_mae_std': float(cv_scores.std()),
                'train_mae': float(mae),
                'train_r2': float(r2),
                'feature_importance': {k: float(v) for k, v in zip(features, model.feature_importances_)}
            }
            
            model_path = os.path.join(self.cryptobert_dir, f'{model_name}_{target_name}.joblib')
            joblib.dump(model, model_path)
            
            print(f"    ✅ CV MAE: {-cv_scores.mean():.3f} ±{cv_scores.std():.3f}")
        
        return results
    
    def analyze_feature_importance(self, classification_results, regression_results, 
                                 baseline_features, text_features, cryptobert_features):
        """Analyze feature importance across all feature types"""
        print("\n🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Aggregate importance
        all_importances = {}
        model_count = 0
        
        for results_dict in [classification_results, regression_results]:
            for target, models in results_dict.items():
                for model_name, metrics in models.items():
                    model_count += 1
                    for feature, importance in metrics['feature_importance'].items():
                        if feature not in all_importances:
                            all_importances[feature] = 0
                        all_importances[feature] += importance
        
        avg_importances = {k: v/model_count for k, v in all_importances.items()}
        
        # Group importance by feature type
        baseline_importance = sum(avg_importances.get(f, 0) for f in baseline_features)
        text_importance = sum(avg_importances.get(f, 0) for f in text_features)
        cryptobert_importance = sum(avg_importances.get(f, 0) for f in cryptobert_features)
        total_importance = baseline_importance + text_importance + cryptobert_importance
        
        print(f"📊 Feature Group Importance:")
        print(f"   🔧 Baseline: {baseline_importance:.3f} ({baseline_importance/total_importance*100:.1f}%)")
        print(f"   📝 Text: {text_importance:.3f} ({text_importance/total_importance*100:.1f}%)")
        print(f"   🤖 CryptoBERT: {cryptobert_importance:.3f} ({cryptobert_importance/total_importance*100:.1f}%)")
        
        # Top features overall
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            if feature in baseline_features:
                feature_type = "🔧 BASELINE"
            elif feature in text_features:
                feature_type = "📝 TEXT"
            else:
                feature_type = "🤖 CRYPTOBERT"
            print(f"   {i+1:2d}. {feature}: {importance:.3f} {feature_type}")
        
        return {
            'baseline_importance': baseline_importance,
            'text_importance': text_importance,
            'cryptobert_importance': cryptobert_importance,
            'baseline_percent': baseline_importance/total_importance*100,
            'text_percent': text_importance/total_importance*100,
            'cryptobert_percent': cryptobert_importance/total_importance*100,
            'top_features': sorted_features[:15]
        }
    
    def run_cryptobert_training(self):
        """Execute complete CryptoBERT training pipeline"""
        print("🚀 PHASE 3: CRYPTOBERT-ENHANCED MODEL TRAINING")
        print("="*70)
        
        # Load data with CryptoBERT features
        X, targets, all_features, baseline_features, text_features, cryptobert_features = self.load_data_with_cryptobert()
        
        # Train classification models
        classification_results = {}
        for target in ['direction_1d', 'direction_3d', 'direction_7d']:
            classification_results[target] = self.train_classification_models(
                X, targets[target], target, all_features
            )
        
        # Train regression models
        regression_results = {}
        for target in ['price_change_1d', 'price_change_3d', 'price_change_7d']:
            regression_results[target] = self.train_regression_models(
                X, targets[target], target, all_features
            )
        
        # Analyze feature importance
        feature_analysis = self.analyze_feature_importance(
            classification_results, regression_results, 
            baseline_features, text_features, cryptobert_features
        )
        
        # Compile results
        self.results = {
            'phase': 'cryptobert_phase3',
            'description': 'Ultimate models with price + technical + text-derived + CryptoBERT embeddings',
            'features_used': all_features,
            'baseline_features': baseline_features,
            'text_features': text_features,
            'cryptobert_features': cryptobert_features,
            'num_features': len(all_features),
            'num_baseline_features': len(baseline_features),
            'num_text_features': len(text_features),
            'num_cryptobert_features': len(cryptobert_features),
            'dataset_size': len(X),
            'classification_results': classification_results,
            'regression_results': regression_results,
            'feature_analysis': feature_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(self.cryptobert_dir, 'cryptobert_phase3_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*70)
        print("📊 PHASE 3 CRYPTOBERT RESULTS SUMMARY")
        print("="*70)
        
        print(f"🔧 Features: {self.results['num_features']} total")
        print(f"   📊 Baseline: {self.results['num_baseline_features']} features")
        print(f"   📝 Text-derived: {self.results['num_text_features']} features")
        print(f"   🤖 CryptoBERT: {self.results['num_cryptobert_features']} features")
        print(f"📊 Dataset: {self.results['dataset_size']} samples")
        
        print("\n🎯 CLASSIFICATION PERFORMANCE (Direction Prediction):")
        for target, models in self.results['classification_results'].items():
            print(f"\n  {target.upper()}:")
            for model, metrics in models.items():
                print(f"    {model}: {metrics['cv_accuracy_mean']:.3f} ±{metrics['cv_accuracy_std']:.3f} CV accuracy")
        
        print("\n📈 REGRESSION PERFORMANCE (Price Change Prediction):")
        for target, models in self.results['regression_results'].items():
            print(f"\n  {target.upper()}:")
            for model, metrics in models.items():
                print(f"    {model}: R²={metrics['train_r2']:.3f}, MAE={metrics['cv_mae_mean']:.3f}")
        
        fa = self.results['feature_analysis']
        print(f"\n🔍 FEATURE CONTRIBUTION ANALYSIS:")
        print(f"   🔧 Baseline: {fa['baseline_percent']:.1f}%")
        print(f"   📝 Text: {fa['text_percent']:.1f}%")
        print(f"   🤖 CryptoBERT: {fa['cryptobert_percent']:.1f}%")
        print(f"   🏆 Top feature: {fa['top_features'][0][0]} ({fa['top_features'][0][1]:.3f})")
        
        print(f"\n💾 Models saved to: {self.cryptobert_dir}")
        print("="*70)

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/home/thej/Desktop/CryptoPulse/data/simplified_ml_dataset.csv"
    MODELS_DIR = "/home/thej/Desktop/CryptoPulse/models"
    DB_PATH = "/home/thej/Desktop/CryptoPulse/db/cryptopulse.db"
    
    # Initialize and run
    trainer = CryptoBERTModelTrainer(DATA_PATH, MODELS_DIR, DB_PATH)
    results = trainer.run_cryptobert_training()
    
    print("\n🎯 PHASE 3 COMPLETE: Ultimate CryptoBERT-enhanced models!")
    print("📊 Ready for comprehensive evaluation and comparison.")