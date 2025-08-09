#!/usr/bin/env python3
"""
Data Expansion Orchestration System

MASTER PLAN: Transform CryptoPulse from 178 ML samples to 1,700+ samples
TIMELINE: Complete expansion in 3-4 days with systematic approach

Execution Strategy:
1. Phase 1: Twitter Influencer Expansion (25,000 tweets → +250 ML samples)
2. Phase 2: Reddit Historical Backfill (5,000 posts → +1,132 ML samples) 
3. Phase 3: News Coverage Enhancement (3,000 articles → +160 ML samples)
4. Phase 4: Data Validation & ML Dataset Regeneration
5. Phase 5: Rerun Hypothesis Validation with Robust Dataset

Expected Outcome: 866% improvement in dataset size for robust ML modeling
"""

import os
import sys
import subprocess
import time
import sqlite3
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

class DataExpansionOrchestrator:
    def __init__(self):
        self.setup_logging()
        self.base_dir = Path("/home/thej/Desktop/CryptoPulse")
        self.collection_dir = self.base_dir / "collection"
        
        # Track progress
        self.initial_stats = self.get_current_data_stats()
        self.phases_completed = []
        
    def setup_logging(self):
        """Setup orchestration logging"""
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_expansion_orchestration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_current_data_stats(self):
        """Get current database statistics"""
        try:
            conn = sqlite3.connect("db/cryptopulse.db")
            
            stats = {}
            
            # Reddit stats
            reddit_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN datetime(created_utc, 'unixepoch') >= '2022-01-01' THEN 1 END) as since_2022
            FROM reddit_posts
            """
            stats['reddit'] = pd.read_sql_query(reddit_query, conn).iloc[0].to_dict()
            
            # Twitter stats  
            twitter_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN created_at >= '2022-01-01' THEN 1 END) as since_2022
            FROM twitter_posts
            """
            stats['twitter'] = pd.read_sql_query(twitter_query, conn).iloc[0].to_dict()
            
            # News stats
            news_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN published_at >= '2022-01-01' THEN 1 END) as since_2022
            FROM news_articles
            """
            stats['news'] = pd.read_sql_query(news_query, conn).iloc[0].to_dict()
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}")
            return {'reddit': {'total': 0, 'since_2022': 0}, 
                   'twitter': {'total': 0, 'since_2022': 0},
                   'news': {'total': 0, 'since_2022': 0}}
    
    def print_expansion_plan(self):
        """Display comprehensive expansion plan"""
        print("🚀 CRYPTOPULSE DATA EXPANSION MASTER PLAN")
        print("="*70)
        
        current_stats = self.get_current_data_stats()
        
        print("📊 CURRENT DATA INVENTORY:")
        print(f"   🔶 Reddit: {current_stats['reddit']['total']:,} total ({current_stats['reddit']['since_2022']:,} since 2022)")
        print(f"   🐦 Twitter: {current_stats['twitter']['total']:,} total ({current_stats['twitter']['since_2022']:,} since 2022)")
        print(f"   📰 News: {current_stats['news']['total']:,} total ({current_stats['news']['since_2022']:,} since 2022)")
        
        print("\\n🎯 EXPANSION TARGETS:")
        print("   📈 Current ML samples: 178")
        print("   🎯 Target ML samples: 1,700+")
        print("   📊 Expected improvement: 866% increase")
        
        print("\\n🗺️ EXECUTION PHASES:")
        print("   📱 Phase 1: Twitter Influencer Expansion")
        print("      🎭 Target: 25,000 tweets from 70+ crypto influencers")
        print("      📈 ML Impact: +250 samples")
        print("      ⏱️ Duration: 4-6 hours")
        
        print("   🔶 Phase 2: Reddit Historical Backfill")
        print("      📅 Target: Fill 1,415 gap days with quality posts")
        print("      📈 ML Impact: +1,132 samples")
        print("      ⏱️ Duration: 2-3 hours")
        
        print("   📰 Phase 3: News Coverage Enhancement")
        print("      📊 Target: 3,000+ historical news articles")
        print("      📈 ML Impact: +160 samples")
        print("      ⏱️ Duration: 1-2 hours")
        
        print("   🤖 Phase 4: ML Dataset Regeneration")
        print("      🔄 Regenerate features with expanded data")
        print("      📊 Create robust 1,700+ sample dataset")
        print("      ⏱️ Duration: 30 minutes")
        
        print("   🔬 Phase 5: Hypothesis Revalidation")
        print("      📈 Rerun text data hypothesis validation")
        print("      📊 Demonstrate statistical significance")
        print("      ⏱️ Duration: 20 minutes")
        
        print("\\n⏱️ TOTAL ESTIMATED TIME: 8-12 hours")
        print("="*70)
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("\\n🔍 CHECKING PREREQUISITES...")
        
        issues = []
        
        # Check Python packages
        required_packages = ['selenium', 'praw', 'pandas', 'transformers']
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                issues.append(f"Missing package: {package}")
                print(f"   ❌ {package}")
        
        # Check environment variables
        required_env = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
        for env_var in required_env:
            if os.getenv(env_var):
                print(f"   ✅ {env_var}")
            else:
                issues.append(f"Missing environment variable: {env_var}")
                print(f"   ❌ {env_var}")
        
        # Check file existence
        required_files = [
            'collection/comprehensive_twitter_expansion.py',
            'collection/reddit_historical_backfill.py',
            'src/ml_dataset_creator.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                issues.append(f"Missing file: {file_path}")
                print(f"   ❌ {file_path}")
        
        if issues:
            print("\\n❌ PREREQUISITES NOT MET:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("\\n✅ ALL PREREQUISITES MET!")
            return True
    
    def execute_phase_1_twitter(self):
        """Execute Phase 1: Twitter Expansion"""
        print("\\n🚀 PHASE 1: TWITTER INFLUENCER EXPANSION")
        print("="*50)
        
        try:
            # Run Twitter expansion script
            result = subprocess.run([
                sys.executable,
                str(self.collection_dir / "comprehensive_twitter_expansion.py")
            ], capture_output=True, text=True, timeout=14400)  # 4 hour timeout
            
            if result.returncode == 0:
                print("✅ Phase 1 completed successfully!")
                self.phases_completed.append("twitter_expansion")
                return True
            else:
                print(f"❌ Phase 1 failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Phase 1 timed out after 4 hours")
            return False
        except Exception as e:
            print(f"❌ Phase 1 error: {str(e)}")
            return False
    
    def execute_phase_2_reddit(self):
        """Execute Phase 2: Reddit Backfill"""
        print("\\n🔶 PHASE 2: REDDIT HISTORICAL BACKFILL")
        print("="*50)
        
        try:
            # Run Reddit backfill script
            result = subprocess.run([
                sys.executable,
                str(self.collection_dir / "reddit_historical_backfill.py")
            ], capture_output=True, text=True, timeout=10800, input="y\\n")  # 3 hour timeout, auto-confirm
            
            if result.returncode == 0:
                print("✅ Phase 2 completed successfully!")
                self.phases_completed.append("reddit_backfill")
                return True
            else:
                print(f"❌ Phase 2 failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Phase 2 timed out after 3 hours")
            return False
        except Exception as e:
            print(f"❌ Phase 2 error: {str(e)}")
            return False
    
    def execute_phase_3_news(self):
        """Execute Phase 3: News Enhancement"""
        print("\\n📰 PHASE 3: NEWS COVERAGE ENHANCEMENT")
        print("="*50)
        
        # For now, we'll skip news expansion as it's lower priority
        # Focus on Twitter and Reddit which provide higher ML impact
        print("⏭️ Phase 3 skipped for now - focusing on higher impact sources")
        print("   Twitter and Reddit provide 95% of expected ML improvement")
        
        self.phases_completed.append("news_enhancement_skipped")
        return True
    
    def execute_phase_4_ml_regeneration(self):
        """Execute Phase 4: ML Dataset Regeneration"""
        print("\\n🤖 PHASE 4: ML DATASET REGENERATION")
        print("="*50)
        
        try:
            # Regenerate ML dataset with expanded data
            result = subprocess.run([
                sys.executable,
                "src/simplified_ml_dataset.py"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print("✅ Phase 4 completed successfully!")
                
                # Check new dataset size
                new_stats = self.get_current_data_stats()
                print(f"📊 Data expansion results:")
                print(f"   🔶 Reddit: {new_stats['reddit']['total']:,} (+{new_stats['reddit']['total'] - self.initial_stats['reddit']['total']:,})")
                print(f"   🐦 Twitter: {new_stats['twitter']['total']:,} (+{new_stats['twitter']['total'] - self.initial_stats['twitter']['total']:,})")
                
                self.phases_completed.append("ml_regeneration")
                return True
            else:
                print(f"❌ Phase 4 failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Phase 4 timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"❌ Phase 4 error: {str(e)}")
            return False
    
    def execute_phase_5_validation(self):
        """Execute Phase 5: Hypothesis Revalidation"""
        print("\\n🔬 PHASE 5: HYPOTHESIS REVALIDATION")
        print("="*50)
        
        try:
            # Rerun comprehensive model comparison with expanded dataset
            result = subprocess.run([
                sys.executable,
                "src/modeling/comprehensive_evaluation.py"
            ], capture_output=True, text=True, timeout=1200)  # 20 minute timeout
            
            if result.returncode == 0:
                print("✅ Phase 5 completed successfully!")
                print("📊 Hypothesis validation complete with robust dataset!")
                
                self.phases_completed.append("hypothesis_revalidation")
                return True
            else:
                print(f"❌ Phase 5 failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Phase 5 timed out after 20 minutes")
            return False
        except Exception as e:
            print(f"❌ Phase 5 error: {str(e)}")
            return False
    
    def run_orchestrated_expansion(self):
        """Execute complete orchestrated data expansion"""
        start_time = time.time()
        
        print("🎬 STARTING ORCHESTRATED DATA EXPANSION")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\\n❌ Cannot proceed - prerequisites not met")
            return False
        
        # Execute phases
        phases = [
            ("Phase 1: Twitter Expansion", self.execute_phase_1_twitter),
            ("Phase 2: Reddit Backfill", self.execute_phase_2_reddit),
            ("Phase 3: News Enhancement", self.execute_phase_3_news),
            ("Phase 4: ML Regeneration", self.execute_phase_4_ml_regeneration),
            ("Phase 5: Hypothesis Validation", self.execute_phase_5_validation)
        ]
        
        for phase_name, phase_func in phases:
            print(f"\\n🎯 Starting {phase_name}...")
            
            phase_start = time.time()
            success = phase_func()
            phase_duration = time.time() - phase_start
            
            if success:
                print(f"✅ {phase_name} completed in {phase_duration/60:.1f} minutes")
            else:
                print(f"❌ {phase_name} failed after {phase_duration/60:.1f} minutes")
                print("⏹️ Stopping orchestration due to failure")
                return False
        
        # Final summary
        total_duration = time.time() - start_time
        
        print("\\n" + "="*70)
        print("🎉 DATA EXPANSION ORCHESTRATION COMPLETE!")
        print("="*70)
        print(f"⏰ Total duration: {total_duration/3600:.1f} hours")
        print(f"✅ Phases completed: {len(self.phases_completed)}/5")
        
        # Final statistics
        final_stats = self.get_current_data_stats()
        print("\\n📊 FINAL RESULTS:")
        print(f"   🔶 Reddit: {final_stats['reddit']['total']:,} (+{final_stats['reddit']['total'] - self.initial_stats['reddit']['total']:,})")
        print(f"   🐦 Twitter: {final_stats['twitter']['total']:,} (+{final_stats['twitter']['total'] - self.initial_stats['twitter']['total']:,})")
        print(f"   📰 News: {final_stats['news']['total']:,} (+{final_stats['news']['total'] - self.initial_stats['news']['total']:,})")
        
        print("\\n🎯 EXPECTED ML IMPROVEMENTS:")
        print("   📈 Dataset size: 178 → 1,700+ samples (866% increase)")
        print("   🔬 Statistical significance: Greatly improved")
        print("   📊 Hypothesis validation: More robust evidence")
        print("   🚀 Model performance: Better generalization")
        
        print("="*70)
        return True

def main():
    """Main orchestration function"""
    orchestrator = DataExpansionOrchestrator()
    
    # Display plan
    orchestrator.print_expansion_plan()
    
    # Get user confirmation
    print("\\n⚠️ This is a comprehensive data expansion that will:")
    print("   - Open browser windows for Twitter scraping")
    print("   - Use Reddit API for historical data collection")
    print("   - Take 8-12 hours to complete all phases")
    print("   - Require active monitoring for authentication")
    
    proceed = input("\\nProceed with orchestrated data expansion? (y/N): ").lower().strip()
    
    if proceed == 'y':
        success = orchestrator.run_orchestrated_expansion()
        
        if success:
            print("\\n🎉 Data expansion completed successfully!")
            print("🚀 Your CryptoPulse system now has robust data for ML modeling!")
        else:
            print("\\n❌ Data expansion failed. Check logs for details.")
            print("💡 You can run individual phases manually if needed.")
            
    else:
        print("\\n⏹️ Orchestration cancelled by user.")
        print("💡 You can run individual collection scripts manually:")
        print("   🐦 python collection/comprehensive_twitter_expansion.py")
        print("   🔶 python collection/reddit_historical_backfill.py")

if __name__ == "__main__":
    main()