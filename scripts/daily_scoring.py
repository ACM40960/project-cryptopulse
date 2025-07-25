#!/usr/bin/env python3
# scripts/daily_scoring.py

"""
Daily metrics scoring script for CryptoPulse.
Processes all unprocessed text entries and calculates 5-metric scores.
Designed to run after data collection via cron job.
"""
import sys
import os
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from score_metrics import CryptoMetricsScorer

def setup_logging():
    """Setup logging for daily scoring."""
    log_file = f"logs/daily_scoring_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_daily_scoring():
    """Process all unprocessed entries with 5-metric scoring."""
    setup_logging()
    logging.info("=== Starting Daily CryptoPulse Metrics Scoring ===")
    
    try:
        # Initialize scorer
        logging.info("Initializing CryptoMetricsScorer...")
        scorer = CryptoMetricsScorer()
        
        # Process all unprocessed entries in batches
        logging.info("Processing unprocessed entries...")
        total_processed = scorer.process_all_unprocessed(
            batch_size=20,      # Reasonable batch size for daily processing
            max_batches=50      # Limit to prevent excessive runtime
        )
        
        # Summary
        logging.info("=== Daily Scoring Complete ===")
        logging.info(f"Total entries processed: {total_processed}")
        
        return total_processed
        
    except Exception as e:
        logging.error(f"Error in daily scoring: {e}")
        return 0

if __name__ == "__main__":
    results = run_daily_scoring()
    print(f"Daily scoring complete. Processed {results} entries.")