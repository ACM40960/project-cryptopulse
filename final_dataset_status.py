#!/usr/bin/env python3
# final_dataset_status.py

"""
Final comprehensive analysis of dataset after all collection efforts.
Determine if we've achieved professional-grade ML training standards.
"""
import sys
import sqlite3
import pandas as pd
from datetime import datetime

sys.path.append('src')
from database import CryptoPulseDB

def final_dataset_analysis():
    """Comprehensive final analysis of complete dataset."""
    print("🏆 FINAL DATASET STATUS - POST ALL COLLECTION EFFORTS")
    print("=" * 80)
    
    db = CryptoPulseDB()
    conn = sqlite3.connect(db.db_path)
    
    # Overall dataset statistics
    overview_queries = {
        'reddit_total': "SELECT COUNT(*) FROM reddit_posts",
        'twitter_total': "SELECT COUNT(*) FROM twitter_posts", 
        'news_total': "SELECT COUNT(*) FROM news_articles",
        'prices_total': "SELECT COUNT(*) FROM eth_prices",
        'processed_total': "SELECT COUNT(*) FROM text_metrics"
    }
    
    stats = {}
    for key, query in overview_queries.items():
        result = conn.execute(query).fetchone()
        stats[key] = result[0] if result else 0
    
    print(f"📊 COMPLETE DATASET OVERVIEW:")
    print(f"   Reddit Posts: {stats['reddit_total']:,}")
    print(f"   Twitter Posts: {stats['twitter_total']:,}")
    print(f"   News Articles: {stats['news_total']:,}")
    print(f"   ETH Price Points: {stats['prices_total']:,}")
    print(f"   Processed Entries: {stats['processed_total']:,}")
    
    total_raw = stats['reddit_total'] + stats['twitter_total'] + stats['news_total']
    print(f"   TOTAL RAW ENTRIES: {total_raw:,}")
    
    # 2022-2023 specific analysis (our target timeframe)
    timeframe_query = """
    WITH daily_sentiment AS (
        SELECT 
            DATE(datetime(created_utc, 'unixepoch')) as date,
            COUNT(*) as reddit_count
        FROM reddit_posts 
        WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01' 
        AND datetime(created_utc, 'unixepoch') < '2024-01-01'
        GROUP BY DATE(datetime(created_utc, 'unixepoch'))
        
        UNION ALL
        
        SELECT 
            DATE(datetime(published_at, 'unixepoch')) as date,
            COUNT(*) as news_count
        FROM news_articles 
        WHERE datetime(published_at, 'unixepoch') >= '2022-01-01' 
        AND datetime(published_at, 'unixepoch') < '2024-01-01'
        GROUP BY DATE(datetime(published_at, 'unixepoch'))
    )
    SELECT 
        date,
        SUM(reddit_count) as daily_total
    FROM daily_sentiment
    GROUP BY date
    ORDER BY date
    """
    
    df_daily = pd.read_sql_query(timeframe_query, conn)
    
    # Calculate key metrics
    total_days = len(df_daily)
    total_entries_2022_23 = df_daily['daily_total'].sum()
    avg_daily = df_daily['daily_total'].mean()
    days_with_10_plus = (df_daily['daily_total'] >= 10).sum()
    days_with_5_plus = (df_daily['daily_total'] >= 5).sum()
    days_with_zero = (df_daily['daily_total'] == 0).sum()
    
    print(f"\n🎯 2022-2023 OPTIMAL TIMEFRAME ANALYSIS:")
    print(f"   Total days: {total_days}")
    print(f"   Total entries: {total_entries_2022_23:,}")
    print(f"   Average entries/day: {avg_daily:.1f}")
    print(f"   Days with 10+ entries: {days_with_10_plus} ({days_with_10_plus/total_days*100:.1f}%)")
    print(f"   Days with 5+ entries: {days_with_5_plus} ({days_with_5_plus/total_days*100:.1f}%)")
    print(f"   Days with 0 entries: {days_with_zero} ({days_with_zero/total_days*100:.1f}%)")
    
    # Professional standards assessment
    print(f"\n🏅 PROFESSIONAL STANDARDS ASSESSMENT:")
    
    standards = {
        "Academic Minimum": {"daily": 5, "total": 1800, "verdict": ""},
        "Strong Academic": {"daily": 15, "total": 10000, "verdict": ""},
        "Industry Standard": {"daily": 25, "total": 25000, "verdict": ""},
        "Research Excellence": {"daily": 50, "total": 70000, "verdict": ""}
    }
    
    for standard_name, criteria in standards.items():
        meets_daily = avg_daily >= criteria["daily"]
        meets_total = stats['processed_total'] >= criteria["total"]
        
        if meets_daily and meets_total:
            verdict = "✅ EXCEEDS"
        elif meets_total:
            verdict = "⚠️ PARTIAL (total✅, daily❌)"
        else:
            verdict = "❌ BELOW"
        
        standards[standard_name]["verdict"] = verdict
        
        print(f"   {standard_name}: {verdict}")
        print(f"     Daily: {avg_daily:.1f} / {criteria['daily']} {'✅' if meets_daily else '❌'}")
        print(f"     Total: {stats['processed_total']:,} / {criteria['total']:,} {'✅' if meets_total else '❌'}")
    
    # ML Training Readiness Assessment
    print(f"\n🤖 ML TRAINING READINESS:")
    
    if avg_daily >= 15 and stats['processed_total'] >= 10000:
        ml_readiness = "✅ READY FOR PROFESSIONAL ML TRAINING"
        recommendation = "Proceed with model development immediately"
    elif avg_daily >= 5 and stats['processed_total'] >= 5000:
        ml_readiness = "⚠️ SUITABLE FOR ACADEMIC RESEARCH"
        recommendation = "Can train models, but may need more data for production"
    else:
        ml_readiness = "❌ INSUFFICIENT FOR RELIABLE ML"
        recommendation = "Continue data collection before model training"
    
    print(f"   Status: {ml_readiness}")
    print(f"   Recommendation: {recommendation}")
    
    # Gap analysis
    if avg_daily < 15:
        daily_gap = 15 - avg_daily
        total_additional_needed = int(daily_gap * total_days)
        print(f"\n📈 GAP ANALYSIS:")
        print(f"   Need +{daily_gap:.1f} entries/day to reach Strong Academic standard")
        print(f"   Total additional entries needed: ~{total_additional_needed:,}")
        print(f"   Current data utilization: {avg_daily/15*100:.1f}% of Strong Academic target")
    
    # Data quality assessment
    consistency_score = days_with_5_plus / total_days
    density_score = min(avg_daily / 15, 1.0)  # Cap at 100%
    volume_score = min(stats['processed_total'] / 12000, 1.0)  # Cap at 100%
    
    overall_quality = (consistency_score + density_score + volume_score) / 3
    
    print(f"\n📊 DATA QUALITY SCORES:")
    print(f"   Consistency: {consistency_score:.1%} (days with 5+ entries)")
    print(f"   Density: {density_score:.1%} (avg daily vs target)")
    print(f"   Volume: {volume_score:.1%} (total processed vs target)")
    print(f"   OVERALL QUALITY: {overall_quality:.1%}")
    
    if overall_quality >= 0.8:
        quality_verdict = "🏆 EXCELLENT"
    elif overall_quality >= 0.6:
        quality_verdict = "✅ GOOD"
    elif overall_quality >= 0.4:
        quality_verdict = "⚠️ ADEQUATE" 
    else:
        quality_verdict = "❌ POOR"
    
    print(f"   Quality Rating: {quality_verdict}")
    
    conn.close()
    
    return {
        'stats': stats,
        'daily_avg': avg_daily,
        'total_days': total_days,
        'ml_readiness': ml_readiness,
        'overall_quality': overall_quality,
        'standards': standards
    }

def final_recommendations(analysis_results):
    """Provide final recommendations based on analysis."""
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    print("=" * 40)
    
    daily_avg = analysis_results['daily_avg']
    ml_readiness = analysis_results['ml_readiness']
    overall_quality = analysis_results['overall_quality']
    
    if "READY FOR PROFESSIONAL" in ml_readiness:
        print(f"🎉 CONGRATULATIONS!")
        print(f"✅ Dataset meets professional ML training standards")
        print(f"🚀 RECOMMENDED NEXT STEPS:")
        print(f"   1. Begin price labeling and feature engineering")
        print(f"   2. Implement time-series cross-validation")
        print(f"   3. Train multiple model types (Ridge, RF, LSTM)")
        print(f"   4. Focus on model optimization rather than more data")
        
    elif "SUITABLE FOR ACADEMIC" in ml_readiness:
        print(f"✅ GOOD PROGRESS!")
        print(f"⚠️ Dataset suitable for research, approaching professional grade")
        print(f"🎯 RECOMMENDED APPROACH:")
        print(f"   1. Start with baseline models on current data")
        print(f"   2. Continue targeted collection in parallel")
        print(f"   3. Focus on improving daily consistency")
        print(f"   4. Aim for 15+ entries/day average")
        
    else:
        print(f"⚠️ MORE DATA NEEDED")
        print(f"❌ Dataset below reliable ML training threshold")
        print(f"🎯 PRIORITY ACTIONS:")
        print(f"   1. Intensive daily gap filling (focus on zero-days)")
        print(f"   2. Expand to 2021 data for more volume")
        print(f"   3. Consider alternative data sources")
        print(f"   4. Target consistent 10+ entries/day before ML")
    
    # Specific improvement targets
    if daily_avg < 15:
        improvement_needed = 15 - daily_avg
        print(f"\n📈 SPECIFIC IMPROVEMENT TARGETS:")
        print(f"   Current: {daily_avg:.1f} entries/day")
        print(f"   Target: 15.0 entries/day (Strong Academic)")
        print(f"   Gap: +{improvement_needed:.1f} entries/day needed")
        print(f"   Strategy: Focus on {int(improvement_needed * 730)} additional entries over 2-year period")

def main():
    """Main analysis function."""
    analysis = final_dataset_analysis()
    final_recommendations(analysis)
    
    print(f"\n🎯 EXECUTIVE SUMMARY:")
    print(f"   Total Dataset: {analysis['stats']['reddit_total'] + analysis['stats']['twitter_total'] + analysis['stats']['news_total']:,} raw entries")
    print(f"   Processed: {analysis['stats']['processed_total']:,} entries")
    print(f"   Daily Average: {analysis['daily_avg']:.1f} entries/day")
    print(f"   ML Readiness: {analysis['ml_readiness']}")
    print(f"   Quality Score: {analysis['overall_quality']:.1%}")

if __name__ == "__main__":
    main()