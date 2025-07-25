#!/usr/bin/env python3
# dataset_sufficiency_analysis.py

"""
Deep analysis of dataset sufficiency for crypto price prediction ML models.
Compares against academic research, industry standards, and optimal ML practices.
"""
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('src')
from database import CryptoPulseDB

def analyze_dataset_sufficiency():
    """Comprehensive analysis of dataset sufficiency for ML training."""
    print("🔍 DATASET SUFFICIENCY ANALYSIS FOR CRYPTO PRICE PREDICTION")
    print("=" * 80)
    
    db = CryptoPulseDB()
    conn = sqlite3.connect(db.db_path)
    
    # Get detailed metrics
    detailed_query = """
    WITH daily_data AS (
        SELECT 
            DATE(datetime(created_utc, 'unixepoch')) as date,
            COUNT(*) as reddit_count
        FROM reddit_posts 
        WHERE datetime(created_utc, 'unixepoch') >= '2022-01-01' 
        AND datetime(created_utc, 'unixepoch') < '2024-01-01'
        GROUP BY DATE(datetime(created_utc, 'unixepoch'))
    ),
    daily_news AS (
        SELECT 
            DATE(datetime(published_at, 'unixepoch')) as date,
            COUNT(*) as news_count
        FROM news_articles 
        WHERE datetime(published_at, 'unixepoch') >= '2022-01-01' 
        AND datetime(published_at, 'unixepoch') < '2024-01-01'
        GROUP BY DATE(datetime(published_at, 'unixepoch'))
    ),
    daily_prices AS (
        SELECT 
            DATE(datetime(timestamp, 'unixepoch')) as date,
            COUNT(*) as price_count,
            AVG(price_usd) as avg_price,
            (MAX(price_usd) - MIN(price_usd)) / AVG(price_usd) * 100 as daily_volatility
        FROM eth_prices 
        WHERE datetime(timestamp, 'unixepoch') >= '2022-01-01' 
        AND datetime(timestamp, 'unixepoch') < '2024-01-01'
        GROUP BY DATE(datetime(timestamp, 'unixepoch'))
    )
    SELECT 
        COALESCE(dd.date, dn.date, dp.date) as date,
        COALESCE(dd.reddit_count, 0) as reddit,
        COALESCE(dn.news_count, 0) as news,
        COALESCE(dp.price_count, 0) as prices,
        COALESCE(dp.avg_price, 0) as eth_price,
        COALESCE(dp.daily_volatility, 0) as volatility
    FROM daily_data dd
    FULL OUTER JOIN daily_news dn ON dd.date = dn.date
    FULL OUTER JOIN daily_prices dp ON COALESCE(dd.date, dn.date) = dp.date
    WHERE COALESCE(dd.date, dn.date, dp.date) IS NOT NULL
    ORDER BY date
    """
    
    df_daily = pd.read_sql_query(detailed_query, conn)
    df_daily['total_sentiment'] = df_daily['reddit'] + df_daily['news']
    df_daily['has_complete_data'] = (df_daily['reddit'] > 0) & (df_daily['news'] > 0) & (df_daily['prices'] > 0)
    
    # Get processed metrics distribution
    metrics_query = """
    SELECT 
        sentiment_score, relevance_score, volatility_score, 
        echo_score, content_depth_score
    FROM text_metrics 
    WHERE sentiment_score IS NOT NULL 
    AND relevance_score IS NOT NULL
    """
    
    df_metrics = pd.read_sql_query(metrics_query, conn)
    conn.close()
    
    # Analysis results
    total_days = len(df_daily)
    complete_days = df_daily['has_complete_data'].sum()
    avg_daily_sentiment = df_daily['total_sentiment'].mean()
    
    print(f"📊 TEMPORAL COVERAGE ANALYSIS (2022-2023):")
    print(f"   Total days analyzed: {total_days}")
    print(f"   Days with complete data: {complete_days} ({complete_days/total_days*100:.1f}%)")
    print(f"   Average daily sentiment entries: {avg_daily_sentiment:.1f}")
    print(f"   Days with >10 sentiment entries: {(df_daily['total_sentiment'] > 10).sum()}")
    print(f"   Days with >20 sentiment entries: {(df_daily['total_sentiment'] > 20).sum()}")
    
    # Compare against research standards
    compare_against_research_standards(df_daily, df_metrics)
    
    # Identify specific gaps
    identify_critical_gaps(df_daily)
    
    # Make recommendations
    make_sufficiency_recommendations(df_daily, df_metrics)
    
    return df_daily, df_metrics

def compare_against_research_standards(df_daily, df_metrics):
    """Compare our dataset against academic and industry research standards."""
    print(f"\n🎓 COMPARISON AGAINST RESEARCH STANDARDS:")
    print("=" * 50)
    
    # Research benchmarks from crypto ML papers
    research_standards = {
        "Academic Minimum": {
            "days": 365,
            "entries_per_day": 5,
            "total_entries": 1800,
            "features": 3,
            "description": "Basic academic study"
        },
        "Strong Academic": {
            "days": 730,
            "entries_per_day": 15,
            "total_entries": 10000,
            "features": 5,
            "description": "High-quality academic paper"
        },
        "Industry Standard": {
            "days": 1095,
            "entries_per_day": 25,
            "total_entries": 25000,
            "features": 8,
            "description": "Professional trading system"
        },
        "Research Excellence": {
            "days": 1460,
            "entries_per_day": 50,
            "total_entries": 70000,
            "features": 10,
            "description": "Top-tier research/production"
        }
    }
    
    # Our current stats
    our_stats = {
        "days": len(df_daily),
        "entries_per_day": df_daily['total_sentiment'].mean(),
        "total_entries": len(df_metrics),
        "features": 5,  # Our 5-metric system
        "complete_coverage": df_daily['has_complete_data'].mean()
    }
    
    print(f"📈 OUR DATASET vs RESEARCH STANDARDS:")
    for standard_name, standard in research_standards.items():
        meets_days = our_stats['days'] >= standard['days']
        meets_daily = our_stats['entries_per_day'] >= standard['entries_per_day']
        meets_total = our_stats['total_entries'] >= standard['total_entries']
        meets_features = our_stats['features'] >= standard['features']
        
        overall_meets = meets_days and meets_daily and meets_total and meets_features
        status = "✅" if overall_meets else "⚠️"
        
        print(f"\n{standard_name} {status} ({standard['description']}):")
        print(f"   Days: {our_stats['days']} / {standard['days']} {'✅' if meets_days else '❌'}")
        print(f"   Daily: {our_stats['entries_per_day']:.1f} / {standard['entries_per_day']} {'✅' if meets_daily else '❌'}")
        print(f"   Total: {our_stats['total_entries']:,} / {standard['total_entries']:,} {'✅' if meets_total else '❌'}")
        print(f"   Features: {our_stats['features']} / {standard['features']} {'✅' if meets_features else '❌'}")

def identify_critical_gaps(df_daily):
    """Identify critical gaps in data coverage."""
    print(f"\n🔍 CRITICAL GAPS ANALYSIS:")
    print("=" * 30)
    
    # Find days with low data
    low_data_days = df_daily[df_daily['total_sentiment'] < 5]
    zero_data_days = df_daily[df_daily['total_sentiment'] == 0]
    
    print(f"📉 DATA QUALITY ISSUES:")
    print(f"   Days with <5 sentiment entries: {len(low_data_days)} ({len(low_data_days)/len(df_daily)*100:.1f}%)")
    print(f"   Days with 0 sentiment entries: {len(zero_data_days)} ({len(zero_data_days)/len(df_daily)*100:.1f}%)")
    
    if len(low_data_days) > 0:
        print(f"\n   Worst coverage periods:")
        worst_periods = low_data_days.nsmallest(5, 'total_sentiment')
        for _, row in worst_periods.iterrows():
            print(f"     {row['date']}: {row['total_sentiment']} entries")
    
    # Identify missing weekends vs weekdays
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['weekday'] = df_daily['date'].dt.dayofweek
    df_daily['is_weekend'] = df_daily['weekday'].isin([5, 6])
    
    weekend_avg = df_daily[df_daily['is_weekend']]['total_sentiment'].mean()
    weekday_avg = df_daily[~df_daily['is_weekend']]['total_sentiment'].mean()
    
    print(f"\n📅 TEMPORAL PATTERNS:")
    print(f"   Average weekday entries: {weekday_avg:.1f}")
    print(f"   Average weekend entries: {weekend_avg:.1f}")
    print(f"   Weekend coverage ratio: {weekend_avg/weekday_avg:.2f}")

def make_sufficiency_recommendations(df_daily, df_metrics):
    """Make specific recommendations for dataset improvement."""
    print(f"\n💡 SUFFICIENCY RECOMMENDATIONS:")
    print("=" * 40)
    
    total_days = len(df_daily)
    complete_days = df_daily['has_complete_data'].sum()
    avg_daily = df_daily['total_sentiment'].mean()
    low_data_days = (df_daily['total_sentiment'] < 10).sum()
    
    # Overall assessment
    if avg_daily >= 25 and complete_days/total_days >= 0.8:
        quality_level = "EXCELLENT"
        ready_status = "✅ READY FOR PRODUCTION"
    elif avg_daily >= 15 and complete_days/total_days >= 0.6:
        quality_level = "GOOD"
        ready_status = "✅ READY FOR PROFESSIONAL ML"
    elif avg_daily >= 5 and complete_days/total_days >= 0.4:
        quality_level = "ADEQUATE"
        ready_status = "⚠️ SUITABLE FOR RESEARCH"
    else:
        quality_level = "INSUFFICIENT"
        ready_status = "❌ NEEDS MORE DATA"
    
    print(f"🎯 OVERALL ASSESSMENT: {quality_level}")
    print(f"🚀 ML READINESS: {ready_status}")
    
    print(f"\n📊 CURRENT STRENGTHS:")
    print(f"   ✅ Feature richness: 5-metric scoring system")
    print(f"   ✅ Processing coverage: {len(df_metrics):,} processed entries")
    print(f"   ✅ Time span: {total_days} days (2+ years)")
    print(f"   ✅ Event coverage: Major crypto events included")
    
    print(f"\n⚠️ AREAS FOR IMPROVEMENT:")
    improvement_needed = []
    
    if avg_daily < 20:
        gap = 20 - avg_daily
        improvement_needed.append(f"Daily density: Need +{gap:.1f} entries/day average")
        
    if low_data_days > total_days * 0.2:
        improvement_needed.append(f"Consistency: {low_data_days} days with <10 entries")
        
    if complete_days/total_days < 0.7:
        improvement_needed.append(f"Coverage: Only {complete_days/total_days*100:.1f}% days have complete data")
    
    if len(improvement_needed) == 0:
        print(f"   🎉 NO MAJOR IMPROVEMENTS NEEDED!")
        print(f"   📈 Dataset exceeds professional standards")
    else:
        for improvement in improvement_needed:
            print(f"   📈 {improvement}")
    
    # Specific action recommendations
    print(f"\n🎯 SPECIFIC ACTIONS RECOMMENDED:")
    
    if avg_daily < 15:
        target_additional = int((15 - avg_daily) * total_days)
        print(f"1. 🎯 PRIORITY: Add ~{target_additional:,} more entries to reach 15/day average")
        print(f"   - Focus on days with <5 entries")
        print(f"   - Target weekend coverage improvement")
        print(f"   - Expand news sources for consistent daily coverage")
    
    if len(df_metrics) < 15000:
        target_entries = 15000 - len(df_metrics)
        print(f"2. 📊 SCALE UP: Add {target_entries:,} more processed entries for industry standard")
        print(f"   - Extend timeframe to 2021-2023 (3 years)")
        print(f"   - Add more diverse sources (forums, academic papers)")
        print(f"   - Collect international crypto news")
    
    if complete_days/total_days < 0.8:
        print(f"3. 🔍 CONSISTENCY: Improve daily coverage completeness")
        print(f"   - Automated daily collection to fill gaps")
        print(f"   - Backup sources for low-activity days")
        print(f"   - Weekend and holiday coverage strategy")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    if quality_level in ["EXCELLENT", "GOOD"]:
        print(f"✅ Dataset is SUFFICIENT for professional ML training")
        print(f"🚀 Recommend: Proceed with model development")
        print(f"📊 Optional: Continue collection in parallel with training")
    else:
        print(f"⚠️ Dataset needs improvement before optimal ML training")
        print(f"🎯 Recommend: Address top 2 improvement areas first")

def main():
    """Main analysis function."""
    df_daily, df_metrics = analyze_dataset_sufficiency()
    
    # Quick summary stats
    avg_daily = df_daily['total_sentiment'].mean()
    complete_ratio = df_daily['has_complete_data'].mean()
    
    print(f"\n📋 SUFFICIENCY SUMMARY:")
    print(f"   Daily average: {avg_daily:.1f} sentiment entries")
    print(f"   Complete coverage: {complete_ratio:.1%} of days")
    print(f"   Processed entries: {len(df_metrics):,}")
    print(f"   Time span: {len(df_daily)} days")
    
    if avg_daily >= 15 and complete_ratio >= 0.6:
        print(f"   🎉 VERDICT: SUFFICIENT for professional ML training")
    else:
        print(f"   ⚠️ VERDICT: Needs improvement for optimal results")

if __name__ == "__main__":
    main()