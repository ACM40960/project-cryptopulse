# CryptoPulse Dataset Temporal Analysis Summary

## Executive Summary

The CryptoPulse dataset contains **14,851** total entries spanning from **August 31, 2015** to **July 28, 2025** across three data sources:
- **Reddit Posts**: 9,446 entries (63.6%)
- **News Articles**: 4,112 entries (27.7%) 
- **Twitter Posts**: 1,293 entries (8.7%)

**Key Finding**: Only **5.6%** of days have sufficient data density (15+ entries/day) for robust ML training, with significant data quality improvements occurring in 2024-2025.

## Data Distribution by Year and Month

### Yearly Distribution
| Year | Total Entries | Avg/Day | Reddit | Twitter | News | Days ≥15 entries |
|------|---------------|---------|--------|---------|------|------------------|
| 2015 | 4 | 0.0 | 1 | 0 | 3 | 0 |
| 2016 | 26 | 0.1 | 23 | 0 | 3 | 0 |
| 2017 | 148 | 0.4 | 142 | 0 | 6 | 0 |
| 2018 | 107 | 0.3 | 102 | 0 | 5 | 0 |
| 2019 | 108 | 0.3 | 107 | 0 | 1 | 0 |
| 2020 | 243 | 0.7 | 88 | 0 | 155 | 0 |
| 2021 | 1,102 | 3.0 | 419 | 0 | 683 | 3 |
| 2022 | 1,743 | 4.8 | 347 | 0 | 1,396 | 10 |
| 2023 | 1,343 | 3.7 | 165 | 0 | 1,178 | 2 |
| **2024** | **4,201** | **11.5** | **4,201** | **0** | **0** | **126** |
| **2025** | **5,826** | **27.9** | **3,851** | **1,293** | **682** | **63** |

### Top Monthly Periods by Volume
1. **2025-07**: 3,921 entries (140.0/day) - Reddit: 2,228, Twitter: 1,293, News: 400
2. **2025-06**: 1,093 entries (36.4/day) - Reddit: 1,045, News: 48
3. **2024-12**: 916 entries (29.5/day) - Reddit: 916
4. **2024-11**: 684 entries (22.8/day) - Reddit: 684
5. **2024-05**: 500 entries (16.1/day) - Reddit: 500

## Data Density Analysis

### Overall Density Distribution
- **Total days in dataset**: 3,620 days
- **Days with any data**: 1,957 days (54.1%)
- **Days with sufficient data (15+ entries)**: 204 days (5.6%)
- **Days with high volume (50+ entries)**: 37 days (1.0%)
- **Days with very high volume (100+ entries)**: 9 days (0.2%)

### High-Density Periods (15+ entries/day)
1. **2025-06-01 to 2025-07-28** (58 days): 56.4 avg entries/day, 3,273 total
2. **2024-10-08 to 2024-12-31** (85 days): 25.9 avg entries/day, 1,710 total  
3. **2024-04-25 to 2024-08-13** (111 days): 17.7 avg entries/day, 956 total

## Data Quality Assessment

### Quality Score Methodology
Quality scores calculated using:
- **50% Volume** (log-scaled total entries)
- **30% Consistency** (inverse of coefficient of variation)
- **20% Source Diversity** (number of active sources / 3)

### Top Quality Periods
1. **2025-07**: Quality score 0.833 (3,921 entries, 3 sources)
2. **2025-06**: Quality score 0.766 (1,093 entries, 2 sources)
3. **2024-12**: Quality score 0.716 (916 entries, 1 source)
4. **2024-11**: Quality score 0.687 (684 entries, 1 source)
5. **2024-05**: Quality score 0.684 (500 entries, 1 source)

## Gap Analysis and Risk Assessment

### Major Data Gaps (30+ days with <5 entries/day)
1. **2015-08-31 to 2019-10-09** (1,501 days): 0.2 avg entries/day
2. **2019-10-16 to 2020-07-30** (289 days): 0.5 avg entries/day
3. **2020-08-01 to 2020-12-02** (124 days): 0.7 avg entries/day
4. **2020-12-04 to 2021-02-15** (74 days): 1.4 avg entries/day
5. **2024-02-29 to 2024-04-23** (55 days): 0.9 avg entries/day

### Risk Factors for ML Training
- **High Risk**: 6 major gaps (30+ days with minimal data)
- **Medium Risk**: 835 Reddit-only days, 524 News-only days
- **Low Risk**: Only 2 continuous high-quality periods identified

## Optimal Training Window Recommendations

### Ranked by Combined Quality Score

#### 1. **6-Month Period: February 2025 - July 2025** ⭐ RECOMMENDED
- **Total entries**: 5,717
- **Quality score**: 0.700
- **Source breakdown**: Reddit 65.6%, Twitter 22.6%, News 11.8%
- **Strengths**: Multi-source, recent data, high volume, good consistency
- **Recommendation strength**: 0.681

#### 2. **12-Month Period: August 2024 - July 2025**
- **Total entries**: 8,498  
- **Quality score**: 0.681
- **Source breakdown**: Reddit 76.8%, Twitter 15.2%, News 8.0%
- **Strengths**: Large volume, includes Twitter data, recent trends
- **Recommendation strength**: 0.657

#### 3. **18-Month Period: October 2021 - March 2023**
- **Total entries**: 2,449
- **Quality score**: 0.601
- **Source breakdown**: Reddit 21.1%, News 78.9%
- **Strengths**: News-heavy perspective, different market conditions
- **Recommendation strength**: 0.592

#### 4. **24-Month Period: May 2021 - April 2023**
- **Total entries**: 2,978
- **Quality score**: 0.597  
- **Source breakdown**: Reddit 24.0%, News 76.0%
- **Strengths**: Longer historical perspective, balanced news coverage
- **Recommendation strength**: 0.589

## Specific Recommendations for ML Training

### Primary Recommendation: **February 2025 - July 2025**
- **Rationale**: Best balance of volume, source diversity, and data quality
- **Ideal for**: Real-time sentiment analysis, recent market behavior modeling
- **Considerations**: Limited historical market cycle coverage

### Secondary Recommendation: **May 2024 - December 2024**  
- **Rationale**: Consistent high-quality Reddit data, good volume
- **Ideal for**: Reddit-focused sentiment analysis, consistent data patterns
- **Considerations**: Single-source dependence, limited market event coverage

### Alternative for Historical Analysis: **2021-2023 News-Heavy Periods**
- **Rationale**: Different perspective, covers various market conditions
- **Ideal for**: News sentiment analysis, longer-term trend analysis
- **Considerations**: Lower overall volume, source imbalance

### Periods to Avoid
- **Pre-2021**: Extremely sparse data (avg <1 entry/day)
- **2023-Early 2024 Transition**: Inconsistent data patterns
- **Individual months with single-source dependence**: Risk of bias

## Text Metrics Coverage
- **Total processed entries**: 14,848 with text metrics
- **Coverage by source**: Reddit 9,443, Twitter 1,293, News 4,112
- **Processing completeness**: ~100% coverage across all sources

## Conclusion

The CryptoPulse dataset shows a clear evolution in data quality, with **2024-2025 representing the optimal period for ML training** due to:
1. **Highest data density** (15+ entries/day consistently)
2. **Multi-source coverage** (especially in 2025-07)
3. **Complete text metrics processing**
4. **Recent market relevance**

For robust ML model development, the **February 2025 - July 2025** period provides the best foundation, with the **May 2024 - December 2024** period serving as an excellent supplement for training data volume.