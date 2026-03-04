"""
Feature Engineering Script
Transforms raw historical data into ML-ready features

Input:  historical_data.csv (262 rows, 5 columns)
Output: training-data.csv (262 rows, 29 features + label)
        train.csv (208 rows for training)
        test.csv (54 rows for testing)
"""

import pandas as pd
import numpy as np

# Thresholds (from anomaly detection logic)
THRESHOLDS = {
    'sentiment_high': -0.06,
    'sentiment_low': -0.45,
    'market_significant': 0.75,
    'vix_elevated': 20,
    'vix_high': 25,
    'vix_panic': 35
}

def calculate_anomaly_score(row):
    """Rule-based anomaly scoring (0-100)"""
    score = 0
    divergence_type = "none"
    
    sentiment = row['sentiment']
    market_return = row['market_return']
    vix = row['vix']
    
    # Divergence patterns
    if sentiment > THRESHOLDS['sentiment_high'] and market_return < -0.5:
        score += 50
        divergence_type = "positive_news_negative_market"
    elif sentiment < THRESHOLDS['sentiment_low'] and market_return > 0.5:
        score += 40
        divergence_type = "negative_news_positive_market"
    elif abs(sentiment - market_return) > 1.5 and vix > THRESHOLDS['vix_elevated']:
        score += 35
        divergence_type = "moderate_divergence_with_stress"
    
    # VIX thresholds
    if vix > THRESHOLDS['vix_panic']:
        score += 50
        if divergence_type == "none":
            divergence_type = "panic_level_vix"
    elif vix > THRESHOLDS['vix_high']:
        score += 30
        if divergence_type == "none":
            divergence_type = "high_vix"
    elif vix > THRESHOLDS['vix_elevated']:
        score += 15
    
    # Magnitude bonus
    divergence_magnitude = abs(sentiment - market_return)
    if divergence_magnitude > 2.5:
        score += 20
    
    score = min(score, 100)
    
    return score, divergence_type, divergence_magnitude

def create_features(df):
    """Generate all 29 features"""
    
    print("🔧 Generating features...")
    
    # 1. BASE FEATURES (5)
    print("  ✓ Base features (5)")
    df['divergence_magnitude'] = abs(df['sentiment'] - df['market_return'])
    
    # Calculate anomaly scores
    anomaly_data = df.apply(calculate_anomaly_score, axis=1, result_type='expand')
    df['anomaly_score'] = anomaly_data[0]
    df['divergence_type'] = anomaly_data[1]
    df['divergence_magnitude'] = anomaly_data[2]
    
    # 2. ANOMALY FLAGS (4)
    print("  ✓ Anomaly flags (4)")
    df['is_anomaly'] = (df['anomaly_score'] >= 50).astype(int)
    df['is_high_risk'] = (df['anomaly_score'] >= 70).astype(int)
    df['is_medium_risk'] = ((df['anomaly_score'] >= 50) & (df['anomaly_score'] < 70)).astype(int)
    df['is_low_risk'] = (df['anomaly_score'] < 50).astype(int)
    
    # 3. DIVERGENCE TYPE FLAGS (one-hot encoded) (4)
    print("  ✓ Divergence type flags (4)")
    df['div_positive_news_negative_market'] = (df['divergence_type'] == 'positive_news_negative_market').astype(int)
    df['div_negative_news_positive_market'] = (df['divergence_type'] == 'negative_news_positive_market').astype(int)
    df['div_moderate_divergence'] = (df['divergence_type'] == 'moderate_divergence_with_stress').astype(int)
    df['div_none'] = (df['divergence_type'] == 'none').astype(int)
    
    # 4. VIX CATEGORY FLAGS (5)
    print("  ✓ VIX category flags (5)")
    df['vix_high'] = (df['vix'] > THRESHOLDS['vix_high']).astype(int)
    df['vix_panic'] = (df['vix'] > THRESHOLDS['vix_panic']).astype(int)
    df['vix_elevated_high'] = ((df['vix'] > THRESHOLDS['vix_high']) & (df['vix'] <= THRESHOLDS['vix_panic'])).astype(int)
    df['vix_elevated'] = ((df['vix'] > THRESHOLDS['vix_elevated']) & (df['vix'] <= THRESHOLDS['vix_high'])).astype(int)
    df['vix_normal'] = (df['vix'] <= THRESHOLDS['vix_elevated']).astype(int)
    
    # 5. ROLLING AVERAGES (6)
    print("  ✓ Rolling averages (6)")
    df['sentiment_3d_avg'] = df['sentiment'].rolling(window=3, min_periods=1).mean()
    df['market_3d_avg'] = df['market_return'].rolling(window=3, min_periods=1).mean()
    df['vix_3d_avg'] = df['vix'].rolling(window=3, min_periods=1).mean()
    
    df['sentiment_7d_avg'] = df['sentiment'].rolling(window=7, min_periods=1).mean()
    df['market_7d_avg'] = df['market_return'].rolling(window=7, min_periods=1).mean()
    df['vix_7d_avg'] = df['vix'].rolling(window=7, min_periods=1).mean()
    
    # 6. INTERACTION TERMS (3)
    print("  ✓ Interaction terms (3)")
    df['sentiment_market_interaction'] = df['sentiment'] * df['market_return']
    df['vix_divergence_interaction'] = df['vix'] * df['divergence_magnitude']
    df['sentiment_vix_interaction'] = df['sentiment'] * df['vix']
    
    # 7. LAG FEATURES (3)
    print("  ✓ Lag features (3)")
    df['prev_sentiment'] = df['sentiment'].shift(1)
    df['prev_market'] = df['market_return'].shift(1)
    df['prev_vix'] = df['vix'].shift(1)
    
    # Fill NaN values in lag features (first row)
    df['prev_sentiment'].fillna(df['sentiment'].iloc[0], inplace=True)
    df['prev_market'].fillna(df['market_return'].iloc[0], inplace=True)
    df['prev_vix'].fillna(df['vix'].iloc[0], inplace=True)
    
    print(f"\n Total features: {len(df.columns) - 6}")  # Subtract date, close_price, did_crash, divergence_type
    
    return df

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING FOR ML")
    print("=" * 70)
    
    # 1. LOAD DATA
    print("\n📥 Loading historical_data.csv...")
    try:
        df = pd.read_csv('historical_data.csv')
        print(f"  ✓ Loaded {len(df)} rows")
    except FileNotFoundError:
        print("  Error: historical_data.csv not found!")
        print("  Run backfill_historical.py first.")
        return
    
    # 2. VALIDATE DATA
    print("\n🔍 Validating data...")
    required_cols = ['date', 'sentiment', 'market_return', 'vix', 'close_price', 'did_crash']
    if not all(col in df.columns for col in required_cols):
        print(f"  Error: Missing required columns!")
        print(f"  Expected: {required_cols}")
        print(f"  Found: {list(df.columns)}")
        return
    print("  ✓ All required columns present")
    
    # 3. CREATE FEATURES
    print("\n🔧 Creating features...")
    df = create_features(df)
    
    # 4. SELECT FEATURE COLUMNS (29 features)
    feature_cols = [
        # Base (5)
        'sentiment', 'market_return', 'vix', 'divergence_magnitude', 'anomaly_score',
        
        # Anomaly flags (4)
        'is_anomaly', 'is_high_risk', 'is_medium_risk', 'is_low_risk',
        
        # Divergence types (4)
        'div_positive_news_negative_market', 'div_negative_news_positive_market', 
        'div_moderate_divergence', 'div_none',
        
        # VIX categories (5)
        'vix_high', 'vix_panic', 'vix_elevated_high', 'vix_elevated', 'vix_normal',
        
        # Rolling averages (6)
        'sentiment_3d_avg', 'market_3d_avg', 'vix_3d_avg',
        'sentiment_7d_avg', 'market_7d_avg', 'vix_7d_avg',
        
        # Interactions (3)
        'sentiment_market_interaction', 'vix_divergence_interaction', 'sentiment_vix_interaction',
        
        # Lags (3)
        'prev_sentiment', 'prev_market', 'prev_vix'
    ]
    
    # 5. CREATE ML DATASET
    print("\n Creating ML dataset...")
    ml_data = df[['date'] + feature_cols + ['did_crash']].copy()
    
    # 6. TRAIN/TEST SPLIT (80/20)
    print("\n Splitting train/test...")
    split_idx = int(len(ml_data) * 0.8)
    train_data = ml_data[:split_idx]
    test_data = ml_data[split_idx:]
    
    print(f"  Training samples:   {len(train_data)}")
    print(f"  Test samples:       {len(test_data)}")
    
    # 7. SAVE FILES
    print("\n Saving files...")
    ml_data.to_csv('training-data.csv', index=False)
    print(f"  ✓ training-data.csv ({len(ml_data)} rows, {len(feature_cols)} features)")
    
    # Save train/test without date column (for SageMaker)
    train_data[feature_cols + ['did_crash']].to_csv('train.csv', index=False, header=False)
    print(f"  ✓ train.csv ({len(train_data)} rows)")
    
    test_data[feature_cols + ['did_crash']].to_csv('test.csv', index=False, header=False)
    print(f"  ✓ test.csv ({len(test_data)} rows)")
    
    # 8. SUMMARY STATISTICS
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total days:         {len(ml_data)}")
    print(f"Features:           {len(feature_cols)}")
    print(f"Crashes:            {ml_data['did_crash'].sum()} ({ml_data['did_crash'].sum()/len(ml_data)*100:.1f}%)")
    print(f"Anomalies:          {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")
    print(f"High-risk:          {df['is_high_risk'].sum()}")
    print(f"Medium-risk:        {df['is_medium_risk'].sum()}")
    
    print("\n Feature engineering complete!")
    print("\nNext step: Upload train.csv and test.csv to S3, then run train_model.py")

if __name__ == "__main__":
    main()
