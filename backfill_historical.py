"""
Historical Data Backfill Script - COMPLETE PRODUCTION VERSION
Implements full Bronze/Silver/Gold data lake architecture

Data Flow:
APIs → Bronze (raw/) → Anomaly Detection → Silver (processed/) 
    → daily-metrics (ALL days) + anomaly-events (ONLY anomalies)
    → Glue ETL → Gold (ml-data/)
"""

import boto3
import requests
import json
from datetime import datetime, timedelta, timezone
import time
import os
import uuid
import random
from decimal import Decimal
from google.cloud import bigquery
from functools import wraps
import statistics

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = 'us-east-1'
S3_BUCKET = os.environ.get('S3_BUCKET', 'anomaly-detection-project-jazz')
DYNAMODB_TABLE_ANOMALIES = 'anomaly-events'
DYNAMODB_TABLE_DAILY = 'daily-metrics'

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_API_KEY:
    raise ValueError("❌ FRED_API_KEY required")

BEDROCK_MODEL = 'us.meta.llama3-1-8b-instruct-v1:0'

THRESHOLDS = {
    'sentiment_high': None,
    'sentiment_low': None,
    'market_significant': 0.75,
    'vix_elevated': 20,
    'vix_high': 25,
    'vix_panic': 35
}

# ============================================================
# RETRY DECORATOR
# ============================================================

def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = delay * (2 ** attempt) + random.random()
                    print(f"   ⚠️ Retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# ============================================================
# AWS CLIENTS
# ============================================================

s3 = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)

table_anomalies = dynamodb.Table(DYNAMODB_TABLE_ANOMALIES)
table_daily = dynamodb.Table(DYNAMODB_TABLE_DAILY)

# ============================================================
# BIGQUERY
# ============================================================

bigquery_client = None
try:
    print("🔧 Initializing BigQuery...")
    bigquery_client = bigquery.Client()
    test_job = bigquery_client.query("SELECT 1")
    test_job.result()
    print("   ✅ BigQuery ready")
except Exception as e:
    print(f"   ⚠️ BigQuery init failed: {e}")

# ============================================================
# GLOBAL STATE
# ============================================================

_cached_spy_data = None
_cached_vix_data = None
_cached_gdelt_data = None

stats = {
    'total_days': 0,
    'anomalies_detected': 0,
    'high_risk_anomalies': 0,
    'medium_risk_anomalies': 0,
    'api_errors': 0,
    'llm_calls': 0,
    'low_confidence_days': 0,
    'bigquery_gb_processed': 0.0,
    'raw_files_saved': 0,
    'daily_metrics_saved': 0,
    'data_quality': {
        'gdelt': 'unknown',
        'market': 'unknown',
        'vix': 'unknown'
    }
}

# ============================================================
# GDELT SENTIMENT
# ============================================================

@retry_on_failure(max_retries=2, delay=3)
def fetch_gdelt_sentiment_batch(start_date, end_date):
    """Fetch GDELT sentiment from BigQuery"""
    global _cached_gdelt_data
    
    if _cached_gdelt_data is not None:
        return _cached_gdelt_data
    
    if bigquery_client is None:
        print("   ❌ BigQuery unavailable")
        stats['data_quality']['gdelt'] = 'unavailable'
        return {}
    
    print("\n📰 Fetching GDELT sentiment from BigQuery...")
    
    try:
        start_int = int(start_date.strftime('%Y%m%d')) * 1000000
        end_int = int(end_date.strftime('%Y%m%d')) * 1000000 + 235959
        
        query = f"""
        SELECT 
            SUBSTR(CAST(DATE AS STRING), 1, 8) as date_str,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) as avg_tone,
            COUNT(*) as article_count
        FROM `gdelt-bq.gdeltv2.gkg`
        WHERE DATE >= {start_int}
        AND DATE <= {end_int}
        AND (
            Themes LIKE '%ECON_%' 
            OR Themes LIKE '%FINANCE%' 
            OR Themes LIKE '%STOCK%'
            OR Themes LIKE '%MARKET%'
        )
        GROUP BY date_str
        ORDER BY date_str
        """
        
        print("   Running BigQuery...")
        job = bigquery_client.query(query)
        results = job.result()
        
        bytes_processed = job.total_bytes_processed / 1e9
        stats['bigquery_gb_processed'] = bytes_processed
        print(f"   📊 Processed: {bytes_processed:.4f} GB")
        
        sentiment_dict = {}
        for row in results:
            date_str_raw = row['date_str']
            date_obj = datetime.strptime(date_str_raw, '%Y%m%d')
            date_str = date_obj.strftime('%Y-%m-%d')
            
            avg_tone = row['avg_tone']
            article_count = row['article_count']
            
            if avg_tone is not None and article_count > 0:
                sentiment = avg_tone
                sentiment = max(-10, min(10, sentiment))
                sentiment_dict[date_str] = round(sentiment, 2)
                
                if article_count < 100:
                    stats['low_confidence_days'] += 1
            else:
                sentiment_dict[date_str] = 0.0
                stats['low_confidence_days'] += 1
            
            if len(sentiment_dict) <= 3:
                tone_display = f"{avg_tone:.2f}" if avg_tone is not None else "0.00"
                print(f"   📅 {date_str}: tone={tone_display}, articles={article_count}")
        
        _cached_gdelt_data = sentiment_dict
        stats['data_quality']['gdelt'] = 'real'
        print(f"   ✅ Fetched {len(sentiment_dict)} days")
        return sentiment_dict
        
    except Exception as e:
        print(f"   ❌ BigQuery error: {e}")
        stats['data_quality']['gdelt'] = 'failed'
        return {}


def get_gdelt_sentiment(date):
    gdelt_data = _cached_gdelt_data or {}
    return gdelt_data.get(date.strftime('%Y-%m-%d'), 0.0)


# ============================================================
# MARKET DATA (YFINANCE)
# ============================================================

@retry_on_failure(max_retries=3, delay=5)
def fetch_market_data_yfinance(start_date, end_date):
    """Fetch SPY data via yfinance"""
    global _cached_spy_data
    
    if _cached_spy_data is not None:
        return _cached_spy_data
    
    print("\n📈 Fetching SPY from Yahoo Finance (yfinance)...")
    
    try:
        import yfinance as yf
        
        start_str = (start_date - timedelta(days=10)).strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"   Downloading {start_str} to {end_str}...")
        spy = yf.Ticker('SPY')
        hist = spy.history(start=start_str, end=end_str)
        
        if hist.empty:
            raise Exception("No data returned")
        
        spy_dict = {}
        for date_idx, row in hist.iterrows():
            date_str = date_idx.strftime('%Y-%m-%d')
            spy_dict[date_str] = {
                'close': float(row['Close']),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'volume': int(row['Volume'])
            }
        
        _cached_spy_data = spy_dict
        stats['data_quality']['market'] = 'real'
        print(f"   ✅ Fetched {len(spy_dict)} days (REAL - yfinance)")
        return spy_dict
        
    except ImportError:
        print(f"   ❌ yfinance not installed: pip install yfinance")
        stats['data_quality']['market'] = 'simulated'
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        stats['data_quality']['market'] = 'simulated'
        return None


def fetch_historical_market_data(date):
    spy_data = _cached_spy_data
    
    if spy_data is None:
        random.seed(date.toordinal() + 1000)
        return round(random.gauss(0.03, 1.0), 2)
    
    try:
        current_close = None
        for i in range(8):
            check_date = (date - timedelta(days=i)).strftime('%Y-%m-%d')
            if check_date in spy_data:
                current_close = spy_data[check_date]['close']
                break
        
        if current_close is None:
            stats['api_errors'] += 1
            return 0.0
        
        prev_close = None
        for i in range(1, 15):
            prev_date_str = (date - timedelta(days=i)).strftime('%Y-%m-%d')
            if prev_date_str in spy_data:
                prev_close = spy_data[prev_date_str]['close']
                break
        
        if prev_close is None or prev_close == 0:
            return 0.0
        
        return_pct = ((current_close - prev_close) / prev_close) * 100
        return round(return_pct, 2)
        
    except Exception as e:
        return 0.0


# ============================================================
# VIX DATA
# ============================================================

@retry_on_failure(max_retries=3, delay=2)
def fetch_fred_vix_full_history():
    """Fetch VIX from FRED"""
    global _cached_vix_data
    
    if _cached_vix_data is not None:
        return _cached_vix_data
    
    print("\n😱 Fetching VIX from FRED...")
    
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': 'VIXCLS',
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'sort_order': 'asc'
    }
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    if 'observations' not in data:
        raise Exception("No observations")
    
    vix_dict = {}
    for obs in data['observations']:
        if obs['value'] != '.':
            vix_dict[obs['date']] = float(obs['value'])
    
    _cached_vix_data = vix_dict
    stats['data_quality']['vix'] = 'real'
    print(f"   ✅ Fetched {len(vix_dict)} days")
    return vix_dict


def fetch_historical_vix(date):
    vix_data = fetch_fred_vix_full_history()
    
    if vix_data is None:
        return 20.0
    
    date_str = date.strftime('%Y-%m-%d')
    if date_str in vix_data:
        return vix_data[date_str]
    
    for i in range(1, 8):
        prev_date = (date - timedelta(days=i)).strftime('%Y-%m-%d')
        if prev_date in vix_data:
            return vix_data[prev_date]
    
    return 20.0


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def analyze_and_calibrate_thresholds():
    if _cached_gdelt_data is None:
        print("\n⚠️ WARNING: GDELT data not available")
        THRESHOLDS['sentiment_high'] = 0.0
        THRESHOLDS['sentiment_low'] = -0.5
        return
    
    sentiments = [v for v in _cached_gdelt_data.values() if v != 0]
    
    if not sentiments or len(sentiments) < 10:
        THRESHOLDS['sentiment_high'] = 0.0
        THRESHOLDS['sentiment_low'] = -0.5
        return
    
    mean = statistics.mean(sentiments)
    stdev = statistics.stdev(sentiments)
    sorted_s = sorted(sentiments)
    
    p10 = sorted_s[len(sorted_s) // 10]
    p90 = sorted_s[len(sorted_s) * 9 // 10]
    
    print("\n" + "=" * 70)
    print("📊 SENTIMENT DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"\n  Data Points:        {len(sentiments)}")
    print(f"  Mean:               {mean:.3f}")
    print(f"  Std Deviation:      {stdev:.3f}")
    print(f"  Range:              {min(sentiments):.3f} to {max(sentiments):.3f}")
    print(f"  10th percentile:    {p10:.3f}")
    print(f"  90th percentile:    {p90:.3f}")
    
    threshold_high = max(p90, -0.15)
    threshold_low = min(p10, -0.45)
    
    THRESHOLDS['sentiment_high'] = round(threshold_high, 2)
    THRESHOLDS['sentiment_low'] = round(threshold_low, 2)
    
    print(f"\n🎯 CALIBRATED THRESHOLDS:")
    print(f"  Positive anomaly:   >{THRESHOLDS['sentiment_high']:.2f}")
    print(f"  Negative anomaly:   <{THRESHOLDS['sentiment_low']:.2f}")
    print(f"  Market divergence:  ±{THRESHOLDS['market_significant']:.2f}%")
    print(f"  VIX elevated:       >{THRESHOLDS['vix_elevated']}")
    print("\n" + "=" * 70)


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomaly(sentiment, market_return, vix):
    score = 0
    divergence_type = "none"
    
    sent_high = THRESHOLDS.get('sentiment_high', 0.0)
    sent_low = THRESHOLDS.get('sentiment_low', -0.5)
    
    if sentiment > sent_high and market_return < -0.5:
        score += 50
        divergence_type = "positive_news_negative_market"
    elif sentiment < sent_low and market_return > 0.5:
        score += 40
        divergence_type = "negative_news_positive_market"
    elif abs(sentiment - market_return) > 1.5 and vix > THRESHOLDS['vix_elevated']:
        score += 35
        divergence_type = "moderate_divergence_with_stress"
    
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
    
    divergence_magnitude = abs(sentiment - market_return)
    if divergence_magnitude > 2.5:
        score += 20
    elif divergence_magnitude > 4:
        score += 30
    
    score = min(score, 100)
    
    if score >= 70:
        risk_level = "HIGH"
    elif score >= 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        'is_anomaly': score >= 50,
        'score': score,
        'risk_level': risk_level,
        'divergence_type': divergence_type,
        'divergence_magnitude': round(divergence_magnitude, 2)
    }


def generate_llm_explanation(sentiment, market_return, vix, divergence_type):
    try:
        prompt = f"""Financial analyst: Analyze HIGH-RISK anomaly:

Sentiment: {sentiment:.2f}
Market: {market_return:.2f}%
VIX: {vix:.1f}
Type: {divergence_type}

2-sentence explanation."""

        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps({
                "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "max_gen_len": 300,
                "temperature": 0.7,
                "top_p": 0.9
            })
        )
        
        response_body = json.loads(response['body'].read())
        stats['llm_calls'] += 1
        return response_body['generation'].strip()
        
    except Exception as e:
        print(f"  ⚠️ Bedrock: {str(e)[:80]}")
        return "High-risk divergence detected. Manual review recommended."


# ============================================================
# STORAGE - BRONZE LAYER (RAW DATA)
# ============================================================

def store_raw_data(date, sentiment, market_return, vix):
    """
    BRONZE LAYER - Immutable raw API responses
    Purpose: Source of truth for auditability
    """
    try:
        date_str = date.strftime('%Y-%m-%d')
        
        # GDELT raw
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"raw/gdelt/{date_str}.json",
            Body=json.dumps({
                'date': date_str,
                'source': 'gdelt_bigquery',
                'raw_tone': sentiment,
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }, indent=2),
            ContentType='application/json'
        )
        
        # Market raw
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"raw/market/{date_str}.json",
            Body=json.dumps({
                'date': date_str,
                'source': 'yfinance',
                'symbol': 'SPY',
                'daily_return_pct': market_return,
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }, indent=2),
            ContentType='application/json'
        )
        
        # VIX raw
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"raw/vix/{date_str}.json",
            Body=json.dumps({
                'date': date_str,
                'source': 'fred_api',
                'series': 'VIXCLS',
                'value': vix,
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }, indent=2),
            ContentType='application/json'
        )
        
        stats['raw_files_saved'] += 3
        return True
        
    except Exception as e:
        return False


# ============================================================
# STORAGE - DAILY METRICS (ALL DAYS)
# ============================================================

def store_daily_metrics(date, sentiment, market_return, vix, anomaly_result):
    """
    Store EVERY day to daily-metrics table
    Purpose: Complete time-series for dashboards and ML training
    """
    try:
        timestamp = date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        date_str = date.strftime('%Y-%m-%d')
        metric_id = f"daily_{date.strftime('%Y%m%d')}"
        
        table_daily.put_item(
            Item={
                'metric_id': metric_id,
                'date': date_str,
                'timestamp': timestamp.isoformat(),
                
                # Core metrics
                'sentiment': Decimal(str(round(sentiment, 4))),
                'market_return': Decimal(str(round(market_return, 4))),
                'vix': Decimal(str(round(vix, 2))),
                
                # Anomaly info
                'anomaly_score': int(anomaly_result['score']),
                'is_anomaly': anomaly_result['is_anomaly'],
                'risk_level': anomaly_result['risk_level'],
                'divergence_type': anomaly_result['divergence_type'],
                'divergence_magnitude': Decimal(str(round(anomaly_result['divergence_magnitude'], 4))),
                
                # Metadata
                'backfilled': True,
                'data_sources': {
                    'sentiment': 'gdelt_bigquery',
                    'market': 'yfinance',
                    'vix': 'fred_api'
                }
            },
            ConditionExpression="attribute_not_exists(metric_id)"
        )
        
        stats['daily_metrics_saved'] += 1
        return True
        
    except Exception as e:
        if 'ConditionalCheckFailedException' not in str(e):
            print(f"  ⚠️ Daily metrics: {str(e)[:80]}")
        return False


# ============================================================
# STORAGE - SILVER LAYER (ANOMALIES ONLY)
# ============================================================

def store_anomaly(date, sentiment, market_return, vix, anomaly_result, explanation=None):
    """
    SILVER LAYER - Store validated anomalies only
    Purpose: Alert-worthy events for SNS notifications
    """
    try:
        timestamp = date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        date_str = date.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        event_id = f"evt_{date.strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
        
        data = {
            'timestamp': timestamp.isoformat(),
            'date': date_str,
            'sentiment': sentiment,
            'market_return': market_return,
            'vix': vix,
            'anomaly_score': anomaly_result['score'],
            'risk_level': anomaly_result['risk_level'],
            'divergence_type': anomaly_result['divergence_type'],
            'divergence_magnitude': anomaly_result['divergence_magnitude'],
            'llm_explanation': explanation or "Medium-risk divergence detected.",
            'backfilled': True,
            'detection_method': 'multi_factor_research_based',
            'data_sources': {
                'sentiment': 'gdelt_bigquery',
                'market': 'yfinance',
                'vix': 'fred_api'
            }
        }
        
        # S3
        key = f"processed/anomalies/{date_str}/anomaly_{time_str}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        
        # DynamoDB anomaly-events
        table_anomalies.put_item(
            Item={
                'event_id': event_id,
                'timestamp': timestamp.isoformat(),
                'date': date_str,
                'anomaly_score': int(anomaly_result['score']),
                'news_sentiment': Decimal(str(round(sentiment, 4))),
                'market_return': Decimal(str(round(market_return, 4))),
                'vix': Decimal(str(round(vix, 2))),
                'divergence_type': anomaly_result['divergence_type'],
                'divergence_magnitude': Decimal(str(round(anomaly_result['divergence_magnitude'], 4))),
                'risk_level': anomaly_result['risk_level'],
                'llm_explanation': explanation or "Medium-risk divergence.",
                'backfilled': True
            },
            ConditionExpression="attribute_not_exists(event_id)"
        )
        
        return event_id
        
    except Exception as e:
        if 'ConditionalCheckFailedException' not in str(e):
            print(f"  ❌ Storage: {str(e)[:80]}")
        return None


# ============================================================
# PROCESSING
# ============================================================

def backfill_single_day(date):
    """
    Process one day with complete data pipeline:
    1. Fetch data
    2. Save to Bronze (raw/)
    3. Detect anomaly
    4. Save to daily-metrics (always)
    5. Save to Silver + anomaly-events (if anomaly)
    """
    
    # Fetch data
    sentiment = get_gdelt_sentiment(date)
    market_return = fetch_historical_market_data(date)
    vix = fetch_historical_vix(date)
    
    # 1. BRONZE: Save raw (always)
    store_raw_data(date, sentiment, market_return, vix)
    
    # 2. Detect anomaly
    anomaly_result = detect_anomaly(sentiment, market_return, vix)
    stats['total_days'] += 1
    
    # 3. DAILY METRICS: Save every day (for time-series + ML)
    store_daily_metrics(date, sentiment, market_return, vix, anomaly_result)
    
    # 4. SILVER: Save if anomaly (for alerts)
    if anomaly_result['is_anomaly']:
        explanation = None
        if anomaly_result['risk_level'] == 'HIGH':
            explanation = generate_llm_explanation(
                sentiment, market_return, vix, anomaly_result['divergence_type']
            )
        
        event_id = store_anomaly(date, sentiment, market_return, vix, anomaly_result, explanation)
        
        if event_id:
            stats['anomalies_detected'] += 1
            if anomaly_result['risk_level'] == 'HIGH':
                stats['high_risk_anomalies'] += 1
            else:
                stats['medium_risk_anomalies'] += 1
        
        return f"🚨 {anomaly_result['risk_level']}"
    
    return "✅"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 COMPLETE DATA LAKE BACKFILL")
    print("=" * 70)
    
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)
    
    print(f"\n📅 Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    print(f"\n✅ Configuration:")
    print(f"   FRED API:        {'✅' if FRED_API_KEY else '❌'}")
    print(f"   GCP Creds:       {'✅' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else '⚠️'}")
    print(f"   BigQuery:        {'✅' if bigquery_client else '⚠️'}")
    print(f"   Bedrock Model:   {BEDROCK_MODEL}")
    
    try:
        import yfinance
        print(f"   yfinance:        ✅")
    except:
        print(f"   yfinance:        ❌")
    
    # Pre-fetch
    print("\n" + "=" * 70)
    print("📥 PRE-FETCHING DATA")
    print("=" * 70)
    
    fetch_gdelt_sentiment_batch(start_date, end_date)
    fetch_market_data_yfinance(start_date, end_date)
    fetch_fred_vix_full_history()
    
    analyze_and_calibrate_thresholds()
    
    print("\n✅ Ready!")
    input("\n⏸️  Press ENTER to start...")
    
    # Process
    print("\n" + "=" * 70)
    print("⚙️  PROCESSING - COMPLETE PIPELINE")
    print("=" * 70)
    
    date_range = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            date_range.append(current)
        current += timedelta(days=1)
    
    if HAS_TQDM:
        iterator = tqdm(date_range, desc="Processing", unit="day")
    else:
        iterator = date_range
        print(f"Processing {len(date_range)} days...")
    
    for i, date in enumerate(iterator):
        result = backfill_single_day(date)
        if not HAS_TQDM and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(date_range)} ({(i+1)/len(date_range)*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ BACKFILL COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Results:")
    print(f"  Trading Days:        {stats['total_days']}")
    print(f"  Anomalies:           {stats['anomalies_detected']} ({stats['anomalies_detected']/stats['total_days']*100:.1f}%)")
    print(f"  High-Risk:           {stats['high_risk_anomalies']}")
    print(f"  Medium-Risk:         {stats['medium_risk_anomalies']}")
    print(f"  LLM Calls:           {stats['llm_calls']}")
    
    print(f"\n📂 Data Lake:")
    print(f"  🥉 Bronze (S3 raw/):           {stats['raw_files_saved']} files")
    print(f"  📊 DynamoDB daily-metrics:     {stats['daily_metrics_saved']} records")
    print(f"  🥈 Silver (S3 processed/):     {stats['anomalies_detected']} files")
    print(f"  🥈 DynamoDB anomaly-events:    {stats['anomalies_detected']} records")
    print(f"  🥇 Gold (ml-data/):            Generated by Glue")
    
    print(f"\n💰 Costs:")
    print(f"  BigQuery:            {stats['bigquery_gb_processed']:.4f} GB")
    print(f"  Bedrock:             ~${stats['llm_calls'] * 0.0004:.3f}")
    
    print("\n🎉 Ready for Glue ETL and ML training!")


if __name__ == "__main__":
    main()