import boto3
import json
from datetime import datetime, timedelta, timezone
import os
from decimal import Decimal
import uuid

# AWS Clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
bedrock = boto3.client('bedrock-runtime')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Config
S3_BUCKET = os.environ.get('S3_BUCKET', 'anomaly-detection-project-jazz')
DYNAMODB_TABLE_ANOMALIES = 'anomaly-events'
DYNAMODB_TABLE_DAILY = 'daily-metrics'
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
SAGEMAKER_ENDPOINT = 'anomaly-crash-predictor'

table_anomalies = dynamodb.Table(DYNAMODB_TABLE_ANOMALIES)
table_daily = dynamodb.Table(DYNAMODB_TABLE_DAILY)

BEDROCK_MODEL = 'us.meta.llama3-1-8b-instruct-v1:0'

THRESHOLDS = {
    'sentiment_high': -0.06,
    'sentiment_low': -0.45,
    'market_significant': 0.75,
    'vix_elevated': 20,
    'vix_high': 25,
    'vix_panic': 35
}

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_gdelt_sentiment():
    """
    Fetch sentiment from GDELT CSV (free, no quota limits)
    Uses latest 15-minute update from GDELT v2
    Better than BigQuery: fresher data, unlimited, zero cost
    """
    try:
        from urllib.request import urlopen
        import zipfile
        import csv
        import io
        
        print(" Fetching latest GDELT CSV...")
        
        # Get latest GKG file URL from GDELT
        with urlopen('http://data.gdeltproject.org/gdeltv2/lastupdate.txt', timeout=10) as response:
            lastupdate = response.read().decode('utf-8')
        
        # Find the .gkg.csv.zip URL (third column)
        gkg_url = None
        for line in lastupdate.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3 and 'gkg.csv.zip' in parts[2]:
                gkg_url = parts[2]
                break
        
        if not gkg_url:
            print(" No GKG file found, using 0.0")
            return 0.0
        
        print(f" Downloading: {gkg_url.split('/')[-1]}")
        
        # Download and unzip CSV
        with urlopen(gkg_url, timeout=30) as zip_response:
            zip_data = zip_response.read()
        
        # Parse CSV from ZIP
        tones = []
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as csv_file:
                # GDELT GKG is tab-delimited
                reader = csv.reader(io.TextIOWrapper(csv_file, encoding='utf-8', errors='ignore'), delimiter='\t')
                
                for row in reader:
                    try:
                        # Column 7 (index 6) = Themes
                        # Column 16 (index 15) = V2Tone
                        if len(row) > 15:
                            themes = row[6] if len(row) > 6 else ''
                            tone_str = row[15]
                            
                            # Filter for economic/market themes
                            if themes and ('ECON' in themes or 'FINANCE' in themes or 'STOCK' in themes or 
               'MARKET' in themes or 'TAX' in themes or 'WB_' in themes or 
               'GENERAL_BUSINESS' in themes or 'WORLDBANK' in themes):
                                if tone_str and tone_str.strip():
                                    # V2Tone format: "tone,positive,negative,polarity,..."
                                    tone_value = float(tone_str.split(',')[0])
                                    tones.append(tone_value)
                    except (ValueError, IndexError):
                        continue
        
        if tones:
            avg_tone = sum(tones) / len(tones)
            sentiment = max(-10, min(10, avg_tone))
            print(f" GDELT CSV: {sentiment:.2f} (from {len(tones)} articles)")
            return round(sentiment, 2)
        else:
            print(f" No economic articles found, using 0.0")
            return 0.0
        
    except Exception as e:
        print(f" GDELT CSV error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def fetch_market_data():
    """Yahoo Finance - using urllib (built-in)"""
    try:
        from urllib.request import urlopen, Request
        from urllib.parse import urlencode
        
        params = {'range': '5d', 'interval': '1d'}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?{urlencode(params)}"
        
        req = Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        result = data['chart']['result'][0]
        closes = result['indicators']['quote'][0]['close']
        valid_closes = [c for c in closes if c is not None]
        
        if len(valid_closes) >= 2:
            current = valid_closes[-1]
            prev = valid_closes[-2]
            daily_return = ((current - prev) / prev) * 100
            
            print(f"   SPY: ${current:.2f}, return: {daily_return:.2f}%")
            return {'return': round(daily_return, 2), 'close': round(current, 2)}
        
        return {'return': 0.0, 'close': 0.0}
        
    except Exception as e:
        print(f" Market error: {e}")
        return {'return': 0.0, 'close': 0.0}


def fetch_vix():
    """FRED API - using urllib (built-in)"""
    try:
        from urllib.request import urlopen
        from urllib.parse import urlencode
        
        fred_key = os.environ.get('FRED_API_KEY', '')
        if not fred_key:
            return 20.0
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        params = {
            'series_id': 'VIXCLS',
            'api_key': fred_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'sort_order': 'desc',
            'limit': '10'
        }
        
        url = f"https://api.stlouisfed.org/fred/series/observations?{urlencode(params)}"
        
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        if 'observations' in data:
            for obs in data['observations']:
                if obs['value'] != '.':
                    vix = float(obs['value'])
                    print(f"   VIX: {vix}")
                    return vix
        
        return 20.0
        
    except Exception as e:
        print(f" VIX error: {e}")
        return 20.0


# ============================================================
# STORAGE
# ============================================================

def store_raw_data(timestamp, sentiment, market_data, vix):
    """BRONZE LAYER"""
    try:
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        raw_data = {
            'timestamp': timestamp.isoformat(),
            'date': date_str,
            'sources': {
                'gdelt': {'sentiment': sentiment},
                'market': {'return': market_data['return'], 'close': market_data['close']},
                'vix': {'value': vix}
            }
        }
        
        key = f"raw/live/{date_str}/{time_str}.json"
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(raw_data, indent=2),
            ContentType='application/json'
        )
        
        print(f" Raw data stored to S3")
        return key
        
    except Exception as e:
        print(f" S3 error: {e}")
        return None

def store_daily_metrics(timestamp, sentiment, market_return, vix, anomaly_result):
    """Store to daily-metrics"""
    try:
        date_str = timestamp.strftime('%Y-%m-%d')
        
        table_daily.put_item(
            Item={
                'date': date_str,
                'timestamp': timestamp.isoformat(),
                'sentiment': Decimal(str(round(sentiment, 4))),
                'market_return': Decimal(str(round(market_return, 4))),
                'vix': Decimal(str(round(vix, 2))),
                'anomaly_score': int(anomaly_result['score']),
                'is_anomaly': anomaly_result['is_anomaly'],
                'risk_level': anomaly_result['risk_level'],
                'divergence_type': anomaly_result['divergence_type'],
                'divergence_magnitude': Decimal(str(round(anomaly_result['divergence_magnitude'], 4))),
                'backfilled': False,
                'data_source': 'gdelt_csv'
            }
        )
        
        print(f" Daily metrics saved")
        return True
        
    except Exception as e:
        print(f" DynamoDB error: {e}")
        return False


def detect_anomaly(sentiment, market_return, vix):
    """Rule-based detection"""
    score = 0
    divergence_type = "none"
    
    if sentiment > THRESHOLDS['sentiment_high'] and market_return < -0.5:
        score += 50
        divergence_type = "positive_news_negative_market"
    elif sentiment < THRESHOLDS['sentiment_low'] and market_return > 0.5:
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


def prepare_features_for_ml(sentiment, market_return, vix, divergence_magnitude, anomaly_score):
    """Prepare 29 features"""
    features = [
        sentiment, market_return, vix, divergence_magnitude, anomaly_score,
        1 if anomaly_score >= 50 else 0,
        1 if anomaly_score >= 70 else 0,
        1 if 50 <= anomaly_score < 70 else 0,
        1 if anomaly_score < 50 else 0,
        0, 0, 0,
        1 if vix > 25 else 0,
        1 if vix > 35 else 0,
        1 if 25 < vix <= 35 else 0,
        1 if 20 < vix <= 25 else 0,
        1 if vix <= 20 else 0,
        sentiment, vix, market_return,
        sentiment, vix, market_return,
        sentiment * market_return,
        vix * divergence_magnitude,
        sentiment * vix,
        sentiment, market_return, vix
    ]
    return features


def get_ml_prediction(features):
    """Call SageMaker"""
    try:
        payload = ','.join([str(f) for f in features])
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='text/csv',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # Handle response format
        if isinstance(result, (int, float)):
            crash_probability = float(result)
        elif isinstance(result, list) and len(result) > 0:
            crash_probability = float(result[0])
        elif isinstance(result, dict) and 'predictions' in result:
            crash_probability = float(result['predictions'][0])
        else:
            return {
                'crash_probability': None,
                'crash_predicted': False,
                'confidence': 'unavailable'
            }
        
        print(f"   ML: {crash_probability:.2%}")
        
        return {
            'crash_probability': crash_probability,
            'crash_predicted': crash_probability > 0.5,
            'confidence': 'high' if crash_probability > 0.7 or crash_probability < 0.3 else 'medium'
        }
        
    except Exception as e:
        print(f" ML error: {e}")
        return {
            'crash_probability': None,
            'crash_predicted': False,
            'confidence': 'unavailable'
        }

def generate_llm_explanation(sentiment, market_return, vix, anomaly_result, ml_prediction):
    """
    Generate LLM explanation using Amazon Bedrock
    Only called for high-risk anomalies (score >= 50)
    """
    try:
        # Format crash probability safely
        crash_prob_str = f"{ml_prediction['crash_probability']:.1%}" if ml_prediction['crash_probability'] is not None else "N/A"
        
        # Create prompt
        prompt = f"""You are a financial analyst. Explain this market anomaly concisely in 2-3 sentences.

Metrics:
- News Sentiment: {sentiment} (scale: -10 to +10)
- Market Return: {market_return}%
- VIX (Fear Index): {vix}
- Anomaly Score: {anomaly_result['score']}/100
- Divergence Type: {anomaly_result['divergence_type']}
- ML Crash Probability: {crash_prob_str}

Explain why this is concerning and what it signals."""

        # Call Bedrock
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                'prompt': prompt,
                'max_gen_len': 200,
                'temperature': 0.7,
                'top_p': 0.9
            })
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        explanation = response_body.get('generation', 'Unable to generate explanation')
        
        print(f" Bedrock explanation generated")
        return explanation
        
    except Exception as e:
        print(f" Bedrock error: {e}")
        import traceback
        traceback.print_exc()
        return "LLM explanation unavailable"

def store_anomaly_event(timestamp, sentiment, market_return, vix, anomaly_result, ml_prediction, llm_explanation):
    """Store anomaly to S3 Silver layer and DynamoDB"""
    try:
        event_id = f"evt_{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        anomaly_data = {
            'event_id': event_id,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'sentiment': sentiment,
            'market_return': market_return,
            'vix': vix,
            'anomaly_score': anomaly_result['score'],
            'risk_level': anomaly_result['risk_level'],
            'divergence_type': anomaly_result['divergence_type'],
            'ml_crash_probability': ml_prediction['crash_probability'],
            'llm_explanation': llm_explanation,
            'backfilled': False
        }
        
        # Store to S3 Silver
        s3_key = f"processed/anomalies/{timestamp.strftime('%Y-%m-%d')}/anomaly_{timestamp.strftime('%H-%M-%S')}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(anomaly_data, indent=2),
            ContentType='application/json'
        )
        
        # Store to DynamoDB anomaly-events
        table_anomalies.put_item(
            Item={
                'event_id': event_id,
                'timestamp': anomaly_data['timestamp'],
                'date': anomaly_data['date'],
                'sentiment': Decimal(str(sentiment)),
                'market_return': Decimal(str(market_return)),
                'vix': Decimal(str(vix)),
                'anomaly_score': anomaly_result['score'],
                'risk_level': anomaly_result['risk_level'],
                'divergence_type': anomaly_result['divergence_type'],
                'ml_crash_probability': Decimal(str(ml_prediction['crash_probability'])) if ml_prediction['crash_probability'] is not None else Decimal('0'),
                'llm_explanation': llm_explanation or 'N/A',
                'backfilled': False
            }
        )
        
        print(f" Anomaly event stored: {event_id}")
        return event_id
        
    except Exception as e:
        print(f" Anomaly storage error: {e}")
        import traceback
        traceback.print_exc()
        return None

def lambda_handler(event, context):
    """Main handler"""
    
    print("=" * 70)
    print("ANOMALY DETECTOR - GDELT CSV + ML + LLM")
    print("=" * 70)
    
    try:
        timestamp = datetime.now(timezone.utc)
        
        # FETCH
        print("\n Fetching...")
        sentiment = fetch_gdelt_sentiment()
        market_data = fetch_market_data()
        vix = fetch_vix()
        
        market_return = market_data['return']
        
        # STORE RAW DATA TO S3 BRONZE LAYER
        store_raw_data(timestamp, sentiment, market_data, vix)
        
        print(f"\n Metrics:")
        print(f"   Sentiment: {sentiment}")
        print(f"   Market: {market_return}%")
        print(f"   VIX: {vix}")
        
        # DETECT ANOMALY
        anomaly_result = detect_anomaly(sentiment, market_return, vix)
        print(f"\n Score: {anomaly_result['score']}, Risk: {anomaly_result['risk_level']}")
        
        # STORE TO DYNAMODB
        store_daily_metrics(timestamp, sentiment, market_return, vix, anomaly_result)
        
        # ML PREDICTION
        print(f"\n ML prediction...")
        features = prepare_features_for_ml(
            sentiment, market_return, vix,
            anomaly_result['divergence_magnitude'],
            anomaly_result['score']
        )
        
        ml_prediction = get_ml_prediction(features)
        
        if ml_prediction['crash_probability'] is not None:
            print(f"   Crash probability: {ml_prediction['crash_probability']:.2%}")

        # GENERATE LLM EXPLANATION FOR ANOMALIES
        llm_explanation = None
        if anomaly_result['is_anomaly'] and anomaly_result['risk_level'] in ['HIGH', 'MEDIUM']:
            print(f"\n Generating LLM explanation...")
            llm_explanation = generate_llm_explanation(
                sentiment, market_return, vix, 
                anomaly_result, ml_prediction
            )

        # STORE ANOMALY EVENT (with LLM explanation)
        if anomaly_result['is_anomaly']:
            store_anomaly_event(
                timestamp, sentiment, market_return, vix,
                anomaly_result, ml_prediction, llm_explanation
            )
        
        # RETURN
        print(f"\n Complete")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'normal' if not anomaly_result['is_anomaly'] else 'anomaly',
                'sentiment': sentiment,
                'market_return': market_return,
                'vix': vix,
                'ml_crash_prob': ml_prediction['crash_probability'],
                'data_source': 'gdelt_csv'
            })
        }
            
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}