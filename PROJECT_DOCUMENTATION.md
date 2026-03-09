# Market Anomaly Detection System - Technical Documentation

**Serverless ML System with Automated Anomaly Detection and Explainable AI**

---

## Project Overview

### Objective
Built a serverless ML system that detects market anomalies by identifying divergence patterns between news sentiment and market behavior, with automated natural language explanations.

### Problem Statement
Manual market monitoring requires:
- Constant attention to multiple data sources
- Technical expertise to interpret patterns
- Significant time investment for analysis

### Solution Built
An automated system that:
- Ingests data from 3 sources every 30 minutes
- Applies dual-detection (rule-based + ML)
- Generates plain-English explanations via LLM
- Stores results in queryable database
- Operates with minimal manual intervention

### Technology Stack
- **Cloud Platform:** AWS (7 services)
- **ML Framework:** XGBoost via SageMaker
- **LLM:** Amazon Bedrock (Llama 3.1 8B)
- **Languages:** Python 3.12
- **Data Processing:** PySpark (AWS Glue), pandas

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────┬─────────────────┬──────────────────────────────┤
│  GDELT News     │  Yahoo Finance  │  FRED Economic Data          │
│  (Sentiment)    │  (SPY Prices)   │  (VIX Volatility)            │
└────────┬────────┴────────┬────────┴──────────┬───────────────────┘
         │                 │                   │
         └─────────────────┴───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ EventBridge │  ◄── Triggers every 30 min
                    │  Scheduler  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────────────────┐
                    │     AWS LAMBDA FUNCTION             │
                    │  (Orchestration & Processing)       │
                    │                                     │
                    │  1. Fetch data from 3 sources       │
                    │  2. Calculate features              │
                    │  3. Rule-based scoring              │
                    │  4. ML prediction (SageMaker)       │
                    │  5. LLM explanation (Bedrock)       │
                    │  6. Store results                   │
                    └──────┬──────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼─────┐    ┌─────▼──────┐
    │   S3    │      │ DynamoDB  │    │ SageMaker  │
    │ Bronze  │      │  Tables   │    │  Endpoint  │
    │ Silver  │      │           │    │ (XGBoost)  │
    │  Gold   │      │ 2 Tables  │    └────────────┘
    └─────────┘      └───────────┘
```

### Data Flow

**Phase 1: Historical Backfill**
```
Historical Data (262 days)
    ↓
Fetch GDELT, Yahoo Finance, FRED
    ↓
Label crash events (>2% drops)
    ↓
Engineer 29 features
    ↓
Train XGBoost model
    ↓
Deploy to SageMaker endpoint
```

**Phase 2: Real-Time Detection**
```
EventBridge triggers Lambda (every 30 min)
    ↓
Fetch current data from 3 APIs
    ↓
Calculate features in real-time
    ↓
Rule-based anomaly scoring (0-100)
    ↓
ML prediction via SageMaker
    ↓
If anomaly: Generate LLM explanation
    ↓
Store to DynamoDB + S3
```

### Medallion Architecture (Data Lake)

**Bronze Layer (Raw):**
- Path: `s3://bucket/raw/`
- Content: Immutable raw JSON from all sources
- Purpose: Audit trail, reprocessing capability

**Silver Layer (Validated):**
- Path: `s3://bucket/processed/anomalies/`
- Content: Detected anomalies with LLM explanations
- Purpose: High-value events for analysis

**Gold Layer (Analytics-Ready):**
- Path: `s3://bucket/ml-data/`
- Content: Feature-engineered datasets for ML
- Purpose: Training data, model evaluation

---

## AWS Services Used

### 1. AWS Lambda
**Purpose:** Serverless compute for orchestration

**Configuration:**
- Runtime: Python 3.12
- Memory: 512 MB
- Timeout: 180 seconds
- Execution: ~6 seconds average

**Why Lambda:**
- Zero cost when idle
- Automatic scaling
- Event-driven architecture
- No server management

**Key Functions:**
- Data fetching from 3 APIs
- Feature engineering
- SageMaker inference calls
- Bedrock LLM calls
- Data storage orchestration

<p align="center">
  <img src="https://github.com/jasduvv7929/AnomalyDetection/blob/31570eb76d5296a277d724edcc5cfe31efefa5d1/Images/CloudWatch.png" alt="Lamba Monitoring"/>
</p>

---

### 2. Amazon S3
**Purpose:** Data lake storage

**Bucket Structure:**
```
bucket-name/
├── raw/                      # Bronze layer
│   ├── backfill/            # Historical data
│   └── live/                # Real-time data
├── processed/               # Silver layer
│   └── anomalies/           # Detected events
└── ml-data/                 # Gold layer
    ├── training-data.csv
    ├── train/
    └── test/
```

**Why S3:**
- Unlimited scalability
- 99.999999999% durability
- Low cost ($0.023/GB)
- Native AWS integration

<p align="center">
  <img src="https://github.com/jasduvv7929/AnomalyDetection/blob/31570eb76d5296a277d724edcc5cfe31efefa5d1/Images/S3.png" alt="S3 Bucket"/>
</p>

---

### 3. Amazon DynamoDB
**Purpose:** NoSQL operational database

**Tables:**

**Table 1: `daily-metrics`**
```
Partition Key: date (String)
Attributes:
  - timestamp
  - sentiment (Decimal)
  - market_return (Decimal)
  - vix (Decimal)
  - anomaly_score (Number)
  - is_anomaly (Boolean)
  - risk_level (String)
  - divergence_type (String)
  - backfilled (Boolean)
```

**Table 2: `anomaly-events`**
```
Partition Key: event_id (String)
Attributes:
  - timestamp
  - date
  - sentiment, market_return, vix
  - anomaly_score
  - ml_crash_probability (Decimal)
  - llm_explanation (String)
  - backfilled (Boolean)
```

**Why DynamoDB:**
- Sub-second query performance
- Serverless (pay per request)
- Auto-scaling
- Native Decimal type for financial data

<p align="center">
  <img src="https://github.com/jasduvv7929/AnomalyDetection/blob/31570eb76d5296a277d724edcc5cfe31efefa5d1/Images/DynamoDB.png" alt="DynamoDB tables"/>
</p>

<p align="center">
  <img src="https://github.com/jasduvv7929/AnomalyDetection/blob/31570eb76d5296a277d724edcc5cfe31efefa5d1/Images/DynamoDB_1.png" alt="anamoly detection logs"/>
</p>

---

### 4. AWS Glue
**Purpose:** Distributed ETL processing

**Job Configuration:**
- Type: PySpark
- Worker Type: G.1X
- Workers: 2
- Purpose: Feature engineering at scale

**Key Transformations:**
- DynamoDB Decimal → Float casting
- Window functions for rolling averages
- Date partitioning
- Parquet output for ML consumption

**Why Glue:**
- Serverless Spark
- Handles big data transformations
- Native DynamoDB connector
- Production-grade data quality

---

### 5. Amazon SageMaker
**Purpose:** ML model training and deployment

**Training:**
- Instance: ml.m5.xlarge
- Algorithm: XGBoost (built-in)
- Training time: ~5 minutes
- Cost: ~$0.50 per training job

**Endpoint:**
- Instance: ml.t2.medium
- Latency: <2 seconds
- Cost: ~$40/month (can be stopped when not in use)

**Why SageMaker:**
- Managed ML infrastructure
- One-click deployment
- Auto-scaling endpoints
- Built-in algorithms

---

### 6. Amazon EventBridge
**Purpose:** Scheduled automation

**Rule Configuration:**
```
Name: anomaly-detector-30min-v2
Schedule: rate(30 minutes)
Target: Lambda function
State: ENABLED
```

**Why EventBridge:**
- Cron-like scheduling
- Automatic retries
- Event-driven architecture
- Pay per invocation ($0.000001/event)

---

### 7. Amazon Bedrock
**Purpose:** LLM for natural language explanations

**Model:** us.meta.llama3-1-8b-instruct-v1:0

**Configuration:**
- max_gen_len: 200 tokens
- temperature: 0.7
- top_p: 0.9

**Input:** Metrics + context
**Output:** 2-3 sentence explanation

**Why Bedrock:**
- Managed LLM service
- No infrastructure needed
- Multiple model options
- Pay per token (~$0.003 per explanation)

**Sample Output:**
> "This represents a classic divergence where the market is rallying (+2.0%) despite strongly negative news sentiment (-0.60), suggesting investors are ignoring warning signals. The elevated VIX at 28 indicates underlying market stress, and the 73% ML crash probability confirms this is a high-risk scenario."

---

## Data Sources

### 1. GDELT (Global Database of Events, Language, and Tone)
**What it provides:** News sentiment from global media

**Implementation:**
- Source: Direct CSV download (not BigQuery)
- Update frequency: Every 15 minutes
- File format: Tab-delimited GKG (Global Knowledge Graph)
- File size: 20-30 MB compressed

**Data extraction:**
```python
# Download latest GKG file
url = 'http://data.gdeltproject.org/gdeltv2/lastupdate.txt'
# Parse for .gkg.csv.zip URL
# Download ZIP (~20-30 MB)
# Extract and parse CSV
# Filter for ECON/FINANCE/STOCK/MARKET themes
# Extract V2Tone values
# Calculate average sentiment
```

**Output:** Sentiment score (-10 to +10 scale)

**Why GDELT:**
- Free, unlimited access
- Global coverage (100+ languages)
- 15-minute freshness
- Rich metadata (themes, locations, entities)

---

### 2. Yahoo Finance (SPY - S&P 500 ETF)
**What it provides:** Market price data

**Implementation:**
- Library: yfinance (Python wrapper)
- Metric: Daily close price
- Calculation: Daily return percentage

**Code:**
```python
import yfinance as yf
spy = yf.download('SPY', period='5d', interval='1d')
current_price = spy['Close'][-1]
prev_price = spy['Close'][-2]
daily_return = ((current_price - prev_price) / prev_price) * 100
```

**Output:** Daily return (%)

**Note:** yfinance is a web scraper, not official API. For production, would use Alpha Vantage or IEX Cloud.

---

### 3. FRED (Federal Reserve Economic Data)
**What it provides:** VIX Volatility Index

**Implementation:**
- API: Official FRED REST API
- Series ID: VIXCLS
- Authentication: API key required (free)

**Code:**
```python
url = f"https://api.stlouisfed.org/fred/series/observations"
params = {
    'series_id': 'VIXCLS',
    'api_key': FRED_API_KEY,
    'file_type': 'json',
    'sort_order': 'desc'
}
response = requests.get(url, params=params)
vix = float(response.json()['observations'][0]['value'])
```

**Output:** VIX level (typically 10-40)

**VIX Interpretation:**
- VIX < 15: Market calm
- VIX 15-20: Moderate concern
- VIX 20-30: Elevated fear
- VIX > 30: Panic

---

## System Components

### Component 1: Historical Backfill (`backfill_historical.py`)

**Purpose:** Collect labeled training data

**Process:**
1. Iterate through date range (Jan 1 - Sep 30, 2024)
2. For each day:
   - Query GDELT BigQuery for sentiment
   - Fetch SPY close price
   - Fetch VIX value
   - Label: did_crash = True if market_return < -2%
3. Export to CSV with 262 rows

**Output:** `historical_data.csv`
```
date,sentiment,market_return,vix,close_price,did_crash
2024-01-01,-0.31,1.2,18.5,450.23,False
2024-01-02,-0.18,-2.5,21.3,438.72,True
...
```

**Key features:**
- Rate limiting with exponential backoff
- Progress tracking
- Error handling for missing data
- Data validation

---

### Component 2: Feature Engineering (`create_ml_dataset.py`)

**Purpose:** Transform raw data into ML features

**Features created (29 total):**

**Base features (5):**
- sentiment, market_return, vix
- divergence_magnitude = |sentiment - market_return|
- anomaly_score (rule-based, 0-100)

**Binary flags (12):**
- is_anomaly, is_high_risk, is_medium_risk, is_low_risk
- Divergence types (one-hot encoded)
- VIX categories (panic, high, elevated, normal)

**Temporal features (6):**
- sentiment_3d_avg, market_3d_avg, vix_3d_avg
- sentiment_7d_avg, market_7d_avg, vix_7d_avg

**Interaction terms (3):**
- sentiment × market_return
- vix × divergence_magnitude
- sentiment × vix

**Lag features (3):**
- prev_sentiment, prev_market, prev_vix

**Code example:**
```python
# Rolling averages
df['sentiment_3d_avg'] = df['sentiment'].rolling(3).mean()

# Interaction terms
df['sentiment_market_interaction'] = df['sentiment'] * df['market_return']

# Binary flags
df['is_high_risk'] = (df['anomaly_score'] >= 70).astype(int)
```

**Output:** `training-data.csv` (262 rows × 29 features)

---

### Component 3: AWS Glue ETL (`glue_job.py`)

**Purpose:** Process data at scale with PySpark

**Key transformations:**

**1. DynamoDB Decimal type resolution:**
```python
resolved_frame = resolveChoice(
    dynamic_frame,
    specs=[
        ('sentiment', 'cast:double'),
        ('market_return', 'cast:double'),
        ('vix', 'cast:double')
    ]
)
```

**2. Window functions for rolling calculations:**
```python
from pyspark.sql.window import Window
import pyspark.sql.functions as F

window_3d = Window.orderBy('date').rowsBetween(-2, 0)
df = df.withColumn('sentiment_3d_avg', F.avg('sentiment').over(window_3d))
```

**3. Partitioning for performance:**
```python
df.write.partitionBy('date').parquet('s3://bucket/ml-data/')
```

**Why Glue instead of local processing:**
- Scalable to millions of rows
- Distributed computing (Spark)
- Handles large datasets (GB-TB range)
- Production-grade

---

### Component 4: Model Training (`anomaly_crash_predictor.ipynb`)

**Purpose:** Train XGBoost crash prediction model

**Hyperparameters:**
```python
{
    'objective': 'binary:logistic',
    'num_round': 100,
    'max_depth': 3,
    'eta': 0.1,           # Learning rate
    'subsample': 0.8,     # Row sampling
    'eval_metric': 'auc'
}
```

**Training process:**
1. Load train.csv and test.csv from S3
2. Create SageMaker XGBoost estimator
3. Train on ml.m5.xlarge instance
4. Evaluate on test set
5. Deploy to ml.t2.medium endpoint

**Results:**
- AUC: 0.74
- Accuracy: 81.1%
- Precision: 57%
- Recall: 36%

**Why XGBoost:**
- Optimal for tabular data
- Works well with small datasets
- Interpretable feature importance
- Industry standard for financial prediction

---

### Component 5: Real-Time Lambda (`lambda_function.py`)

**Purpose:** Orchestrate real-time detection

**Execution flow:**
```python
def lambda_handler(event, context):
    # 1. FETCH DATA
    sentiment = fetch_gdelt_sentiment()      # ~1 sec
    market_data = fetch_market_data()        # ~0.2 sec
    vix = fetch_vix()                        # ~4 sec
    
    # 2. STORE RAW (Bronze)
    store_raw_data(timestamp, sentiment, market_data, vix)
    
    # 3. DETECT ANOMALY (Rule-based)
    anomaly_result = detect_anomaly(sentiment, market_data['return'], vix)
    
    # 4. PREPARE FEATURES
    features = prepare_features_for_ml(
        sentiment, market_data['return'], vix,
        anomaly_result['divergence_magnitude'],
        anomaly_result['score']
    )
    
    # 5. ML PREDICTION
    ml_prediction = get_ml_prediction(features)  # ~0.1 sec
    
    # 6. LLM EXPLANATION (if anomaly)
    if anomaly_result['is_anomaly']:
        llm_explanation = generate_llm_explanation(
            sentiment, market_data['return'], vix,
            anomaly_result, ml_prediction
        )  # ~1 sec
        
        # 7. STORE ANOMALY (Silver)
        store_anomaly_event(...)
    
    # 8. STORE DAILY METRICS (DynamoDB)
    store_daily_metrics(...)
    
    return {'statusCode': 200, 'body': 'Success'}
```

**Key functions:**

**fetch_gdelt_sentiment():**
```python
# Download latest CSV
response = requests.get(lastupdate_url)
gkg_url = parse_gkg_url(response.text)

# Unzip and parse
with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
    reader = csv.reader(..., delimiter='\t')
    
    for row in reader:
        themes = row[6]
        if 'ECON' in themes or 'FINANCE' in themes:
            tone_value = float(row[15].split(',')[0])
            tones.append(tone_value)

return average(tones)
```

**detect_anomaly():**
```python
score = 0

# Divergence patterns
if sentiment > -0.06 and market_return < -0.5:
    score += 50  # Positive news, market drops

# VIX thresholds
if vix > 35:
    score += 50  # Panic level
elif vix > 25:
    score += 30  # High stress

# Magnitude bonus
if abs(sentiment - market_return) > 2.5:
    score += 20

# Risk classification
if score >= 70:
    risk_level = "HIGH"
elif score >= 50:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"
```

**generate_llm_explanation():**
```python
prompt = f"""You are a financial analyst. Explain this market anomaly concisely.

Metrics:
- News Sentiment: {sentiment}
- Market Return: {market_return}%
- VIX: {vix}
- Anomaly Score: {anomaly_score}/100
- ML Crash Probability: {ml_crash_prob:.1%}

Explain why this is concerning."""

response = bedrock.invoke_model(
    modelId='us.meta.llama3-1-8b-instruct-v1:0',
    body=json.dumps({
        'prompt': prompt,
        'max_gen_len': 200,
        'temperature': 0.7
    })
)

return response_body['generation']
```

---

## Implementation Guide

### Prerequisites

**AWS Account:**
- Free tier eligible recommended
- IAM permissions for Lambda, S3, DynamoDB, SageMaker, Glue, EventBridge, Bedrock

**Local Environment:**
- Python 3.12
- pip for package management
- AWS CLI configured

**API Keys:**
- FRED API key (free from https://fred.stlouisfed.org/docs/api/api_key.html)
- AWS credentials configured

---

### Step 1: Set Up S3 Bucket

**Create bucket:**
```bash
aws s3 mb s3://bucket-name --region us-east-1
```

**Create folder structure:**
```bash
aws s3api put-object --bucket bucket-name --key raw/
aws s3api put-object --bucket bucket-name --key raw/backfill/
aws s3api put-object --bucket bucket-name --key raw/live/
aws s3api put-object --bucket bucket-name --key processed/anomalies/
aws s3api put-object --bucket bucket-name --key ml-data/
```

---

### Step 2: Create DynamoDB Tables

**Table 1: daily-metrics**
```bash
aws dynamodb create-table \
    --table-name daily-metrics \
    --attribute-definitions AttributeName=date,AttributeType=S \
    --key-schema AttributeName=date,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

**Table 2: anomaly-events**
```bash
aws dynamodb create-table \
    --table-name anomaly-events \
    --attribute-definitions AttributeName=event_id,AttributeType=S \
    --key-schema AttributeName=event_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

---

### Step 3: Run Historical Backfill

**Install dependencies:**
```bash
pip install pandas google-cloud-bigquery yfinance requests
```

**Set environment variables:**
```bash
export FRED_API_KEY='your_fred_api_key'
export GOOGLE_APPLICATION_CREDENTIALS='path/to/gcp_credentials.json'
```

**Run backfill:**
```bash
python backfill_historical.py
```

**Expected output:**
- `historical_data.csv` with 262 rows
- Progress updates every 10 days
- Completion message with anomaly count

**Note:** This step used BigQuery (hit quota). Now using CSV method in production.

---

### Step 4: Feature Engineering

**Run feature creation:**
```bash
python create_ml_dataset.py
```

**Output:**
- `training-data.csv` with 29 features
- Train/test split (80/20)

---

### Step 5: Train ML Model

**Upload data to S3:**
```bash
aws s3 cp train.csv s3://bucket-name/ml-data/train/
aws s3 cp test.csv s3://bucket-name/ml-data/test/
```

**What happens:**
1. Creates SageMaker training job
2. Trains XGBoost on ml.m5.xlarge (~5 min)
3. Evaluates on test set
4. Deploys to ml.t2.medium endpoint
5. Prints metrics (AUC, accuracy, precision, recall)

**Expected output:**
```
Training complete!
AUC: 0.74
Accuracy: 81.1%
Endpoint: anomaly-crash-predictor
Status: InService
```

---

### Step 6: Deploy Lambda Function

**Create Lambda function:**
```bash
aws lambda create-function \
    --function-name anomaly-detector \
    --runtime python3.12 \
    --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
    --handler lambda_function.lambda_handler \
    --timeout 180 \
    --memory-size 512 \
    --region us-east-1 \
    --zip-file fileb://lambda_function.zip
```

**Set environment variables:**
```bash
aws lambda update-function-configuration \
    --function-name anomaly-detector \
    --environment Variables={
        S3_BUCKET=bucket-name,
        FRED_API_KEY=your_key_here
    }
```

**Test manually:**
```bash
aws lambda invoke \
    --function-name anomaly-detector \
    --payload '{}' \
    response.json
```

---

### Step 7: Set Up EventBridge Schedule

**Create rule:**
```bash
aws events put-rule \
    --name anomaly-detector-30min-v2 \
    --schedule-expression "rate(30 minutes)" \
    --state ENABLED \
    --region us-east-1
```

**Add Lambda as target:**
```bash
aws events put-targets \
    --rule anomaly-detector-30min-v2 \
    --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:ACCOUNT_ID:function:anomaly-detector"
```

**Add Lambda permission for EventBridge:**
```bash
aws lambda add-permission \
    --function-name anomaly-detector \
    --statement-id EventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/anomaly-detector-30min-v2
```

---

## 📊 Results

### Model Performance

**Metrics on 53-sample test set:**
- **AUC:** 0.74 (decent separation between classes)
- **Accuracy:** 81.1% (correctly classified 81% of days)
- **Precision:** 57% (when predicting crash, correct 57% of time)
- **Recall:** 36% (catches 36% of actual crashes)

**Confusion Matrix:**
```
                Predicted
              Normal  Crash
Actual Normal    39      3
       Crash      7      4
```

**Analysis:**
- Good: Low false positive rate (7%)
- Weak: Misses 64% of crashes
- Root cause: Limited training data (only 262 days)
- Improvement path: Expand to 5+ years, apply SMOTE

---

### System Performance

**Execution metrics:**
- Average runtime: 6 seconds per execution
- Memory usage: ~150 MB peak
- Success rate: 100% (after fixing encoding issues)

**Latency breakdown:**
- GDELT CSV download: ~1 sec
- Yahoo Finance: ~0.2 sec
- FRED VIX: ~4 sec (slowest)
- SageMaker inference: ~0.1 sec
- Bedrock LLM: ~1 sec (when called)
- Storage: ~0.1 sec

**Cost (3-day demo run):**
- Lambda executions: $0 (free tier)
- S3 storage: $0 (negligible)
- DynamoDB: $0 (free tier)
- SageMaker endpoint: ~$4 (ml.t2.medium × 72 hours)
- Bedrock: ~$0.03 (8 LLM calls)
- **Total: ~$4**

---

### Data Collected

**Historical backfill:**
- Trading days: 262
- Date range: Jan 1 - Sep 30, 2024
- Total data volume: 1,045.5 GB (BigQuery processing)
- Anomalies detected: 26 (9.9% rate)
- High-risk events: 8
- Medium-risk events: 18

**Real-time collection:**
- Runs: 48 per day (every 30 minutes)
- Data per run: ~30 MB (GDELT CSV + API responses)
- Articles per run: 50-500 (economic themes)

---

---

## Key Learnings

### Technical Skills Gained

**AWS Services:**
- Lambda: Serverless orchestration, event-driven architecture
- S3: Data lake design, lifecycle policies, static websites
- DynamoDB: NoSQL design, partition keys, Decimal handling
- Glue: PySpark ETL, window functions, type casting
- SageMaker: ML training, endpoint deployment, inference
- EventBridge: Scheduling, retry policies, resource permissions
- Bedrock: LLM integration, prompt engineering, cost optimization

**Data Engineering:**
- Medallion architecture (Bronze/Silver/Gold)
- Multi-source data integration
- ETL pipeline design
- Error handling and graceful degradation

**Machine Learning:**
- Feature engineering for time-series
- Handling class imbalance
- Model evaluation (AUC, precision, recall trade-offs)
- Production ML deployment

**LLMs:**
- Prompt engineering for financial analysis
- Cost-per-call optimization (only call on anomalies)
- Response parsing and error handling

---
