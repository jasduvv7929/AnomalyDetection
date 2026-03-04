# Market Anomaly Detection System

**Serverless ML system for automated financial market anomaly detection with explainable AI**

---

## 📖 Overview

Built an automated system that detects market anomalies by identifying divergence patterns between news sentiment and market behavior, using machine learning and large language models.

**What I Built:**
- 🤖 Dual-detection system combining rule-based scoring with XGBoost ML
- 💬 Automated explanations using Amazon Bedrock (Llama 3.1 8B)
- ⚡ Serverless architecture integrating 7 AWS services
- 📊 Automated data ingestion at 30-minute intervals

---

## 🏗️ Architecture

**Data Flow:**
```
GDELT News Sentiment → ┐
Market Data          → → EventBridge (30-min schedule) → Lambda → SageMaker XGBoost
VIX Volatility       → ┘                                  ↓              ↓
                                                     Bedrock LLM    Predictions
                                                          ↓              ↓
                                                     DynamoDB ←→ S3 Data Lake
```

**AWS Services Integrated:**
- **Lambda** - Serverless orchestration
- **S3** - Multi-tier data lake (Bronze/Silver/Gold)
- **DynamoDB** - Dual-table schema for metrics and events
- **Glue** - Distributed ETL with PySpark
- **EventBridge** - Scheduled automation (30-minute intervals)
- **SageMaker** - XGBoost training and deployment
- **Bedrock** - LLM-powered explanations (Llama 3.1 8B)

---

## 📊 Results

**Model Performance:**
- Training Data: 262 trading days (1+ TB processed)
- Algorithm: XGBoost
- Accuracy: 81.1%
- AUC: 0.74
- Anomalies Detected: 26 (9.9% of days)
- High-Risk Events: 8

**System Performance:**
- Execution Time: 6 seconds average
- Automation Frequency: 30-minute intervals
- Infrastructure: Fully serverless (Lambda, DynamoDB, S3)

---

## 🚀 Deployment

**Prerequisites:**
- AWS Account with free tier access
- Python 3.12+
- FRED API key (free registration)


**Key Components:**
1. S3 bucket for data lake storage
2. DynamoDB tables for metrics and events
3. Lambda function for orchestration
4. EventBridge rule for scheduling
5. SageMaker endpoint for ML inference
6. IAM roles and permissions

Complete technical documentation: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

---

## 📁 Project Structure

**Core Python Files:**
- `backfill_historical.py` - Historical data collection (262 days)
- `create_ml_dataset.py` - Feature engineering (29 features)
- `train_model.py` - XGBoost model training on SageMaker
- `lambda_function.py` - Real-time detection Lambda function
- `glue_job.py` - AWS Glue PySpark ETL job

**Documentation:**
- `README.md` - Project overview (this file)
- `PROJECT_DOCUMENTATION.md` - Complete technical documentation

**Configuration:**
- `requirements.txt` - Python dependencies

---

## 💡 Technical Implementation

### 1. Serverless Architecture
Integrated 7 AWS services (Lambda, Glue, EventBridge, S3, DynamoDB, SageMaker, Bedrock) with automated orchestration

### 2. LLM Integration
Integrated Amazon Bedrock (Llama 3.1 8B) to generate automated natural language explanations for detected anomalies

### 3. Data Engineering
Implemented multi-tier data architecture (Bronze/Silver/Gold layers) processing 262 days of historical data (1+ TB)

### 4. Machine Learning
Trained XGBoost model on engineered features, deployed to SageMaker endpoint for automated inference

---

## 🎯 System Capabilities

**Automated Detection:**
- Identifies divergence patterns between news sentiment and market behavior
- Applies rule-based scoring (0-100 scale)
- Generates ML predictions via SageMaker XGBoost
- Creates natural language explanations using Bedrock LLM

**Data Storage:**
- Bronze layer: Raw data from all sources
- Silver layer: Validated anomalies with explanations
- DynamoDB: Dual-table schema for fast queries
- Complete audit trail for reprocessing

---

## 🛠️ Technologies Used

**AWS Services:**
- Lambda (serverless compute)
- S3 (data lake storage)
- DynamoDB (NoSQL database)
- Glue (distributed ETL with PySpark)
- EventBridge (scheduled automation)
- SageMaker (ML training and deployment)
- Bedrock (LLM inference)

**Data Sources:**
- GDELT - Global news sentiment data
- Market data - SPY (S&P 500 ETF) prices
- FRED - VIX volatility index

**Machine Learning:**
- XGBoost (SageMaker built-in algorithm)
- 29 engineered features
- PySpark for distributed processing

**LLM:**
- Amazon Bedrock (Llama 3.1 8B Instruct model)

**Languages & Tools:**
- Python 3.12
- boto3 (AWS SDK)
- pandas (data processing)

---

## 📈 Potential Enhancements

**Model Improvements:**
- Expand training data to 5+ years for better generalization
- Apply SMOTE or other techniques for class balancing
- Implement ensemble approach with multiple algorithms

**Infrastructure:**
- Multi-region SageMaker deployment for higher availability
- Real-time streaming with Kinesis (sub-second latency)
- Comprehensive CloudWatch monitoring and alerting

**Features:**
- Additional data sources (options flow, social media sentiment)
- Backtesting framework to validate predictions
- Web application interface for non-technical users

---
