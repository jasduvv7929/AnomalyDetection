"""
AWS Glue ETL Job - ML Training Data Preparation
Handles DynamoDB Decimal types correctly

Input: DynamoDB daily-metrics table
Output: S3 ml-data/training/*.parquet
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import (
    col, when, lead, to_date, sum as spark_sum, avg as spark_avg, lag
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType
from awsglue.dynamicframe import DynamicFrame

# Initialize
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print("=" * 70)
print("GLUE ETL - ML TRAINING DATA PREPARATION")
print("=" * 70)

# ============================================================
# STEP 1: READ FROM DYNAMODB
# ============================================================

print("\n Step 1: Reading from DynamoDB...")

daily_metrics_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="dynamodb",
    connection_options={
        "dynamodb.input.tableName": "daily-metrics",
        "dynamodb.throughput.read.percent": "1.0"
    }
)

print(f" Loaded {daily_metrics_dyf.count()} records")

# ============================================================
# STEP 2: RESOLVE DYNAMODB DECIMAL TYPES
# ============================================================

print("\n🔧 Step 2: Resolving DynamoDB Decimal types...")

# Convert DynamoDB Decimal types to double
daily_metrics_dyf = daily_metrics_dyf.resolveChoice(specs=[
    ('sentiment', 'cast:double'),
    ('market_return', 'cast:double'),
    ('vix', 'cast:double'),
    ('divergence_magnitude', 'cast:double'),
    ('anomaly_score', 'cast:int')
])

print(f" Decimal types resolved")

# Convert to DataFrame
df = daily_metrics_dyf.toDF()
initial_count = df.count()

print(f" Working with {initial_count} records")

# ============================================================
# STEP 3: TYPE CONVERSIONS
# ============================================================

print("\n🔧 Step 3: Type conversions...")

# Convert date
df = df.withColumn("date", to_date(col("date")))

# Handle is_anomaly boolean
df = df.withColumn("is_anomaly", 
                   when(col("is_anomaly") == True, 1)
                   .when(col("is_anomaly") == 1, 1)
                   .otherwise(0).cast(IntegerType()))

print(f" Types converted")

# ============================================================
# STEP 4: BINARY FEATURES
# ============================================================

print("\n⚙️ Step 4: Creating binary features...")

# Risk levels
df = df.withColumn("is_high_risk", when(col("risk_level") == "HIGH", 1).otherwise(0)) \
    .withColumn("is_medium_risk", when(col("risk_level") == "MEDIUM", 1).otherwise(0)) \
    .withColumn("is_low_risk", when(col("risk_level") == "LOW", 1).otherwise(0))

# Divergence types
df = df.withColumn("div_positive_news_negative_market",
                   when(col("divergence_type") == "positive_news_negative_market", 1).otherwise(0)) \
    .withColumn("div_negative_news_positive_market",
                when(col("divergence_type") == "negative_news_positive_market", 1).otherwise(0)) \
    .withColumn("div_moderate_divergence",
                when(col("divergence_type") == "moderate_divergence_with_stress", 1).otherwise(0)) \
    .withColumn("div_high_vix",
                when(col("divergence_type").contains("vix"), 1).otherwise(0))

# VIX categories
df = df.withColumn("vix_panic", when(col("vix") > 35, 1).otherwise(0)) \
    .withColumn("vix_high", when((col("vix") > 25) & (col("vix") <= 35), 1).otherwise(0)) \
    .withColumn("vix_elevated", when((col("vix") > 20) & (col("vix") <= 25), 1).otherwise(0)) \
    .withColumn("vix_normal", when(col("vix") <= 20, 1).otherwise(0))

print(f" Binary features created")

# ============================================================
# STEP 5: ROLLING WINDOW FEATURES
# ============================================================

print("\n Step 5: Creating rolling features...")

# Sort by date
df = df.orderBy("date")

# 3-day rolling averages
window_3day = Window.orderBy("date").rowsBetween(-2, 0)

df = df.withColumn("sentiment_3day_avg", spark_avg(col("sentiment")).over(window_3day)) \
    .withColumn("vix_3day_avg", spark_avg(col("vix")).over(window_3day)) \
    .withColumn("market_return_3day_avg", spark_avg(col("market_return")).over(window_3day))

# 7-day rolling averages
window_7day = Window.orderBy("date").rowsBetween(-6, 0)

df = df.withColumn("sentiment_7day_avg", spark_avg(col("sentiment")).over(window_7day)) \
    .withColumn("vix_7day_avg", spark_avg(col("vix")).over(window_7day)) \
    .withColumn("market_return_7day_avg", spark_avg(col("market_return")).over(window_7day))

print(f" Rolling averages calculated")

# ============================================================
# STEP 6: INTERACTION FEATURES
# ============================================================

print("\n Step 6: Creating interaction features...")

# Interaction terms
df = df.withColumn("sentiment_x_market", col("sentiment") * col("market_return")) \
    .withColumn("vix_x_divergence", col("vix") * col("divergence_magnitude")) \
    .withColumn("sentiment_x_vix", col("sentiment") * col("vix"))

# Lagged features
window_lag = Window.orderBy("date")
df = df.withColumn("prev_day_sentiment", lag(col("sentiment"), 1).over(window_lag)) \
    .withColumn("prev_day_market", lag(col("market_return"), 1).over(window_lag)) \
    .withColumn("prev_day_vix", lag(col("vix"), 1).over(window_lag))

print(f" Interaction features created")

# ============================================================
# STEP 7: CREATE TARGET LABELS
# ============================================================

print("\n Step 7: Creating target labels...")

window_forward = Window.orderBy("date")

# Get future returns for next 7 days
for i in range(1, 8):
    df = df.withColumn(f"future_return_day{i}",
                       lead(col("market_return"), i).over(window_forward))

# Calculate cumulative drop over next 7 days
df = df.withColumn("max_future_drop",
                   sum([when(col(f"future_return_day{i}") < 0, col(f"future_return_day{i}")).otherwise(0)
                        for i in range(1, 8)]))

# Label: did_crash = 1 if cumulative drop > 3%
df = df.withColumn("did_crash",
                   when(col("max_future_drop") < -3.0, 1).otherwise(0))

# Crash severity
df = df.withColumn("crash_severity",
                   when(col("max_future_drop") < -5.0, "severe")
                   .when(col("max_future_drop") < -3.0, "moderate")
                   .otherwise("none"))

# Drop future columns
future_cols = [f"future_return_day{i}" for i in range(1, 8)]
df = df.drop(*future_cols)

crash_count = df.filter(col("did_crash") == 1).count()
total = df.count()

print(f" Labels created:")
print(f" Crashes: {crash_count} ({crash_count/total*100:.1f}%)")
print(f" Normal:  {total - crash_count} ({(total-crash_count)/total*100:.1f}%)")

# ============================================================
# STEP 8: SELECT FINAL FEATURES
# ============================================================

print("\n Step 8: Selecting final features...")

ml_features = [
    # Target
    "did_crash",
    "crash_severity",
    
    # Identifiers
    "date",
    "timestamp",
    
    # Core signals
    "sentiment",
    "market_return",
    "vix",
    "divergence_magnitude",
    "anomaly_score",
    
    # Binary flags
    "is_anomaly",
    "is_high_risk",
    "is_medium_risk",
    "is_low_risk",
    
    # Divergence types
    "div_positive_news_negative_market",
    "div_negative_news_positive_market",
    "div_moderate_divergence",
    "div_high_vix",
    
    # VIX categories
    "vix_panic",
    "vix_high",
    "vix_elevated",
    "vix_normal",
    
    # Rolling averages
    "sentiment_3day_avg",
    "vix_3day_avg",
    "market_return_3day_avg",
    "sentiment_7day_avg",
    "vix_7day_avg",
    "market_return_7day_avg",
    
    # Interactions
    "sentiment_x_market",
    "vix_x_divergence",
    "sentiment_x_vix",
    
    # Lagged features
    "prev_day_sentiment",
    "prev_day_market",
    "prev_day_vix",
    
    # Metadata
    "risk_level",
    "divergence_type"
]

df_ml = df.select(*ml_features)

# Remove rows with nulls (from rolling windows)
df_ml = df_ml.na.drop()

final_count = df_ml.count()
dropped = total - final_count

print(f" Final dataset: {final_count} rows × {len(ml_features)} features")
print(f" Rows dropped (nulls): {dropped}")

# ============================================================
# STEP 9: SAVE TO S3
# ============================================================

print("\n Step 9: Saving to S3...")

output_path = "s3://anomaly-detection-project-jazz/ml-data/training/"

output_dyf = DynamicFrame.fromDF(df_ml, glueContext, "output_frame")

glueContext.write_dynamic_frame.from_options(
    frame=output_dyf,
    connection_type="s3",
    connection_options={"path": output_path},
    format="parquet",
    format_options={"compression": "snappy"}
)

print(f" Saved to: {output_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(" ETL JOB COMPLETE")
print("=" * 70)

print(f"\n Pipeline Summary:")
print(f"  Input records:       {initial_count}")
print(f"  Output records:      {final_count}")
print(f"  Features:            {len(ml_features)}")
print(f"  Crashes labeled:     {crash_count} ({crash_count/final_count*100:.1f}%)")
print(f"  Normal days:         {final_count - crash_count} ({(final_count-crash_count)/final_count*100:.1f}%)")

print(f"\n Feature Engineering:")
print(f"  Binary encoding:      13 features")
print(f"  Rolling averages:     6 features")
print(f"  Interaction terms:    3 features")
print(f"  Lagged features:      3 features")
print(f"  Total features:       {len(ml_features)}")

print(f"\n Dataset Quality:")
print(f" Complete time series")
print(f" Labeled for ML (did_crash)")
print(f" Format: Parquet")
print(f" Location: {output_path}")

print("\n Ready for SageMaker XGBoost training!")

job.commit()