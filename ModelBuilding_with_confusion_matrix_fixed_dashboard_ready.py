#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import spacy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


warnings.filterwarnings('ignore')

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

for var in ["JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "SPARK_SUBMIT_OPTS", "PYSPARK_SUBMIT_ARGS"]:
    value = os.environ.get(var, "")
    if "-Xmx" in value or "-Xms" in value:
        os.environ.pop(var, None)

SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, lower, regexp_replace, length, explode, pandas_udf
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql.types import ArrayType, StringType
    SPARK_AVAILABLE = True
    print("✅ SPARK NLP POWER UP! 🚀 3-Class Text Classification Ready!")
except ImportError:
    print("⚠️ pyspark not installed. Install: pip install pyspark")
    pandas_udf = None
    ArrayType = None
    StringType = None

MONGODB_AVAILABLE = False
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
    print("✅ MongoDB support enabled")
except ImportError:
    print("⚠️ pymongo not installed, skip MongoDB")

API_AVAILABLE = False
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    API_AVAILABLE = True
    print("✅ FastAPI API support enabled")
except ImportError:
    print("⚠️ FastAPI not installed, skip API")
    BaseModel = object

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

MONGO_URI = "mongodb+srv://CalvinChen:Aa112211@cluster0.szbiybq.mongodb.net/?appName=Cluster0&tlsInsecure=true"

DB_NAME = "ecommerce_text_classification"
DASHBOARD_DIR = Path("dashboard_data")
DASHBOARD_DIR.mkdir(exist_ok=True)

print("🚀 Online Review 3-Class Sentiment Classification System v6.2")
print("✅ Amazon_Unlocked_Mobile.csv ready")
print(f"📂 Dashboard sync folder: {DASHBOARD_DIR.resolve()}")

TRAINED_PIPELINE_MODEL = None
LATEST_RESULTS = None

LABEL_MAP = {
    0.0: "Negative",
    1.0: "Neutral",
    2.0: "Positive"
}

nlp = spacy.load("en_core_web_sm")

CUSTOM_STOPWORDS = {
    "phone", "mobile", "cellphone", "amazon", "product", "device",
    "item", "thing", "stuff", "use", "one"
}


def clean_text_python(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def label_from_rating_3class(rating):
    if rating in [1, 2]:
        return 0.0
    elif rating == 3:
        return 1.0
    elif rating in [4, 5]:
        return 2.0
    return None


def preprocess_input_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_dashboard_outputs(timestamp, class_dist_df, metrics_df, neg_df, neu_df, pos_df, sample_pred_df):
    files_to_write = {
        DASHBOARD_DIR / f'class_distribution_3class_{timestamp}.csv': class_dist_df,
        DASHBOARD_DIR / f'text_classification_3class_metrics_{timestamp}.csv': metrics_df,
        DASHBOARD_DIR / f'top_negative_words_3class_{timestamp}.csv': neg_df,
        DASHBOARD_DIR / f'top_neutral_words_3class_{timestamp}.csv': neu_df,
        DASHBOARD_DIR / f'top_positive_words_3class_{timestamp}.csv': pos_df,
        DASHBOARD_DIR / f'sample_predictions_3class_{timestamp}.csv': sample_pred_df,
    }
    for path, df in files_to_write.items():
        df.to_csv(path, index=False)
    for image_name in ['text_classification_3class_results.png', 'confusion_matrix_3class.png']:
        src = Path(image_name)
        if src.exists():
            shutil.copy2(src, DASHBOARD_DIR / src.name)
    print(f"📂 Dashboard data synced to: {DASHBOARD_DIR.resolve()}")


def load_and_process_data_spark(save_to_mongo=False):
    print("🔥 SPARK 3-Class Text Classification Data Pipeline...")
    if not SPARK_AVAILABLE:
        print("⚠️ Fallback to Pandas")
        return load_and_process_data_pandas(save_to_mongo)

    spark = (
        SparkSession.builder
        .appName("Ecommerce3ClassTextClassification")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("hadoop.security.authentication", "simple")
        .getOrCreate()
    )

    try:
        print("📥 Loading raw review data with Spark...")
        df = spark.read.csv("Amazon_Unlocked_Mobile.csv", header=True, inferSchema=True)
        print(f"📊 Raw data: {df.count()} rows")

        df = df.select("Reviews", "Rating", "Product Name", "Brand Name", "Price").dropna(subset=["Reviews", "Rating"])
        df = df.filter(col("Rating").isin([1, 2, 3, 4, 5]))
        df = df.withColumn("clean_text", lower(regexp_replace(col("Reviews"), "[^a-zA-Z\\s]", " ")))
        df = df.withColumn("clean_text", regexp_replace(col("clean_text"), "\\s+", " "))
        df = df.filter(length(col("clean_text")) > 5)
        df = df.withColumn(
            "label",
            when(col("Rating").isin([1, 2]), 0.0)
            .when(col("Rating") == 3, 1.0)
            .otherwise(2.0)
        )

        print(f"🧹 Cleaned labeled review records: {df.count()}")

        if save_to_mongo and MONGODB_AVAILABLE:
            save_spark_df_to_mongodb(df, "reviews_cleaned_3class")

        return spark, df
    except Exception as e:
        print(f"❌ Spark processing failed: {e}")
        print("⚠️ Switching to Pandas fallback...")
        try:
            spark.stop()
        except Exception:
            pass
        return load_and_process_data_pandas(save_to_mongo)


def load_and_process_data_pandas(save_to_mongo=False):
    print("📊 Pandas fallback 3-class text processing...")
    df = pd.read_csv("Amazon_Unlocked_Mobile.csv")
    print(f"Original data shape: {df.shape}")

    df = df[['Reviews', 'Rating', 'Product Name', 'Brand Name', 'Price']].dropna(subset=['Reviews', 'Rating'])
    df = df[df['Rating'].isin([1, 2, 3, 4, 5])].copy()
    df['clean_text'] = df['Reviews'].apply(clean_text_python)
    df = df[df['clean_text'].str.len() > 5].copy()
    df['label'] = df['Rating'].apply(label_from_rating_3class)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(float)

    print(f"After cleaning: {len(df)} records")

    if save_to_mongo and MONGODB_AVAILABLE:
        save_to_mongodb_pandas(df, "reviews_cleaned_3class")

    return None, df


def save_spark_df_to_mongodb(df, collection_name="reviews_cleaned_3class"):
    try:
        if not MONGODB_AVAILABLE:
            return
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        collection.delete_many({})
        docs = [row.asDict() for row in df.limit(50000).collect()]
        if docs:
            result = collection.insert_many(docs)
            print(f"✅ MongoDB saved: {len(result.inserted_ids)} records into {collection_name}")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB failed: {e}")


def save_to_mongodb_pandas(df, collection_name="reviews_cleaned_3class"):
    try:
        if not MONGODB_AVAILABLE:
            return
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        collection.delete_many({})
        docs = df.head(50000).to_dict(orient='records')
        if docs:
            result = collection.insert_many(docs)
            print(f"✅ MongoDB saved: {len(result.inserted_ids)} records into {collection_name}")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB failed: {e}")


def train_text_classifier_spark_3class(spark, df):
    print("🤖 Training Spark 3-class text classification model...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=8000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.0, family="multinomial")
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, lr])
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    metrics = {
        'Accuracy': evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
        'F1': evaluator.evaluate(predictions, {evaluator.metricName: "f1"}),
        'Precision': evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
        'Recall': evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    }

    print("🏆 3-Class Classification Performance")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, predictions, metrics, train_df, test_df


def train_text_classifier_pandas_3class(df):
    print("🤖 Training Pandas/sklearn 3-class fallback text classifier...")
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SkPipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, stop_words='english')),
        ('clf', SkLogisticRegression(max_iter=1500, multi_class='auto'))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted')
    }

    print("🏆 3-Class Classification Performance")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    pred_df = pd.DataFrame({'clean_text': X_test.values, 'label': y_test.values, 'prediction': y_pred})
    return model, pred_df, metrics, X_train, X_test, y_test


def get_class_distribution_pandas(df):
    dist = df['label'].value_counts().sort_index()
    return pd.DataFrame({'label': ['Negative', 'Neutral', 'Positive'], 'count': [dist.get(0.0, 0), dist.get(1.0, 0), dist.get(2.0, 0)]})


def get_top_words_by_class_pandas(df, target_label, top_n=15):
    subset = df[df['label'] == target_label]
    texts = subset['clean_text'].astype(str).tolist()
    words = []
    for doc in nlp.pipe(texts, batch_size=500):
        for token in doc:
            if not token.is_alpha or token.is_stop or len(token.text) <= 2:
                continue
            lemma = token.lemma_.lower()
            if lemma in CUSTOM_STOPWORDS:
                continue
            if token.pos_ in ("ADJ", "NOUN"):
                words.append(lemma)
    counter = Counter(words).most_common(top_n)
    return pd.DataFrame(counter, columns=["word", "count"])


if SPARK_AVAILABLE:
    @pandas_udf(ArrayType(StringType()))
    def extract_keywords_udf(texts):
        canonical_stopwords = {
            "day", "days", "time", "money", "work", "one", "two", "three", "year", "years",
            "month", "months", "week", "weeks", "people", "thing", "things", "number", "price",
            "shipping", "case", "screen", "battery", "product", "device", "app", "apps", "system",
            "software", "feature", "features", "use", "used", "using", "buy", "bought", "phone",
            "mobile", "cellphone", "amazon", "card", "unlocked", "sim", "seller", "good", "new",
            "apple", "iphone", "display", "charger", "bluetooth", "wifi", "network", "payment",
            "credit", "delivery"
        }
        result = []
        for text in texts:
            if not text:
                result.append([])
                continue
            doc = nlp(str(text).lower())
            words = []
            for token in doc:
                if not token.is_alpha or token.is_stop or len(token.text) <= 2:
                    continue
                if token.pos_ not in ("ADJ", "NOUN"):
                    continue
                lemma = token.lemma_.lower()
                if lemma in CUSTOM_STOPWORDS or lemma in canonical_stopwords:
                    continue
                words.append(lemma)
            result.append(words)
        return pd.Series(result)


def get_sample_predictions_spark(predictions, n=30):
    pdf = predictions.select("clean_text", "label", "prediction", "probability").limit(n).toPandas()
    pdf['actual_sentiment'] = pdf['label'].map(LABEL_MAP)
    pdf['predicted_sentiment'] = pdf['prediction'].map(LABEL_MAP)
    return pdf


def get_class_distribution_spark(df):
    pdf = df.select("label").toPandas()
    return get_class_distribution_pandas(pdf)


def get_top_words_by_class_spark(df, target_label, top_n=15, sample_size=5000):
    sampled_df = df.filter(col("label") == target_label).select("clean_text").limit(sample_size)
    keyword_df = sampled_df.withColumn("keywords", extract_keywords_udf(col("clean_text")))
    top_df = keyword_df.select(explode(col("keywords")).alias("word")).groupBy("word").count().orderBy(col("count").desc()).limit(top_n)
    return top_df.toPandas()


def create_visualizations_3class(class_dist_df, metrics_df, neg_df, neu_df, pos_df):
    print("📊 Generating 3-class text classification visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('🚀 SPARK 3-Class Sentiment Classification Results', fontsize=18, fontweight='bold')

    sns.barplot(data=class_dist_df, x='label', y='count', ax=axes[0, 0], palette=['#EF5350', '#FFCA28', '#66BB6A'])
    axes[0, 0].set_title('3-Class Sentiment Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment Class')
    axes[0, 0].set_ylabel('Review Count')

    mdf = metrics_df.melt(var_name='Metric', value_name='Score')
    sns.barplot(data=mdf, x='Metric', y='Score', ax=axes[0, 1], palette='Blues_d')
    axes[0, 1].set_title('Multiclass Performance', fontweight='bold')
    axes[0, 1].set_ylim(0, 1)

    neg_plot = neg_df.head(10).sort_values('count', ascending=True)
    sns.barplot(data=neg_plot, x='count', y='word', ax=axes[1, 0], palette='Reds_d')
    axes[1, 0].set_title('Top Negative Keywords', fontweight='bold')

    pos_plot = pos_df.head(10).sort_values('count', ascending=True)
    sns.barplot(data=pos_plot, x='count', y='word', ax=axes[1, 1], palette='Greens_d')
    axes[1, 1].set_title('Top Positive Keywords', fontweight='bold')

    plt.tight_layout()
    plt.savefig('text_classification_3class_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Chart saved: text_classification_3class_results.png")


class TextRequest(BaseModel):
    text: str


if API_AVAILABLE:
    app = FastAPI(title="🚀 SPARK 3-Class Sentiment Classification API v6.2")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    async def root():
        return {"message": "🚀 SPARK 3-Class Sentiment Classification API v6.2", "spark": SPARK_AVAILABLE, "mongodb": MONGODB_AVAILABLE, "endpoints": ["/api/metrics", "/api/predict", "/api/samples", "/docs"]}

    @app.get("/api/metrics")
    async def get_metrics():
        global LATEST_RESULTS
        if LATEST_RESULTS is None:
            raise HTTPException(status_code=400, detail="Model not trained yet. Run full analysis first.")
        return {"metrics": LATEST_RESULTS}

    @app.get("/api/samples")
    async def get_samples(limit: int = 10):
        if SPARK_AVAILABLE:
            spark, df = load_and_process_data_spark()
            model, predictions, metrics, _, _ = train_text_classifier_spark_3class(spark, df)
            sample_df = get_sample_predictions_spark(predictions, limit)
            spark.stop()
            return sample_df.to_dict(orient="records")
        _, df = load_and_process_data_pandas()
        model, pred_df, metrics, _, _, _ = train_text_classifier_pandas_3class(df)
        sample_df = pred_df.head(limit).copy()
        sample_df['actual_sentiment'] = sample_df['label'].map(LABEL_MAP)
        sample_df['predicted_sentiment'] = sample_df['prediction'].map(LABEL_MAP)
        return sample_df.to_dict(orient="records")

    @app.post("/api/predict")
    async def predict_text(req: TextRequest):
        global TRAINED_PIPELINE_MODEL
        if TRAINED_PIPELINE_MODEL is None:
            raise HTTPException(status_code=400, detail="Model not trained yet. Run full analysis first.")
        text = preprocess_input_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="Input text is empty after cleaning.")

        if SPARK_AVAILABLE:
            spark = (SparkSession.builder.appName("PredictTextAPI3Class").master("local[*]").config("spark.driver.memory", "1g").config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true").getOrCreate())
            input_df = spark.createDataFrame([(text,)], ["clean_text"])
            pred = TRAINED_PIPELINE_MODEL.transform(input_df).select("prediction", "probability").collect()[0]
            prediction = float(pred["prediction"])
            prob = pred["probability"]
            result = {
                "text": req.text,
                "clean_text": text,
                "prediction": LABEL_MAP.get(prediction, "Unknown"),
                "label": int(prediction),
                "negative_probability": round(float(prob[0]), 4),
                "neutral_probability": round(float(prob[1]), 4),
                "positive_probability": round(float(prob[2]), 4)
            }
            spark.stop()
            return result

        pred = TRAINED_PIPELINE_MODEL.predict([text])[0]
        proba = TRAINED_PIPELINE_MODEL.predict_proba([text])[0]
        return {
            "text": req.text,
            "clean_text": text,
            "prediction": LABEL_MAP.get(pred, "Unknown"),
            "label": int(pred),
            "negative_probability": round(float(proba[0]), 4),
            "neutral_probability": round(float(proba[1]), 4),
            "positive_probability": round(float(proba[2]), 4)
        }


def plot_confusion_matrix_spark(pred_df, labels=["Negative", "Neutral", "Positive"]):
    print("📊 Generating Confusion Matrix for 3-Class Sentiment...")
    true_labels = pred_df['label']
    pred_labels = pred_df['prediction']
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - 3-Class Sentiment Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix_3class.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print("✅ Confusion Matrix saved: confusion_matrix_3class.png")
    print("📋 3-Class Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=labels))
    return cm


def main():
    global TRAINED_PIPELINE_MODEL, LATEST_RESULTS

    print("\n" + "=" * 70)
    print("🚀 SPARK 3-Class Sentiment Classification v6.2 - Big Data NLP Ready!")
    print("=" * 70)
    print("\n📋 Modes:")
    print("1. 🔥 SPARK Full 3-Class Analysis")
    print("2. 🌐 API Service")
    print("3. 💾 MongoDB Only")

    mode = input("\nMode (1/2/3, Default 1): ").strip() or "1"

    if mode == "2" and API_AVAILABLE:
        print("\n🌐 Starting 3-Class Text Classification API...")
        print("📱 Docs: http://localhost:8000/docs")
        print("📊 Swagger: http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    if mode == "3" and MONGODB_AVAILABLE:
        print("\n💾 MongoDB Save (Pandas only)...")
        load_and_process_data_pandas(save_to_mongo=True)
        return

    print("\n🚀 Full 3-Class Text Classification Analysis...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if SPARK_AVAILABLE:
        spark, df = load_and_process_data_spark(save_to_mongo=MONGODB_AVAILABLE)
        model, predictions, metrics, train_df, test_df = train_text_classifier_spark_3class(spark, df)
        TRAINED_PIPELINE_MODEL = model
        LATEST_RESULTS = metrics

        class_dist_df = get_class_distribution_spark(df)
        neg_df = get_top_words_by_class_spark(df, 0.0, 15, sample_size=3000)
        neu_df = get_top_words_by_class_spark(df, 1.0, 15, sample_size=3000)
        pos_df = get_top_words_by_class_spark(df, 2.0, 15, sample_size=3000)
        sample_pred_df = get_sample_predictions_spark(predictions, n=50)
        spark.stop()
    else:
        _, df = load_and_process_data_pandas(save_to_mongo=MONGODB_AVAILABLE)
        model, pred_df, metrics, _, _, _ = train_text_classifier_pandas_3class(df)
        TRAINED_PIPELINE_MODEL = model
        LATEST_RESULTS = metrics

        class_dist_df = get_class_distribution_pandas(df)
        neg_df = get_top_words_by_class_pandas(df, 0.0, 15)
        neu_df = get_top_words_by_class_pandas(df, 1.0, 15)
        pos_df = get_top_words_by_class_pandas(df, 2.0, 15)
        sample_pred_df = pred_df.head(50).copy()
        sample_pred_df['actual_sentiment'] = sample_pred_df['label'].map(LABEL_MAP)
        sample_pred_df['predicted_sentiment'] = sample_pred_df['prediction'].map(LABEL_MAP)

    metrics_df = pd.DataFrame([metrics]).round(4)

    class_dist_df.to_csv(f'class_distribution_3class_{timestamp}.csv', index=False)
    metrics_df.to_csv(f'text_classification_3class_metrics_{timestamp}.csv', index=False)
    neg_df.to_csv(f'top_negative_words_3class_{timestamp}.csv', index=False)
    neu_df.to_csv(f'top_neutral_words_3class_{timestamp}.csv', index=False)
    pos_df.to_csv(f'top_positive_words_3class_{timestamp}.csv', index=False)
    sample_pred_df.to_csv(f'sample_predictions_3class_{timestamp}.csv', index=False)

    create_visualizations_3class(class_dist_df, metrics_df, neg_df, neu_df, pos_df)
    plot_confusion_matrix_spark(sample_pred_df)
    save_dashboard_outputs(timestamp, class_dist_df, metrics_df, neg_df, neu_df, pos_df, sample_pred_df)

    print("\n" + "=" * 60)
    print("🏆 3-Class Classification Performance")
    print("=" * 60)
    print(metrics_df)

    print("\n😡 Top Negative Keywords:")
    print(neg_df.head(10))
    print("\n😐 Top Neutral Keywords:")
    print(neu_df.head(10))
    print("\n😊 Top Positive Keywords:")
    print(pos_df.head(10))

    print("\n" + "=" * 70)
    print("✅ 3-Class Text Classification Analysis Complete!")
    print("📈 text_classification_3class_results.png")
    print("📈 confusion_matrix_3class.png")
    print(f"📊 class_distribution_3class_{timestamp}.csv")
    print(f"📊 text_classification_3class_metrics_{timestamp}.csv")
    print(f"😡 top_negative_words_3class_{timestamp}.csv")
    print(f"😐 top_neutral_words_3class_{timestamp}.csv")
    print(f"😊 top_positive_words_3class_{timestamp}.csv")
    print(f"🧪 sample_predictions_3class_{timestamp}.csv")
    print(f"📂 dashboard_data synced: {DASHBOARD_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
