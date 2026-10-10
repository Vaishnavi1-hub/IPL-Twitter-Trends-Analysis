# IPL 2020 Twitter Trends and Topic Detection
# Preprocessing followed by Data Analytics operations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings

from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 1. FILE AND OUTPUT FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ipl.csv"

OUTPUT_DIR = BASE_DIR / "ipl_analysis_output"
GRAPH_DIR = OUTPUT_DIR / "graphs"
TABLE_DIR = OUTPUT_DIR / "tables"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df.head(10).to_csv(TABLE_DIR / "raw_preview.csv", index=False)
df.isnull().sum().to_csv(TABLE_DIR / "missing_values_before.csv")

# --------------------------------------------------
# 3. IDENTIFY DATASET COLUMNS
# --------------------------------------------------

def find_column(columns, options):
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in columns
    }

    for option in options:
        key = re.sub(r"[^a-z0-9]", "", option.lower())
        if key in normalized:
            return normalized[key]

    return None


text_col = find_column(
    df.columns,
    ["tweet", "text", "full_text", "tweet_text", "content"]
)

date_col = find_column(
    df.columns,
    ["created_at", "datetime", "date", "timestamp", "tweet_date"]
)

id_col = find_column(
    df.columns,
    ["tweet_id", "id", "status_id"]
)

likes_col = find_column(
    df.columns,
    ["likes", "like_count", "favorite_count", "favorites"]
)

retweets_col = find_column(
    df.columns,
    ["retweets", "retweet_count", "reposts"]
)

replies_col = find_column(
    df.columns,
    ["replies", "reply_count"]
)

language_col = find_column(
    df.columns,
    ["language", "lang"]
)

if text_col is None:
    raise ValueError(
        "Tweet text column not found. "
        "Check your dataset column names."
    )

print("\nDetected text column:", text_col)

# --------------------------------------------------
# 4. DATA PREPROCESSING
# --------------------------------------------------

data = df.copy()

# Handle missing tweet text
data["tweet_text"] = data[text_col].fillna("").astype(str).str.strip()

before_empty_removal = len(data)
data = data[data["tweet_text"] != ""].copy()
empty_removed = before_empty_removal - len(data)

# Remove duplicate tweets
before_duplicates = len(data)
data = data.drop_duplicates(subset=["tweet_text"]).copy()
duplicates_removed = before_duplicates - len(data)

# Clean tweet text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b(rt|amp|https|co)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["clean_text"] = data["tweet_text"].apply(clean_text)

# Extract hashtags
data["hashtags"] = data["tweet_text"].apply(
    lambda text: re.findall(r"#(\w+)", text)
)

data["hashtag_count"] = data["hashtags"].apply(len)

# Create word count
data["word_count"] = data["clean_text"].apply(
    lambda text: len(text.split())
)

# Process engagement columns
for source_col, target_col in [
    (likes_col, "likes"),
    (retweets_col, "retweets"),
    (replies_col, "replies")
]:
    if source_col is not None:
        data[target_col] = pd.to_numeric(
            data[source_col], errors="coerce"
        ).fillna(0).clip(lower=0)
    else:
        data[target_col] = 0

data["total_engagement"] = (
    data["likes"] + data["retweets"] + data["replies"]
)

# Process date and time
if date_col is not None:
    timestamps = pd.to_datetime(
        data[date_col], errors="coerce", utc=True
    )
elif id_col is not None:
    ids = pd.to_numeric(data[id_col], errors="coerce")
    valid = ids.notna() & (ids > 0)
    timestamps = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns, UTC]"
    )

    try:
        milliseconds = (
            ids.loc[valid].astype("uint64") >> 22
        ) + 1288834974657

        timestamps.loc[valid] = pd.to_datetime(
            milliseconds, unit="ms", utc=True, errors="coerce"
        )
    except (ValueError, TypeError, OverflowError):
        pass
else:
    timestamps = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns, UTC]"
    )

if timestamps.notna().any():
    data["tweet_datetime"] = timestamps.dt.tz_convert(
        "Asia/Kolkata"
    )
    data["tweet_date"] = data["tweet_datetime"].dt.date
    data["tweet_hour"] = data["tweet_datetime"].dt.hour
    data["day_name"] = data["tweet_datetime"].dt.day_name()
else:
    data["tweet_datetime"] = pd.NaT
    data["tweet_date"] = pd.NaT
    data["tweet_hour"] = np.nan
    data["day_name"] = "Unknown"

# Process language
if language_col is not None:
    data["language"] = data[language_col].fillna("Unknown")
else:
    data["language"] = "Unknown"

# Save preprocessed dataset
data.to_csv(TABLE_DIR / "cleaned_ipl_tweets.csv", index=False)

preprocessing_report = pd.DataFrame({
    "Metric": [
        "Original rows",
        "Empty text rows removed",
        "Duplicate tweets removed",
        "Rows after preprocessing"
    ],
    "Value": [
        len(df),
        empty_removed,
        duplicates_removed,
        len(data)
    ]
})

preprocessing_report.to_csv(
    TABLE_DIR / "preprocessing_report.csv",
    index=False
)

print("\nPreprocessing completed.")
print(preprocessing_report)

# --------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

print("\n--- Exploratory Data Analysis ---")

print("Number of tweets:", len(data))
print("Unique hashtags:", data["hashtags"].explode().nunique())
print("Total likes:", data["likes"].sum())
print("Total retweets:", data["retweets"].sum())
print("Total replies:", data["replies"].sum())

# --------------------------------------------------
# 6. DAILY TWEET TREND
# --------------------------------------------------

dated = data.dropna(subset=["tweet_date"])

if not dated.empty:
    daily_tweets = dated.groupby("tweet_date").size()

    daily_tweets.to_csv(TABLE_DIR / "daily_tweet_counts.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(
        daily_tweets.index.astype(str),
        daily_tweets.values,
        marker="o"
    )

    plt.title("Daily Tweet Activity During IPL 2020")
    plt.xlabel("Date")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "01_daily_tweet_trend.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 7. TOP HASHTAGS
# --------------------------------------------------

hashtags = data["hashtags"].explode().dropna().str.lower()
top_hashtags = hashtags.value_counts().head(20)

top_hashtags.to_csv(TABLE_DIR / "top_hashtags.csv")

if not top_hashtags.empty:
    plt.figure(figsize=(10, 7))

    sns.barplot(
        x=top_hashtags.values,
        y=top_hashtags.index
    )

    plt.title("Top 20 IPL Hashtags")
    plt.xlabel("Frequency")
    plt.ylabel("Hashtag")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "02_top_hashtags.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 8. LANGUAGE DISTRIBUTION
# --------------------------------------------------

language_counts = data["language"].value_counts().head(12)
language_counts.to_csv(TABLE_DIR / "language_distribution.csv")

if language_counts.index.tolist() != ["Unknown"]:
    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=language_counts.values,
        y=language_counts.index
    )

    plt.title("Tweet Language Distribution")
    plt.xlabel("Number of Tweets")
    plt.ylabel("Language")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "03_language_distribution.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 9. ENGAGEMENT DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data["total_engagement"],
    bins=50
)

plt.title("Tweet Engagement Distribution")
plt.xlabel("Total Engagement")
plt.ylabel("Number of Tweets")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "04_engagement_distribution.png", dpi=200)
plt.close()

# --------------------------------------------------
# 10. TOP ENGAGING TWEETS
# --------------------------------------------------

top_tweets = data.nlargest(
    10, "total_engagement"
)

top_tweets[
    ["tweet_text", "likes", "retweets", "replies", "total_engagement"]
].to_csv(TABLE_DIR / "top_10_tweets.csv", index=False)

plt.figure(figsize=(10, 6))

plot_top = top_tweets.sort_values("total_engagement")

sns.barplot(
    data=plot_top,
    x="total_engagement",
    y=plot_top.index.astype(str)
)

plt.title("Top 10 Tweets by Engagement")
plt.xlabel("Total Engagement")
plt.ylabel("Tweet Rank")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_top_engaging_tweets.png", dpi=200)
plt.close()

# --------------------------------------------------
# 11. CORRELATION ANALYSIS
# --------------------------------------------------

correlation = data[
    ["likes", "retweets", "replies", "total_engagement"]
].corr()

correlation.to_csv(TABLE_DIR / "engagement_correlation.csv")

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="Blues",
    fmt=".2f"
)

plt.title("Engagement Correlation Heatmap")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "06_engagement_correlation.png", dpi=200)
plt.close()

# --------------------------------------------------
# 12. HOURLY TWEET ACTIVITY
# --------------------------------------------------

if data["tweet_hour"].notna().any():
    hourly = (
        data.dropna(subset=["tweet_hour"])
        .groupby("tweet_hour")
        .size()
        .reindex(range(24), fill_value=0)
    )

    hourly.to_csv(TABLE_DIR / "hourly_activity.csv")

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=hourly.index,
        y=hourly.values
    )

    plt.title("Hourly Twitter Activity")
    plt.xlabel("Hour of Day (IST)")
    plt.ylabel("Number of Tweets")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "07_hourly_activity.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 13. DAY OF WEEK ANALYSIS
# --------------------------------------------------

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_counts = data["day_name"].value_counts().reindex(
    weekday_order
).fillna(0)

weekday_counts.to_csv(TABLE_DIR / "weekday_activity.csv")

if weekday_counts.sum() > 0:
    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=weekday_counts.index,
        y=weekday_counts.values
    )

    plt.title("Tweet Activity by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "08_weekday_activity.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 14. TOP WORD FREQUENCY
# --------------------------------------------------

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=30,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
)

try:
    word_matrix = vectorizer.fit_transform(data["clean_text"])
    word_counts = np.asarray(word_matrix.sum(axis=0)).ravel()

    word_frequency = pd.Series(
        word_counts,
        index=vectorizer.get_feature_names_out()
    ).sort_values(ascending=False)

    word_frequency.to_csv(TABLE_DIR / "top_words.csv")

    plt.figure(figsize=(10, 7))

    sns.barplot(
        x=word_frequency.head(20).values,
        y=word_frequency.head(20).index
    )

    plt.title("Most Frequently Used Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "09_top_words.png", dpi=200)
    plt.close()

except ValueError:
    print("Word frequency skipped: no usable text.")

# --------------------------------------------------
# 15. WORD CLOUD
# --------------------------------------------------

if WordCloud is not None and data["clean_text"].str.strip().any():
    all_text = " ".join(data["clean_text"])

    wordcloud = WordCloud(
        width=1200,
        height=650,
        background_color="white",
        colormap="viridis"
    ).generate(all_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("IPL 2020 Twitter Word Cloud")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "10_wordcloud.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 16. SENTIMENT ANALYSIS
# --------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

data["sentiment_score"] = data["tweet_text"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

data["sentiment"] = np.select(
    [
        data["sentiment_score"] >= 0.05,
        data["sentiment_score"] <= -0.05
    ],
    ["Positive", "Negative"],
    default="Neutral"
)

sentiment_counts = data["sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"],
    fill_value=0
)

sentiment_counts.to_csv(TABLE_DIR / "sentiment_distribution.csv")

plt.figure(figsize=(8, 5))

sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Sentiment Analysis of IPL Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "11_sentiment_distribution.png", dpi=200)
plt.close()

# --------------------------------------------------
# 17. TOPIC DETECTION USING LDA
# --------------------------------------------------

from sklearn.decomposition import LatentDirichletAllocation

topic_text = data["clean_text"]
topic_text = topic_text[topic_text.str.split().str.len() >= 2]

topic_vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1500,
    min_df=2,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
)

try:
    topic_matrix = topic_vectorizer.fit_transform(topic_text)

    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=10
    )

    lda.fit(topic_matrix)

    feature_names = topic_vectorizer.get_feature_names_out()
    topic_rows = []

    for topic_number, weights in enumerate(lda.components_, start=1):
        top_indices = weights.argsort()[-10:][::-1]
        keywords = [
            feature_names[index]
            for index in top_indices
        ]

        topic_rows.append({
            "Topic": topic_number,
            "Keywords": ", ".join(keywords)
        })

    topic_table = pd.DataFrame(topic_rows)
    topic_table.to_csv(TABLE_DIR / "lda_topics.csv", index=False)

    topic_probabilities = lda.transform(topic_matrix)
    assigned_topics = topic_probabilities.argmax(axis=1) + 1

    topic_counts = pd.Series(
        assigned_topics
    ).value_counts().sort_index()

    topic_counts.to_csv(TABLE_DIR / "topic_distribution.csv")

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=topic_counts.index.astype(str),
        y=topic_counts.values
    )

    plt.title("Tweet Distribution Across LDA Topics")
    plt.xlabel("Topic")
    plt.ylabel("Number of Tweets")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "12_topic_distribution.png", dpi=200)
    plt.close()

except ValueError as error:
    print("LDA topic detection skipped:", error)

# --------------------------------------------------
# 18. K-MEANS CLUSTERING
# --------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    min_df=2,
    max_df=0.95
)

try:
    X = tfidf.fit_transform(data["clean_text"])

    if X.shape[0] >= 5 and X.shape[1] > 0:
        kmeans = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(X)
        data["cluster"] = cluster_labels

        cluster_counts = pd.Series(
            cluster_labels
        ).value_counts().sort_index()

        cluster_counts.to_csv(TABLE_DIR / "kmeans_cluster_counts.csv")

        terms = tfidf.get_feature_names_out()
        cluster_keywords = []

        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            top_indices = center.argsort()[-10:][::-1]
            cluster_keywords.append({
                "Cluster": cluster_id,
                "Top Terms": ", ".join(
                    terms[index] for index in top_indices
                )
            })

        pd.DataFrame(cluster_keywords).to_csv(
            TABLE_DIR / "kmeans_cluster_keywords.csv",
            index=False
        )

        plt.figure(figsize=(8, 5))

        sns.barplot(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values
        )

        plt.title("K-Means Tweet Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Tweets")

        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "13_kmeans_clusters.png", dpi=200)
        plt.close()

except ValueError as error:
    print("K-Means skipped:", error)

# --------------------------------------------------
# 19. PCA DIMENSIONALITY REDUCTION
# --------------------------------------------------

try:
    if "X" in locals() and X.shape[0] >= 3 and X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(X.toarray())

        pca_data = pd.DataFrame(
            coordinates,
            columns=["PC1", "PC2"]
        )

        pca_data.to_csv(TABLE_DIR / "pca_coordinates.csv", index=False)

        variance = pd.DataFrame({
            "Component": ["PC1", "PC2"],
            "Explained Variance": pca.explained_variance_ratio_
        })

        variance.to_csv(TABLE_DIR / "pca_variance.csv", index=False)

        plt.figure(figsize=(9, 6))

        if "cluster" in data.columns:
            pca_data["cluster"] = data["cluster"].astype(str).values
            sns.scatterplot(
                data=pca_data,
                x="PC1",
                y="PC2",
                hue="cluster",
                alpha=0.6,
                s=20
            )
        else:
            plt.scatter(
                pca_data["PC1"],
                pca_data["PC2"],
                alpha=0.5,
                s=20
            )

        plt.title("PCA Visualization of Tweet Patterns")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "14_pca_visualization.png", dpi=200)
        plt.close()

except (ValueError, MemoryError) as error:
    print("PCA skipped:", error)

# --------------------------------------------------
# 20. OUTLIER DETECTION
# --------------------------------------------------

engagement_data = np.log1p(
    data[["likes", "retweets", "replies", "total_engagement"]]
)

if len(engagement_data) >= 20:
    isolation = IsolationForest(
        contamination="auto",
        random_state=42
    )

    flags = isolation.fit_predict(engagement_data)

    data["outlier_status"] = np.where(
        flags == -1,
        "Outlier",
        "Typical"
    )

    outliers = data[data["outlier_status"] == "Outlier"]

    outliers[
        ["tweet_text", "likes", "retweets", "replies", "total_engagement"]
    ].to_csv(TABLE_DIR / "engagement_outliers.csv", index=False)

    outlier_counts = data["outlier_status"].value_counts()
    outlier_counts.to_csv(TABLE_DIR / "outlier_counts.csv")

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=data[["likes", "retweets", "replies"]]
    )

    plt.yscale("symlog")
    plt.title("Engagement Outlier Analysis")
    plt.ylabel("Engagement (symlog scale)")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "15_engagement_outliers.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 21. FINAL SUMMARY
# --------------------------------------------------

summary = {
    "Original rows": len(df),
    "Rows after preprocessing": len(data),
    "Duplicate tweets removed": duplicates_removed,
    "Empty text rows removed": empty_removed,
    "Unique hashtags": hashtags.nunique(),
    "Total likes": data["likes"].sum(),
    "Total retweets": data["retweets"].sum(),
    "Total replies": data["replies"].sum(),
    "Total engagement": data["total_engagement"].sum()
}

pd.Series(summary, name="Value").to_csv(
    TABLE_DIR / "analysis_summary.csv"
)

print("\n--- ANALYSIS COMPLETED ---")
print(pd.Series(summary))
print("\nGraphs saved in:", GRAPH_DIR)
print("Tables saved in:", TABLE_DIR)