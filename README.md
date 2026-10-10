# IPL Twitter Trends & Topic Analysis (2020)

A Python data analytics project exploring engagement and discussion patterns in IPL 2020-related tweets. The workflow cleans tweet text, summarizes activity, and creates charts and tables for sentiment, hashtags, topics, clustering, and engagement.

## What this project covers

- **Data cleaning:** handles missing tweet text, removes empty and duplicate tweets, and standardizes text.
- **Exploratory analysis:** dataset summary, language distribution, hashtag frequency, tweet length, and activity trends where timestamps are available.
- **Engagement analysis:** likes, retweets, replies, total engagement, correlations, and highly engaged tweets.
- **Sentiment analysis:** VADER sentiment scores and distribution.
- **Topic detection:** Latent Dirichlet Allocation (LDA) to explore recurring words/topics.
- **Clustering:** K-Means groups tweets by text features.
- **Dimensionality reduction:** PCA visualization of text-feature patterns.
- **Outlier detection:** Isolation Forest flags unusual engagement records.

> Note: Topic labels and clusters are exploratory outputs, not verified categories. Sentiment tools can misread sarcasm, slang, and cricket-specific language.

## Repository layout

```text
IPL-Twitter-Trends-Analysis/
├── analysis.py
├── ipl.csv
├── requirements.txt
├── .gitignore
└── ipl_analysis_output/
    ├── graphs/   # generated visualizations
    └── tables/   # selected summary tables
```

The script creates the output folders automatically and regenerates analysis tables and charts when run.

## Requirements

- Python 3.10+ recommended
- pip

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended):

   **Windows**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the analysis:

   ```bash
   python analysis.py
   ```

Generated charts and tables will be saved under `ipl_analysis_output/`.

## Example visualizations

### Sentiment distribution
![Sentiment distribution](ipl_analysis_output/graphs/11_sentiment_distribution.png)

### Most-used hashtags
![Top hashtags](ipl_analysis_output/graphs/02_top_hashtags.png)

### Engagement distribution
![Engagement distribution](ipl_analysis_output/graphs/04_engagement_distribution.png)

### Topic distribution
![Topic distribution](ipl_analysis_output/graphs/12_topic_distribution.png)

### K-Means clusters
![K-Means clusters](ipl_analysis_output/graphs/13_kmeans_clusters.png)

### PCA visualization
![PCA visualization](ipl_analysis_output/graphs/14_pca_visualization.png)

## Data and attribution

The included `ipl.csv` is the dataset used by this analysis. Before making this repository public, confirm that the dataset's license or source terms allow redistribution. Add the original dataset URL and attribution here if required. If redistribution is not permitted, remove `ipl.csv` from the repository and document how to obtain it instead.

## Limitations

- Results depend on the dataset's coverage and quality.
- Tweets may contain sarcasm, abbreviations, multilingual text, and cricket-specific expressions that affect sentiment and topic detection.
- PCA is used for visualization, not as proof that clusters are objectively correct.
- Engagement counts describe the captured dataset and should not be interpreted as a complete measure of public opinion.

## Author

**Your Name** — replace this line with your name and optionally add your GitHub profile.
