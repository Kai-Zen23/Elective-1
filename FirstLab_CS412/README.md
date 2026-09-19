# User Clustering and Profiling — Spotify User Behavior Dataset

CS412 Elective Laboratory Activity — user segmentation using **K-Means clustering**.

## Overview

This project analyzes a synthetic Spotify user behavior dataset (50,000 users) to identify
groups of users with similar characteristics and behavioral patterns, using K-Means clustering.
The goal is to discover meaningful user profiles and propose how those insights could be applied
in a real computing system (e.g., personalization, recommendation, adaptive UX).

## Contents

| File | Description |
|---|---|
| `FirstLab_CS412_completed.ipynb` | Full notebook — code, markdown explanations, and executed outputs (plots, tables, results) |
| `FirstLab_CS412_completed.py` | Plain-Python export of the same code (no outputs) |
| `spotify_user_behavior_realistic_50000_rows.xlsx` | Dataset used (50,000 rows × 18 columns) |

## Dataset

50,000 rows, 18 columns, no missing values. Includes demographic fields (`age`, `country`),
account fields (`subscription_type`, `subscription_status`, `signup_date`), and behavioral
fields (`avg_listening_hours_per_week`, `playlists_created`, `avg_skips_per_day`,
`favorite_genre`, `primary_device`, `ad_interaction`, etc.).

## Method

1. **Exploration** — checked shape, dtypes, missing values, and cardinality of categorical fields.
2. **Feature selection** — 4 numeric features (`age`, `avg_listening_hours_per_week`,
   `playlists_created`, `avg_skips_per_day`) + 4 categorical features (`subscription_type`,
   `favorite_genre`, `primary_device`, `ad_interaction`). `user_id` excluded (pure identifier).
3. **Preparation** — `StandardScaler` on numeric features, `OneHotEncoder` on categorical features.
4. **Choosing K** — compared K = 2–8 using the elbow method (inertia) and silhouette score,
   then selected **K = 5** (within the assignment's required 3–5 profile range, and not
   meaningfully worse than neighboring K values on the data).
5. **Clustering** — fit K-Means (`random_state=42`), visualized with a 2D PCA projection.
6. **Analysis** — compared each cluster's averages against overall dataset averages to identify
   what actually separates the groups.
7. **User profiles** — derived 5 data-backed profiles from the cluster statistics.
8. **Application** — proposed a device-context-aware recommendation/UX-adaptation use case.

## Key Finding

Clusters are organized primarily by **device context + age + engagement intensity**
(listening hours, playlist creation, skip rate) — not by musical taste or subscription tier,
which were distributed almost evenly across every cluster.

## How to Run

**Google Colab**
1. Open the notebook in Colab.
2. Run the first cell — uncomment the `google.colab.files.upload()` lines if the dataset isn't
   already in the Colab session, and upload the `.xlsx` file when prompted.
3. Run all cells (`Runtime → Run all`).

**Local Jupyter**
1. Place `spotify_user_behavior_realistic_50000_rows.xlsx` in the same folder as the notebook.
2. `pip install pandas numpy matplotlib seaborn scikit-learn openpyxl`
3. Run all cells top to bottom.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

## Author

CS412 Group Lab Activity — clustering algorithm: **K-Means**.
