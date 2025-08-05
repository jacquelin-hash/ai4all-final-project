# 🎶 VibeNet — Mood-Based Song Clustering & Recommendation System

**VibeNet** is an unsupervised machine learning model that classifies songs into mood-based clusters using Spotify audio features. By analyzing attributes like valence, energy, danceability, acousticness, and tempo, VibeNet allows users to receive **mood-aligned song recommendations** in real time.

> **Slogan:** *“Feel the Mood, Hear the Vibe.”*

---

## 🚀 Features
- Clusters songs into **mood groups** using **K-Means Clustering**.
- **Auto-generates mood labels** based on cluster center traits.
- Interactive **Streamlit Web App UI** for mood prediction & recommendations.
- Recommends **Top 5 similar songs** based on user's mood preferences.
- Visualizes cluster traits with **Radar Charts** and **PCA Scatter Plots**.
- Ready for deployment with **model saving/loading functionality**.

---

## 🧠 Machine Learning Model
- **Algorithm:** K-Means Clustering (Unsupervised Learning)
- **Input Features:**
  - Valence
  - Energy
  - Danceability
  - Acousticness
  - Tempo
- **Output:** Mood Cluster Label & Song Recommendations
- Future plans to enhance model using **Gaussian Mixture Models (GMM)** for soft clustering.

---

## 📊 Data
- Dataset derived from **Spotify audio features**.
- Process includes:
  - Data Cleaning (duplicates removal)
  - Feature Scaling with StandardScaler
  - Clustering & Mood Labeling
  - Exported Clustered Dataset (`clustered_songs_with_moods.csv`)

---
