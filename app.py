import streamlit as st
import numpy as np
import pandas as pd
import joblib



df_songs = pd.read_csv('cleaned_spotify_dataset.csv')  

# Load Scaler and KMeans model
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('song_mood_kmeans.joblib')

# Select only feature columns to scale
features = df_songs[['valence', 'energy', 'danceability', 'acousticness', 'tempo']]

# Scale features
features_scaled = scaler.transform(features)

# Predict cluster for each song
df_songs['cluster'] = kmeans.predict(features_scaled)

# Export clustered dataset
df_songs.to_csv('clustered_songs_with_moods.csv', index=False)

print("Clustered dataset saved as 'clustered_songs_with_moods.csv'")




# Load saved model and scaler
kmeans = joblib.load('song_mood_kmeans.joblib')
scaler = joblib.load('scaler.joblib')

# Load clustered song dataset (with song names & cluster labels)
df_songs = pd.read_csv('clustered_songs_with_moods.csv')

# Streamlit App
st.title("🎶 Song Mood Classifier & Recommender")

# Input sliders
valence = st.slider('Valence (0 = sad, 1 = happy)', 0.0, 1.0, 0.5)
energy = st.slider('Energy (0 = mellow, 1 = energetic)', 0.0, 1.0, 0.5)
danceability = st.slider('Danceability (0 = stiff, 1 = groovy)', 0.0, 1.0, 0.5)
acousticness = st.slider('Acousticness (0 = electronic, 1 = acoustic)', 0.0, 1.0, 0.5)
tempo = st.slider('Tempo (BPM)', 60, 200, 120)

# Predict Button
if st.button('Predict Mood & Recommend Songs'):
    # Scale input features
    user_input = np.array([[valence, energy, danceability, acousticness, tempo]])
    user_input_scaled = scaler.transform(user_input)

    # Predict cluster
    cluster = kmeans.predict(user_input_scaled)[0]
    st.success(f"Predicted Mood Cluster: {cluster}")

    # Filter songs from the same cluster
    cluster_songs = df_songs[df_songs['cluster'] == cluster].copy()

    # Calculate similarity (Euclidean distance)
    cluster_songs['similarity'] = cluster_songs.apply(
        lambda row: np.linalg.norm(user_input - row[['valence', 'energy', 'danceability', 'acousticness', 'tempo']].values), axis=1)

    # Recommend top 5 closest songs
    top_songs = cluster_songs.nsmallest(5, 'similarity')

    # Display recommendations
    st.subheader("🎵 Top 5 Recommended Songs:")
    for idx, row in top_songs.iterrows():
        st.write(f"• {row['track_name']}")
