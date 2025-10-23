import pandas as pd
import numpy as np

# Load your dataset
music_df = pd.read_csv("Music Info.csv")

# Function to generate Spotify full-track URL from track ID
def track_id_to_spotify_url(track_id):
    base_url = "https://open.spotify.com/track/"
    return f"{base_url}{track_id}"

# Add Spotify URL column
music_df['spotify_url'] = music_df['spotify_id'].apply(track_id_to_spotify_url)

# Emotion-based song recommendation function
def recommend_songs_by_emotion(emotion, n=5):
    if emotion == "sad":
        recs = music_df[(music_df['valence'] < 0.4) & (music_df['energy'] < 0.5)]
    elif emotion == "happy":
        recs = music_df[(music_df['valence'] > 0.6) & (music_df['energy'] > 0.5)]
    elif emotion == "angry":
        recs = music_df[(music_df['valence'] < 0.4) & (music_df['energy'] > 0.7)]
    elif emotion == "surprise":
        recs = music_df[(music_df['valence'].between(0.4, 0.7)) & (music_df['energy'] > 0.6)]
    else:  # neutral or others
        recs = music_df[(music_df['valence'].between(0.4, 0.6)) & (music_df['energy'].between(0.4, 0.6))]
    
    # Fallback if no results
    if recs.empty:
        recs = music_df.sample(n)
    
    # Randomly sample from filtered results
    recs_sampled = recs.sample(min(n, len(recs)))

    # Convert to simple list of clickable strings for FastAPI templates
    songs = []
    for _, row in recs_sampled.iterrows():
        name = row['name']
        artist = row['artist']
        url = row['spotify_url']
        songs.append(f'<a href="{url}" target="_blank">{name} - {artist}</a>')
    
    return songs
