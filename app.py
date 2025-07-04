from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and dataset
model = pickle.load(open('spotify_model.pkl', 'rb'))
df = pd.read_csv('SpotifyFeatures.csv')

# Build mapping dictionaries
genre_pop_map = df.groupby('genre')['popularity'].mean().to_dict()
artist_pop_map = df.groupby('artist_name')['popularity'].mean().to_dict()

# Genre list for dropdown
genre_labels = [(g.replace('_', ' ').title(), g) for g in genre_pop_map.keys()]

@app.route('/')
def home():
    return render_template('index.html', genre_labels=genre_labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract input values
    artist_name = data.get('artist_name')
    genre = data.get('genre')

    artist_avg_popularity = artist_pop_map.get(artist_name, df['popularity'].mean())
    genre_avg_popularity = genre_pop_map.get(genre, df['popularity'].mean())
    duration_ms = float(data['duration_ms'])
    duration_mins = duration_ms / 60000

    # Final model input order
    features = [
        float(data['acousticness']),
        float(data['danceability']),
        float(data['energy']),
        float(data['instrumentalness']),
        int(data['key']),
        float(data['liveness']),
        float(data['loudness']),
        int(data['mode']),
        float(data['speechiness']),
        float(data['tempo']),
        int(data['time_signature']),
        float(data['valence']),
        artist_avg_popularity,
        float(data['danceability']) / (float(data['loudness']) + 1e-5),
        float(data['energy']) * float(data['valence']),
        duration_mins,
        genre_avg_popularity
    ]

    prediction = model.predict([features])[0]

    return render_template('index.html',
                           prediction_text=f'Predicted Popularity: {round(prediction, 2)}',
                           genre_labels=genre_labels,
                           artist_name=artist_name,
                           track_name=data.get('track_name'))

if __name__ == '__main__':
    app.run(debug=True)
