from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import requests
import urllib.parse
import os

# Initialize Flask app
app = Flask(__name__)

# Sample data - Replace with your actual dataset
df = pd.read_csv('Cleaned_Anime.csv')

# Combine features (title + description or other features you may have)
combined_features = df['Title'] + ' ' + df['Genre']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

def get_anime_details(title):
	# """Fetch anime details like poster and rating from Jikan API."""
	# encoded_title = urllib.parse.quote(title)
	url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
	response = requests.get(url)

	if response.status_code == 200:
		data = response.json()
		if data["data"]:
			anime_info = data["data"][0]
			return {
				"poster": anime_info["images"]["jpg"]["image_url"],
				"rating": anime_info["score"]
			}
	return {"poster": "static/images/default_poster.png", "rating": "N/A"}

@app.route('/', methods=['GET', 'POST'])
def index():
	recommended_animes = []
	if request.method == 'POST':
		input_anime = request.form['anime_name']
		
		# Extract close matches
		list_of_animes_in_database = df['Title'].tolist()
		close_match = process.extract(input_anime, list_of_animes_in_database)
		
		closest_match = close_match[0]
		index_of_anime = closest_match[2]
		
		# Find similar animes
		similar_animes = list(enumerate(similarity[index_of_anime]))
		sorted_similar_animes = sorted(similar_animes, key=lambda x: x[1], reverse=True)
		
		# Get top 10 similar animes
		# recommended_animes = []
		for i, anime in enumerate(sorted_similar_animes):
			if i < 12:
				anime_data = df.iloc[anime[0]]
				anime_details = get_anime_details(anime_data['Title'])

				recommended_animes.append({
					'title': anime_data['Title'],
					'genre': anime_data['Genre'],
					'poster': anime_details['poster'],
					'rating': anime_details['rating']
        })
			
	return render_template('index.html', recommended_animes=recommended_animes)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
