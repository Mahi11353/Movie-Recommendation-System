import streamlit as st
import pickle
import pandas as pd
import requests
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies.pkl
movies = pickle.load(open('movies.pickle', 'rb'))

# Recompute similarity matrix
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)


# --- Streamlit UI ---
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select or type a movie name:",
    movies['title'].values
)


# --- Fetch poster function with retry logic ---
def fetch_poster(movie_id):
    api_key = "ba898f6333f8d4873ed11bd36fe39d2b"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    for _ in range(3):  # retry 3 times if connection fails
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500/{poster_path}"
            break  # exit loop if request succeeds
        except requests.exceptions.RequestException:
            time.sleep(1)  # wait 1 second before retrying
    # fallback image if poster not found or connection fails
    return "https://via.placeholder.com/500x750?text=No+Image"


# --- Recommend function ---
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters


# --- Button action ---
if st.button('🎥 Show Recommendation'):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster)
