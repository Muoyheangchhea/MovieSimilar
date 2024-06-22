import math
import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark import SparkConf, SparkContext
from streamlit_option_menu import option_menu
import random

def init_spark():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

def load_movie_names():
    movie_names = {}
    with open("ml-100k/u.item", encoding='ISO-8859-1') as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

def load_ratings_data(sc):
    lines = sc.textFile("ml-100k/u.data")
    ratings = lines.map(lambda x: x.split()).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))
    return ratings

def make_pairs(user_ratings):
    (user, ratings) = user_ratings
    ratings = list(ratings)
    pairs = []
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            pairs.append(((ratings[i][0], ratings[j][0]), (ratings[i][1], ratings[j][1])))
    return pairs

def filter_duplicates(movie_pair):
    movie1, movie2 = movie_pair[0]
    return movie1 < movie2

def compute_cosine_similarity(rating_pairs):
    num_pairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in rating_pairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        num_pairs += 1

    denominator = math.sqrt(sum_xx) * math.sqrt(sum_yy)
    if denominator == 0:
        score = 0
    else:
        score = sum_xy / float(denominator)
    
    return (score, num_pairs)

def find_similar_movies(sc, score_threshold=0.97, co_occurrence_threshold=50):
    ratings = load_ratings_data(sc)
    ratings_by_user = ratings.groupByKey()
    movie_pairs = ratings_by_user.flatMap(make_pairs)
    filtered_movie_pairs = movie_pairs.filter(filter_duplicates)
    movie_pair_ratings = filtered_movie_pairs.groupByKey()
    movie_pair_similarities = movie_pair_ratings.mapValues(compute_cosine_similarity).cache()
    
    filtered_results = movie_pair_similarities.filter(
        lambda pairSim: pairSim[1][0] > score_threshold and pairSim[1][1] > co_occurrence_threshold
    )
    
    results = filtered_results.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(False)
    return results

def get_similar_movies(movie_id, results, movie_names, top_n=10):
    top_similar_movies = []
    for result in results.take(top_n):
        (sim, pair) = result
        similar_movie_id = pair[1] if pair[0] == movie_id else pair[0]
        top_similar_movies.append({
            'Movie ID': similar_movie_id,
            'Movie Name': movie_names[similar_movie_id],
            'Similarity Score': sim[0],
            'Co-occurrence': sim[1]
        })
    return top_similar_movies

def get_movie_genres():
    genres = {}
    with open("ml-100k/u.item", encoding='ISO-8859-1') as f:
        for line in f:
            fields = line.split('|')
            movie_id = int(fields[0])
            movie_genres = [idx for idx, genre in enumerate(fields[5:24]) if genre == '1']
            genres[movie_id] = movie_genres
    return genres

def make_pairs(user_ratings):
    (user, ratings) = user_ratings
    ratings = list(ratings)
    pairs = []
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            pairs.append(((ratings[i][0], ratings[j][0]), (ratings[i][1], ratings[j][1])))
    return pairs

def filter_duplicates(movie_pair):
    movie1, movie2 = movie_pair[0]
    return movie1 < movie2

st.set_page_config(page_title="MovieMate: Your Personal Movie Recommender", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    .stSelectbox, .stSlider {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .movie-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
    .stSidebar .css-1d391kg {
        padding: 10px;
    }
    .stSidebar .css-1d391kg h3, .stSidebar .css-1d391kg h4 {
        color: #333;
    }
    .css-1v3fvcr {
        margin-bottom: 30px;
    }
    /* Sidebar Styles */
    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 20px;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sidebar-menu {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sidebar-menu .nav-link {
        margin: 8px 0;
        border-radius: 5px;
        transition: all 0.3s;
        font-weight: 500;
    }
    .sidebar-menu .nav-link:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
    }
    .sidebar-menu .nav-link-selected {
        background-color: #007bff !important;
        color: white !important;
    }
    .sidebar-menu .nav-link-selected .icon {
        color: white !important;
    }
    .sidebar-footer {
        color: #6c757d;
        font-size: 12px;
        border-top: 1px solid #dee2e6;
        padding-top: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 MovieMate</div>', unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Homepage", "Discover Similar Movies", "Movie Explorer"],
        icons=["house-fill", "search", "film"],
        menu_icon="camera-reels-fill",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#000000", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "padding": "10px 15px",
                "--hover-color": "#e9ecef"
            },
            "nav-link-selected": {"background-color": "#007bff", "color": "white", "font-weight": "600"},
        }
    )

    st.markdown("""
    <div class="sidebar-footer">
        © 2024 MovieMate. All rights reserved.<br>
        Powered by Muoyheang Chhea
    </div>
    """, unsafe_allow_html=True)

if selected == "Homepage":
    st.title("📽️ MovieMate: Your Personal Movie Recommender")
    
    st.markdown("""
    ### 👋 Welcome to MovieMate!
    
    Discover your next favorite movie with our advanced recommendation system. 
    MovieMate is not just another movie recommendation system; it's your personal guide to the vast world of cinema. 
    Our state-of-the-art algorithms analyze user ratings and preferences to bring you tailored movie suggestions that align perfectly with your taste.

    ### 🌟 Our Key Features

    - **Personalized Recommendations**: Get movie suggestions tailored to your taste.
    - **Discover Similar Movies**: Discover films that share characteristics with your favorites.
    - **Movie Explorer**: Dive into specific genres and uncover new favorites.
    - **Interactive Visualizations**: See movie similarities and trends at a glance.
    ##### *Start your journey with MovieMate today and transform the way you discover and enjoy movies!🫶*
    """)

elif selected == "Discover Similar Movies":
    st.title("🔍 Discover Similar Movies")
    
    sc = init_spark()
    if sc is not None:
        movie_names = load_movie_names()
        movie_genres = get_movie_genres()
        genre_list = ["Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

        col1, col2 = st.columns(2)
        with col1:
            selected_movie_name = st.selectbox("Select a Movie", options=list(movie_names.values()))
        with col2:
            top_n = st.slider("Number of similar movies", 5, 20, 10)

        movie_id = next(key for key, value in movie_names.items() if value == selected_movie_name)
        if st.button("Find Similar Movies"):
            with st.spinner('Finding similar movies...'):
                results = find_similar_movies(sc)
                similar_movies = get_similar_movies(movie_id, results, movie_names, top_n)
            st.success(f"Top {top_n} similar movies for '{movie_names[movie_id]}':")
            
            for movie in similar_movies:
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(f"https://picsum.photos/200/300?random={movie['Movie ID']}&cachebuster={random.randint(1, 1000)}", width=150)
                    with col2:
                        st.markdown(f"### {movie['Movie Name']}")
                        st.write(f"Similarity Score: {movie['Similarity Score']:.2f}")
                        st.write(f"Co-occurrence: {movie['Co-occurrence']}")
                        genres = [genre_list[i] for i in movie_genres.get(movie['Movie ID'], [])]
                        st.write(f"Genres: {', '.join(genres)}")
                st.markdown("---")

            df = pd.DataFrame(similar_movies)
            fig = px.bar(df, x='Movie Name', y='Similarity Score', 
                         hover_data=['Co-occurrence'], 
                         title=f"Similarity Scores for Movies Similar to '{movie_names[movie_id]}'")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Failed to initialize SparkContext. Please check your Spark installation.")

elif selected == "Movie Explorer":
    st.title("🎞️ Movie Explorer")
    
    movie_names = load_movie_names()
    movie_genres = get_movie_genres()
    genre_list = ["Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    selected_genres = st.multiselect("Select Genres", options=genre_list[1:])

    filtered_movies = [movie_id for movie_id, genres in movie_genres.items() 
                       if all(genre_list.index(genre) in genres for genre in selected_genres)]

    if filtered_movies:
        st.write(f"Found {len(filtered_movies)} movies in the selected genres.")
        
        sample_size = min(10, len(filtered_movies))
        sample_movies = random.sample(filtered_movies, sample_size)
        
        for movie_id in sample_movies:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(f"https://picsum.photos/200/300?random={movie_id}&cachebuster={random.randint(1, 1000)}", width=150)
                with col2:
                    st.markdown(f"### {movie_names[movie_id]}")
                    genres = [genre_list[i] for i in movie_genres[movie_id]]
                    st.write(f"Genres: {', '.join(genres)}")
            st.markdown("---")
    else:
        st.write("No movies found with the selected genres. Try selecting different genres.")


def cleanup():
    try:
        sc = SparkContext.getOrCreate()
        if sc is not None:
            sc.stop()
    except Exception as e:
        pass

import atexit
atexit.register(cleanup)
