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

# Streamlit UI and CSS styling
st.set_page_config(page_title="MovieMate: Your Personal Movie Recommender", layout="wide", page_icon="🎥")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    
    .stSelectbox, .stSlider {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .movie-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .stSidebar {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    .stSidebar .css-1d391kg {
        padding: 20px;
    }
    
    .sidebar-title {
        font-size: 28px;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 30px;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sidebar-menu {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .sidebar-menu .nav-link {
        margin: 8px 0;
        border-radius: 8px;
        transition: all 0.3s;
        font-weight: 500;
        color: #333333;
    }
    
    .sidebar-menu .nav-link:hover {
        background-color: #f1f3f5;
        transform: translateX(5px);
    }
    
    .sidebar-menu .nav-link-selected {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    .sidebar-menu .nav-link-selected .icon {
        color: white !important;
    }
    
    .sidebar-footer {
        color: #6c757d;
        font-size: 12px;
        border-top: 1px solid #e0e0e0;
        padding-top: 15px;
        margin-top: 20px;
        text-align: center;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .info-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎥 MovieMate</div>', unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Homepage", "Discover Similar Movies", "Movie Explorer"],
        icons=["house-fill", "search", "compass"],
        menu_icon="camera-reels-fill",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#333333", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "padding": "10px 15px",
                "--hover-color": "#f1f3f5"
            },
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white", "font-weight": "600"},
        }
    )

    st.markdown("""
    <div class="sidebar-footer">
        © 2024 MovieMate. All rights reserved.<br>
        Powered by Muoyheang Chhea
    </div>
    """, unsafe_allow_html=True)


if selected == "Homepage":
    st.title("🎥 MovieMate: Your Personal Movie Recommender")
    
    st.markdown("""
    <div class="info-box">
        <h3>🌟 Welcome to MovieMate!</h3>
        <p>Embark on a personalized journey through the world of cinema. MovieMate is your intelligent companion in the vast universe of movies, 
        designed to uncover hidden gems and reconnect you with forgotten favorites. Our cutting-edge algorithms dive deep into user preferences 
        to bring you recommendations that resonate with your unique taste.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("🚀 Explore MovieMate Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **🎯 Tailored Recommendations**: Experience personalized movie suggestions that align with your viewing history and preferences.
        - **🔍 Similar Movie Discovery**: Uncover films that share DNA with your favorites, expanding your cinematic horizons.
        """)
    
    with col2:
        st.markdown("""
        - **🌈 Genre Deep Dives**: Explore specific genres to find new favorites and hidden classics.
        - **📊 Interactive Insights**: Visualize movie connections and trends to understand your taste better.
        """)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #e8f5e9; border-radius: 10px;">
        <h2 style="color: #4CAF50;">Our Promise</h2>
        <p style="font-size: 18px; font-style: italic;">
            "Every frame tells a story, every story finds its audience. Let MovieMate be your guide to cinematic wonders tailored just for you."
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎥 Why MovieMate?
    
    In a world overflowing with content, finding the right movie can be overwhelming. MovieMate cuts through the noise, 
    offering a curated experience that grows smarter with every interaction. Whether you're a casual viewer or a cinephile, 
    our platform adapts to your taste, ensuring every recommendation is a potential new favorite.
    
    ### 🌈 Start Your Journey
    
    Ready to transform your movie-watching experience? Dive into MovieMate and let the magic of cinema unfold before your eyes. 
    With each click, like, and watch, you're one step closer to discovering your next beloved film.
    
    <div style="text-align: center; margin-top: 30px; font-weight: bold; font-size: 20px; color: #4CAF50;">
        MovieMate: Where Every Recommendation Is a Standing Ovation Waiting to Happen! 🎭🍿
    </div>
    """, unsafe_allow_html=True)

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
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3>{movie['Movie Name']}</h3>
                        <p><strong>Similarity Score:</strong> {movie['Similarity Score']:.2f}</p>
                        <p><strong>Co-occurrence:</strong> {movie['Co-occurrence']}</p>
                        <p><strong>Genres:</strong> {', '.join([genre_list[i] for i in movie_genres.get(movie['Movie ID'], [])])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            
            similar_df = pd.DataFrame(similar_movies)
            fig = px.bar(similar_df, x='Movie Name', y='Similarity Score', title=f'Similar Movies to {movie_names[movie_id]}')
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
        st.success(f"Found {len(filtered_movies)} movies in the selected genres.")
        
        sample_size = min(10, len(filtered_movies))
        sample_movies = random.sample(filtered_movies, sample_size)
        
        for movie_id in sample_movies:
            with st.container():
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{movie_names[movie_id]}</h3>
                    <p><strong>Genres:</strong> {', '.join([genre_list[i] for i in movie_genres[movie_id]])}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No movies found matching the selected genres.")

    if filtered_movies:
        genre_counts = {genre: sum(1 for movie_id in filtered_movies if genre_list.index(genre) in movie_genres[movie_id]) for genre in genre_list[1:]}
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        fig = px.bar(genre_df, x='Genre', y='Count', title='Genre Distribution of Filtered Movies')
        st.plotly_chart(fig, use_container_width=True)

def cleanup():
    try:
        sc = SparkContext.getOrCreate()
        if sc is not None:
            sc.stop()
    except Exception as e:
        pass

import atexit
atexit.register(cleanup)
