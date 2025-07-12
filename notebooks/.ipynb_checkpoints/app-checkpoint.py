from pyspark.sql.functions import array_contains, col, lit, desc, udf, count
from pyspark.sql.functions import explode, col, min as spark_min, max as spark_max
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import FloatType
import numpy as np
from functools import reduce
import streamlit as st
from pyspark.sql import SparkSession
import math
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

def filter_movies_by_lists(movies_df, selected_genres=None, selected_tags=None, year_range=None):
    df = movies_df

    if selected_genres:
        genre_conditions = [array_contains(col("combined_genres"), genre) for genre in selected_genres]
        df = df.filter(reduce(lambda x, y: x | y, genre_conditions))

    if selected_tags:
        tag_conditions = [array_contains(col("tags"), tag) for tag in selected_tags]
        df = df.filter(reduce(lambda x, y: x | y, tag_conditions))

    if year_range:
        df = df.filter((col("year") >= year_range[0]) & (col("year") <= year_range[1]))

    return df

def recommend_movies(movies_df, user_vector, top_k=10, filter_params=None, excluded_movie_ids=None, search_text=None):

    # 1. Filtracja danych wg wybranych parametrów
    if filter_params is None:
        filter_params = {}

    filtered_movies = filter_movies_by_lists(
        movies_df,
        selected_genres=filter_params.get("combined_genres"),
        selected_tags=filter_params.get("tags"),
        year_range=filter_params.get("year_range")
    )

    if excluded_movie_ids:
        filtered_movies = filtered_movies.filter(~col("movieId").isin(excluded_movie_ids))

    if search_text:
        filtered_movies = filtered_movies.filter(col("clean_title").rlike(f"(?i){search_text}"))

    # 2. Jeśli cold start (user_vector None lub zero), wybierz top N najpopularniejszych i najlepiej ocenianych, najnowszych
    if user_vector is None or np.linalg.norm(user_vector) == 0:
        movies_with_score = filtered_movies.withColumn(
            "popularity_score",
            col("avg_rating") * F.log10(col("num_ratings") + 1)
        )
        popular_movies = movies_with_score.orderBy(
            desc("popularity_score"), desc("year")
        ).limit(top_k)
        return popular_movies

    def cosine_similarity(v):
        if v is None:
            return 0.0
        arr = np.array(v.toArray()) if hasattr(v, "toArray") else np.array(v)
        norm_v = np.linalg.norm(arr)
        norm_user = np.linalg.norm(user_vector)
        if norm_v == 0 or norm_user == 0:
            return 0.0
        return float(np.dot(arr, user_vector) / (norm_v * norm_user))
    
    cosine_sim_udf = udf(cosine_similarity, FloatType())
    
    scored_movies = filtered_movies.withColumn("similarity", cosine_sim_udf(col("final_features")))
    normalized_movies = scored_movies.withColumn("log_popularity", F.log10(col("num_ratings") + 1))

    w_similarity = 0.5
    w_popularity = 0.03
    w_rating = 0.01

    ranked_movies = normalized_movies.withColumn(
        "ranking_score",
        (w_similarity * col("similarity")) +
        (w_popularity * col("log_popularity")) +
        (w_rating * col("avg_rating"))
    )

    return ranked_movies.orderBy(desc("ranking_score")).limit(top_k)

def build_user_vector(user_ratings_dict):
    vectors = []
    for rating, vec in user_ratings_dict.values():
        if vec is None:
            continue
        np_vec = np.array(vec)
        weighted_vec = np_vec * rating
        vectors.append(weighted_vec)

    if not vectors:
        return None  # cold start

    return np.mean(vectors, axis=0)
    
def main():
    st.title("Rekomendacje filmów 🎬")

    spark = SparkSession.builder \
    .appName("RecommendationModel") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

    movies = spark.read.parquet("hdfs:///output/movie_vectors.parquet")

    genres_df = movies.select(explode(col("combined_genres")).alias("genre")) \
    .filter(~col("genre").isin("\\N", "(no genres listed)")) \
    .distinct()

    genres = [row["genre"] for row in genres_df.collect()]
    genres.sort()

    tag_counts_df = movies.select(explode("tags").alias("tag")) \
    .groupBy("tag") \
    .agg(count("*").alias("count")) \
    .orderBy(desc("count")) \
    .limit(50)

    tag_list = [row["tag"] for row in tag_counts_df.collect()]
    
    years = movies.select(spark_min("year").alias("min_year"), spark_max("year").alias("max_year")).collect()[0]
    year_min, year_max = int(years["min_year"]), int(years["max_year"])
    years_list = list(range(year_min, year_max + 1))
    
    # --- Interfejs użytkownika ---

    search_text = st.text_input("Wyszukaj film po tytule (część nazwy)")
    
    selected_genres = st.multiselect("Wybierz gatunki filmów:", genres)

    selected_tags = st.multiselect("Wpisz lub wybierz tagi:", tag_list)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Wybierz rok początkowy:", years_list, index=100)
    with col2:
        end_year = st.selectbox("Wybierz rok końcowy:", years_list, index=len(years_list) - 1)
    
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    
    year_range = (start_year, end_year)

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    if st.button("🗑️ Wyczyść moje oceny"):
        st.session_state.user_ratings = {}

    filter_params = {
        "combined_genres": selected_genres if selected_genres else None,
        "tags": selected_tags if selected_tags else None,
        "year_range": year_range
    }

     # --- Buduj wektor użytkownika ---
    user_vector = build_user_vector(st.session_state.user_ratings)
    excluded_ids = list(st.session_state.user_ratings.keys())

    recommendations_df = recommend_movies(movies, user_vector, top_k=10, filter_params=filter_params, excluded_movie_ids=excluded_ids, search_text=search_text)

    # Wyświetlanie wyników
    st.subheader("Rekomendowane filmy:")
    
    for row in recommendations_df.collect():
        movie_id = row["movieId"]
        title_display = f"{row['clean_title']} ({row['year']}) – ocena: {row['avg_rating']:.2f}"
        st.write(title_display)

        slider_key = f"rating_{movie_id}"
        user_rating = st.slider("Twoja ocena:", 0.0, 5.0, st.session_state.user_ratings.get(movie_id, (0.0,))[0], 0.5, key=slider_key)

        if user_rating > 0:
            features_array = row["final_features"].toArray()
            st.session_state.user_ratings[movie_id] = (user_rating, row["final_features"].toArray())
                
if __name__ == "__main__":
    main()