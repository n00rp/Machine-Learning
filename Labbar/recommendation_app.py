import streamlit as st
from recommendation import recommend_movies, combine_tags_with_genres, get_knn_model, find_similar_titles
from resources import read_movies, read_tags, read_links, read_ratings
import pandas as pd
from app_layout import (
    setup_page_config, 
    display_header, 
    display_search_and_filters, 
    display_movie_details, 
    display_recommendations
)

# Anpassa sidans utseende
setup_page_config()

# Laddar data
with st.spinner('Laddar filmdata...'):
    df_movies = read_movies()
    df_tags = read_tags()
    df_links = read_links()
    df_ratings = read_ratings()  

    # Kombinerar taggar och genrer
    df_combined = combine_tags_with_genres(df_movies, df_tags)
    df_combined = pd.merge(df_combined, df_links, on="movieId", how="left")

    # Träna en KNN-modell
    knn, df_combined, vectorizer = get_knn_model(df_combined)

# Visa appens header
display_header()

# Visa sökruta och filter
search_query, exclude_series = display_search_and_filters()

# Söker efter filmer som matchar söksträngen
if search_query:
    with st.spinner('Söker efter filmer...'):
        matching_movies = find_similar_titles(search_query, df_movies)

    if not matching_movies:
        st.warning("Inga filmnamn matchade sökningen.")
    else:
        selected_movie = st.selectbox("Välj en film", matching_movies)
        
        # Visa information om den valda filmen
        if selected_movie:
            movie_data = df_combined[df_combined["title"] == selected_movie].iloc[0]
            
            # Visa filmdetaljer och få tillbaka om rekommendationsknappen klickades
            recommend_button = display_movie_details(movie_data, selected_movie)
            
            # Om rekommendationsknappen klickades, visa rekommendationer
            if recommend_button:
                with st.spinner('Letar efter liknande filmer...'):
                    recommended_movies = recommend_movies(
                        selected_movie, 
                        knn, 
                        df_combined, 
                        vectorizer,
                        ratings_df=df_ratings,  
                        exclude_series=exclude_series
                    )
                
                # Visa rekommenderade filmer
                display_recommendations(recommended_movies, selected_movie, df_combined)