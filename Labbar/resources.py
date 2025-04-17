import pandas as pd
import streamlit as st


"""

Funktioner för att läsa in datan före implementering.

"""

@st.cache_data
def read_movies():
    df_movie = pd.read_csv("Data/movies.csv")
    print(df_movie.head())

    return df_movie

@st.cache_data
def read_ratings():
    df_rating = pd.read_csv("Data/ratings.csv")
    print(df_rating.head())

    return df_rating

@st.cache_data
def read_tags():
    df_tag = pd.read_csv("Data/tags.csv")
    print(df_tag.head())

    return df_tag

@st.cache_data
def read_links():
    df_links = pd.read_csv("Data/links.csv")
    print(df_links.head())

    return df_links

