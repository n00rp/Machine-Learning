import streamlit as st
import pandas as pd
import requests
from PIL import Image, ImageDraw
from io import BytesIO

""" Har tagit hjälp från https://docs.streamlit.io/ för att ordna med layout"""

def get_movie_poster(imdb_id):
    """Hämtar filmposter från OMDb API baserat på IMDB ID, har tagit hjälp från deras wiki 
    med syntax för denna delen"""
    if pd.isna(imdb_id):
        return None
        
    try:
        # Formatera IMDB ID och hämta data
        imdb_id = str(int(imdb_id)).zfill(7)
        api_url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey=270d0027"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True" and "Poster" in data and data["Poster"] != "N/A":
                # Hämta bilden
                img_response = requests.get(data["Poster"], timeout=5)
                if img_response.status_code == 200:
                    return Image.open(BytesIO(img_response.content))
        return None
    except Exception as e:
        print(f"Fel vid hämtning av filmposter: {e}")
        return None

def get_placeholder_image(title):
    """Skapar en platshållarbild med filmtiteln när ingen poster finns tillgänglig"""
    try:
        # Skapa en enkel bild med filmtiteln
        img = Image.new('RGB', (300, 450), color=(40, 40, 60))
        d = ImageDraw.Draw(img)
        
        # Förenkla titeln för att passa på bilden
        short_title = title.split("(")[0].strip() if "(" in title else title
        if len(short_title) > 20:
            short_title = short_title[:17] + "..."
        
        # Lägg till en mörkare rektangel och text
        d.rectangle([(50, 200), (250, 250)], fill=(20, 20, 40))
        d.text((150, 225), short_title, fill=(255, 255, 255), anchor="mm")
        
        return img
    except Exception:
        return None

def get_image_for_movie(movie_data, title):
    """Hjälpfunktion för att hämta bild för en film"""
    if not pd.isna(movie_data.get("imdbId")):
        poster = get_movie_poster(movie_data["imdbId"])
        if poster:
            return poster
    
    # Använd platshållarbild om ingen poster hittades
    placeholder = get_placeholder_image(title)
    if placeholder:
        return placeholder
    
    # Fallback
    return None

def create_links_html(movie_data):
    """Skapar HTML för länkar till IMDB och TMDB"""
    links_html = []
    
    if not pd.isna(movie_data.get("imdbId")):
        imdb_id = str(int(movie_data["imdbId"])).zfill(7)
        links_html.append(f"<a href='https://www.imdb.com/title/tt{imdb_id}' target='_blank'>IMDB</a>")
    
    if not pd.isna(movie_data.get("tmdbId")):
        tmdb_id = int(movie_data["tmdbId"])
        links_html.append(f"<a href='https://www.themoviedb.org/movie/{tmdb_id}' target='_blank'>TMDB</a>")
    
    return ' | '.join(links_html) if links_html else ""

def setup_page_config():
    """Konfigurerar sidans utseende och stil"""
    st.set_page_config(page_title="Filmrekommendationer", page_icon="🎬", layout="wide")
    
    # Lägg till CSS för styling
    st.markdown("""
    <style>
        .movie-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            height: 100%;
        }
        .movie-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #1E3A8A;
            background-color: #f0f2f6;
            padding: 5px;
            border-radius: 5px;
        }
        .movie-info {
            font-size: 12px;
            margin-bottom: 3px;
        }
        .movie-links {
            margin-top: 5px;
            font-size: 12px;
        }
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .app-header {
            color: #1E3A8A;
            font-size: 42px;
        }
        .movie-poster {
            width: 100%;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Visar appens rubrik och beskrivning"""
    st.markdown('<div class="header-container"><h1 class="app-header">🎬 Filmrekommendationer</h1></div>', unsafe_allow_html=True)
    st.markdown("Hitta nya filmer baserat på dina favoriter! Skriv in en film du gillar så hittar vi liknande filmer åt dig.")

def display_search_and_filters():
    """Visar sökruta och filter"""
    col1, col2 = st.columns([3, 1])
    search_query = col1.text_input("Sök film", "")
    exclude_series = col2.checkbox("Filtrera bort film från samma serie", value=True)
    return search_query, exclude_series

def display_movie_details(movie_data, selected_movie):
    """Visar detaljerad information om en film"""
    col1, col2 = st.columns([1, 3])
    
    # Visa filmposter eller platshållarbild
    with col1:
        movie_image = get_image_for_movie(movie_data, selected_movie)
        if movie_image:
            st.image(movie_image, caption=selected_movie, use_container_width=True)
        else:
            placeholder = get_placeholder_image(selected_movie)
            if placeholder:
                st.image(placeholder, caption=selected_movie, use_container_width=True)
            else:
                st.write(f"Ingen bild tillgänglig för {selected_movie}")
    
    # Visa filminformation
    with col2:
        st.markdown(f"### {selected_movie}")
        st.markdown(f"**Genrer:** {movie_data['genres'].replace('|', ', ')}")
        
        # Visa länkar
        links_html = create_links_html(movie_data)
        if links_html:
            st.markdown(f"<div class='movie-links'><strong>Länkar:</strong> {links_html}</div>", unsafe_allow_html=True)
        
        return st.button("Rekommendera liknande filmer", key="recommend_button")

def display_recommendations(recommended_movies, selected_movie, df_combined):
    """Visar rekommenderade filmer i ett rutnät"""
    if recommended_movies is None:
        st.warning(f"Inga rekommendationer hittades för {selected_movie}")
        return
        
    st.markdown(f"## Rekommenderade filmer baserat på {selected_movie}")
    cols = st.columns(5)
    
    for i, movie_rec in enumerate(recommended_movies):
        movie_title = movie_rec["title"]
        similarity = movie_rec["similarity"]
        common_genres = movie_rec["common_genres"]
        
        movie_data = df_combined[df_combined["title"] == movie_title].iloc[0]
        
        with cols[i % 5]:
            # Visa filmkort med titel
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{movie_title}</div>
            """, unsafe_allow_html=True)
            
            # Visa filmposter eller platshållarbild
            movie_image = get_image_for_movie(movie_data, movie_title)
            if movie_image:
                st.image(movie_image, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x450?text=Film", use_container_width=True)
            
            # Visa information om filmen
            st.markdown(f"""
                <div class="movie-info">Likhet: {similarity:.2f}</div>
                <div class="movie-info">Genrer: {', '.join(common_genres)}</div>
            """, unsafe_allow_html=True)
            
            # Visa länkar
            links_html = create_links_html(movie_data)
            if links_html:
                st.markdown(f"<div class='movie-links'><strong>Länkar:</strong> {links_html}</div>", unsafe_allow_html=True)
            
            # Avsluta filmkortet
            st.markdown("</div>", unsafe_allow_html=True)