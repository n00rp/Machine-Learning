from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import re



def recommend_movies(movie_title, knn, df_combined, vectorizer, ratings_df=None, top_n=5, exclude_series=True):
    # Rekommenderar filmer baserat på vilken film man skrivit in.
    if not movie_title or df_combined[df_combined["title"] == movie_title].empty:
        return None
    
    # Hämta index för den valda filmen.
    movie_idx = df_combined.index[df_combined["title"] == movie_title][0]
    movie_data = df_combined.iloc[movie_idx]
    
    # Gör om filmens kombinerade text (genrer + taggar) till en vektor
    movie_vector = vectorizer.transform([movie_data["combined"]])
    
    # Optimera genom att hämta fler kandidater på en gång
    n_candidates = min(100, len(df_combined))  # Begränsa antalet kandidater
    distances, indices = knn.kneighbors(movie_vector, n_neighbors=n_candidates)
    similarities = 1 - distances.flatten()
    
    # Skapa en lista med kandidater (exklusive den valda filmen)
    candidates = []
    base_title = movie_title.split(" (")[0].strip().lower()
    
    for i, idx in enumerate(indices[0][1:]):  # Skippa första (samma film)
        candidate_data = df_combined.iloc[idx]
        candidate_title = candidate_data["title"]
        
        # Snabb filtrering av serier om det behövs
        if exclude_series:
            candidate_base = candidate_title.split(" (")[0].strip().lower()
            if base_title in candidate_base or candidate_base in base_title:
                continue
        
        # Beräkna rating score
        rating_score = 0.5  # Default värde om inga betyg finns
        if ratings_df is not None:
            movie_ratings = ratings_df[ratings_df['movieId'] == candidate_data['movieId']]
            if not movie_ratings.empty:
                avg_rating = movie_ratings['rating'].mean()
                rating_count = len(movie_ratings)
                # Normalisera betyget och ta hänsyn till antal betyg
                rating_score = (avg_rating - 1) / 4  # Normalisera till 0-1
                confidence = min(rating_count / 100, 1)  # Konfidenspoäng baserad på antal betyg
                rating_score = rating_score * confidence + 0.5 * (1 - confidence)
        
        # Kombinera scores
        combined_score = 0.8 * similarities[i+1] + 0.2 * rating_score
        
        candidates.append({
            "title": candidate_title,
            "similarity": combined_score,
            "raw_similarity": similarities[i+1],
            "rating_score": rating_score,
            "common_genres": set(candidate_data["genres"].split("|")) & set(movie_data["genres"].split("|")),
            "vector": vectorizer.transform([candidate_data["combined"]])
        })
        
        if len(candidates) >= 50:  # Begränsa antalet kandidater för MMR
            break
    
    # Använder MMR (Maximal Marginal Relevanceför att ge bättre rekommendationer
    lambda_param = 0.3
    diversified_recommendations = []
    
    while len(diversified_recommendations) < top_n and candidates:
        max_mmr = -float("inf")
        max_mmr_idx = -1
        
        for i, candidate in enumerate(candidates):
            relevance = candidate["similarity"]
            diversity = 1.0
            
            if diversified_recommendations:
                max_similarity = 0
                for rec in diversified_recommendations:
                    sim = 1 - np.linalg.norm(candidate["vector"].toarray() - rec["vector"].toarray())
                    max_similarity = max(max_similarity, sim)
                diversity = 1 - max_similarity
            
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            if mmr_score > max_mmr:
                max_mmr = mmr_score
                max_mmr_idx = i
        
        best_candidate = candidates.pop(max_mmr_idx)
        recommendation = {
            "title": best_candidate["title"],
            "similarity": best_candidate["similarity"],
            "raw_similarity": best_candidate["raw_similarity"],
            "rating_score": best_candidate["rating_score"],
            "common_genres": best_candidate["common_genres"],
            "mmr_score": max_mmr,
            "vector": best_candidate["vector"]
        }
        diversified_recommendations.append(recommendation)
    
    # Ta bort vector-fältet från de slutliga rekommendationerna
    return [{k: v for k, v in rec.items() if k != "vector"} for rec in diversified_recommendations]


def combine_tags_with_genres(df_movies, df_tags):
    # Ersätt NaN-värden med tomma strängar i 'tag'-kolumnen
    df_tags['tag'] = df_tags['tag'].fillna("")
    
    # Gruppera taggar per film och slå ihop till en sträng
    df_tags_grouped = df_tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index(name="tags")
    df_combined = pd.merge(df_movies, df_tags_grouped, on="movieId", how="left")
    df_combined["tags"] = df_combined["tags"].fillna("")
    
    # Ge genrer högre vikt genom att upprepa dem tre gånger
    # Detta gör att genrerna får större betydelse i TF-IDF-vektoriseringen
    df_combined["combined"] = df_combined["genres"] + " " + df_combined["genres"] + " " + df_combined["genres"] + " " + df_combined["tags"]
    return df_combined

def get_knn_model(df_combined):
    # Använd TF-IDF för att skapa vektorer av kombinerade taggar och genrer.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_combined["combined"])

    # Träna en KNN-modell
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(tfidf_matrix)
    return knn, df_combined, vectorizer


def find_similar_titles(search_query, df_movies, max_results=5):
    # Letar fram filmnamn som är lika med söksträngen
    if not search_query:
        return []
    
    # Gör om till liknande för att göra sökningen mer flexibel
    search_query = search_query.lower()
    
    # Hittar liknande titlar om det exakta söksträngen inte finns
    matching_titles = df_movies[df_movies["title"].str.lower().str.contains(search_query)]

    return matching_titles["title"].head(max_results).tolist()