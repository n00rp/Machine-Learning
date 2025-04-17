# 🎬 Filmrekommendationssystem

Det här är ett enkelt filmsystem som tipsar dig om filmer du kan gilla. Det är byggt med **Python**, **Streamlit** och **maskininlärning** i bakgrunden. Målet är att ge relevanta men också varierade rekommendationer, så det inte bara blir samma typ av film hela tiden.

---

## 🚀 Funktioner

- 🔍 Sök efter filmer i en stor databas
- 🎯 Få rekommendationer baserat på genre och innehåll
- 🔄 Variation i tipsen med hjälp av MMR-algoritmen
- 🚫 Undviker att föreslå flera filmer från samma serie
- 🖼️ Visar filmposters och länkar till IMDb/TMDB


---

## 🧠 Hur funkar det? !

### 🔡 TF-IDF

För att förstå vad en film handlar om omvandlar systemet genre och taggar till siffror med hjälp av **TF-IDF**. För att betona genrer över andra faktorer (som skådespelare) upprepas genrefältet tre gånger i den kombinerade texten innan vektorisering.

### 🧭 K-Nearest Neighbors (KNN)

För att hitta liknande filmer används K-Nearest Neighbors-algoritmen med cosinuslikhet som avståndsmått. Detta är en effektiv metod för att hitta de filmer som är mest lika en given film i vektorrummet.

### 🔄 MMR-algoritmen

En av de viktigaste delerna av systemet är användningen av MMR för att balansera mellan relevans och olikhet. Istället för att bara returnera de mest liknande filmerna väljer MMR filmer som maximerar en kombination av:
1. Relevans (lik filmen du gillar)
2. Variation (inte för lik de andra förslagen)

Jag har ställt in en balansfaktor (lambda) på `0.3` för att få en bra blandning.

### 🎬 Filtrering av serier

Filmer från samma serie (ex: Marvel eller Harry Potter) filtreras bort med hjälp av **Jaccard-likhet**, så det inte blir upprepningar.

---

## 🛠️ Struktur

Projektet består av tre huvudfiler:

- `recommendation.py` – logiken för att ta fram rekommendationer
- `app_layout.py` – hanterar hur sidan ser ut
- `recommendation_app.py` – kopplar ihop allt och kör appen

---

## ⚠️ Begränsningar

- ❄️ **Kallstartsproblem** – funkar sämre om filmen saknar genrer eller taggar
- 🧮 MMR kräver mer beräkning än enklare metoder
- 🌐 Hämtar filmposters via **OMDb API** – kräver internet uppkoppling
- 🙅‍♂️ Ingen personlig inlärning ännu – appen minns inte vad just *du* gillar

---

## 🧪 Designval

- Genrer upprepas 3x för att väga tyngre i analysen  
- Lambda i MMR satt till `0.3` för att få mer varierade tips, kan lätt ändras vid behov  
- Valde **Streamlit** istället för Dash – enklare och snabbare att jobba med

---

## ▶️ Kom igång

1. Klona repo:t  
2. Installera beroenden: `pip install -r requirements.txt`  
3. Kör applikationen: `streamlit run recommendation_app.py`

---

## Filstruktur

```
Labbar/
├── app_layout.py           # Layout och gränssnittskomponenter för Streamlit
├── recommendation.py       # Rekommendationslogik och maskininlärningsmetoder
├── recommendation_app.py   # Huvudfil för Streamlit-appen
├── resources.py            # Funktioner för att läsa in data
├── Data/                   # Katalog med CSV-filer för filmer, betyg, taggar, länkar
└── Readme.md               # Dokumentation (denna fil)
```

## 📚 Referenser

- Scikit-learn: TF-IDFVectorizer dokumentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- MovieLens dataset: https://grouplens.org/datasets/movielens/
- OMDb API: http://www.omdbapi.com/
- Medium-artikel med Python-kod för MMR: https://medium.com/@ankitgeotek/mastering-maximal-marginal-relevance-mmr-a-beginners-guide-0f383035a985
- Streamlit: https://streamlit.io/
- Blog med handledning för att skapa systemet med TF-IDF: https://dev.to/jesse_adu_akowuah_/building-a-movie-recommendation-system-with-streamlit-and-python-5bkm



