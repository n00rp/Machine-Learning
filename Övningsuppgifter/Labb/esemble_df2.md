# Ensemble-modellering för kardiovaskulär sjukdomsprediktion (DF2)

Denna fil implementerar en ensemble-modell med hjälp av VotingClassifier för att förbättra prediktionsnoggrannheten för kardiovaskulära sjukdomar på DF2-datasetet. Koden kombinerar flera maskininlärningsalgoritmer och använder de optimala hyperparametrarna som tidigare identifierats.

## Huvudfunktioner

### load_best_parameters()
Denna funktion laddar de bästa hyperparametrarna för varje modell från en tidigare sparad CSV-fil:
- Läser in "evaluation_scores.csv" med resultat från tidigare modellträning
- Konverterar strängrepresentationer av parametrar till Python-dictionaries med `ast.literal_eval`
- Filtrerar resultaten för att hitta de bästa parametrarna för varje modelltyp baserat på valideringsnoggrannhet
- Returnerar ett dictionary med de bästa parametrarna för varje modell

### create_ensemble_model(best_params)
Skapar en ensemble-modell som kombinerar flera individuella modeller:
- XGBoost - En gradient boosting-algoritm känd för sin höga prestanda
- Logistic Regression - En linjär klassificeringsmodell
- KNN (K-Nearest Neighbors) - En instansbaserad inlärningsalgoritm
- Decision Tree - En trädbaserad modell för klassificering

Alla modeller initieras med sina optimala parametrar och kombineras i en VotingClassifier med "soft voting" för att utnyttja prediktionssannolikheter.

### train_and_evaluate_df2(X_train, X_val, X_test, y_train, y_val, y_test)
Tränar och utvärderar ensemble-modellen på DF2-datasetet:
- Laddar de bästa parametrarna och skapar ensemble-modellen
- Tränar modellen på kombinerad tränings- och valideringsdata
- Utvärderar modellen på testdata och beräknar noggrannhet
- Genererar klassificeringsrapport och confusion matrix
- Jämför ensemble-modellens prestanda med individuella modeller
- Visualiserar resultaten med stapeldiagram
- Sparar resultaten i "ensemble_results_df2.csv"

### train_best_individual_model(X_train, X_val, X_test, y_train, y_val, y_test, results_df)
Identifierar och tränar den bästa individuella modellen:
- Väljer den modell som presterade bäst bland de individuella modellerna
- Tränar den valda modellen på kombinerad tränings- och valideringsdata
- Utvärderar modellen på testdata och genererar detaljerade resultat
- Visualiserar resultaten med confusion matrix

## Tekniska detaljer

- **Röstningsmetod**: Använder "soft voting" där varje modells prediktionssannolikheter vägs samman
- **Datahantering**: Kombinerar tränings- och valideringsdata för slutlig modellträning
- **Visualisering**: Använder matplotlib och seaborn för att skapa tydliga visualiseringar av resultaten
- **Utvärderingsmått**: Fokuserar på noggrannhet (accuracy) som huvudsakligt utvärderingsmått

Denna implementation av ensemble-modellering demonstrerar hur man kan kombinera styrkan hos flera olika maskininlärningsalgoritmer för att uppnå bättre prediktionsresultat för kardiovaskulära sjukdomar.