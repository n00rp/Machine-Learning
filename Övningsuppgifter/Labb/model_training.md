# Model Training - Kardiovaskulär sjukdomsprediktion

Denna fil innehåller kod för att träna och utvärdera olika maskininlärningsmodeller för prediktion av datan. Koden använder GridSearchCV för att hitta optimala hyperparametrar för varje modell och utvärderar modellernas prestanda på både validerings- och testdata.

## Huvudfunktioner

### train_and_evaluate_model
Denna funktion tränar en modell med GridSearchCV för att hitta optimala hyperparametrar och utvärderar modellens prestanda. Funktionen:
1. Tränar modellen med GridSearchCV på träningsdata
2. Utvärderar modellen på valideringsdata
3. Testar modellen på testdata
4. Sparar resultaten (modellnamn, bästa parametrar, valideringsnoggrannhet, testnoggrannhet) i en CSV-fil
5. Returnerar den bästa modellen samt noggrannhet på validerings- och testdata

### train_models
Denna funktion tränar och utvärderar alla modeller (XGBoost, Logistic Regression, KNN, Decision Tree) på både DF1 och DF2 dataset. För varje modell:
1. Initierar modellen med grundläggande parametrar
2. Anropar train_and_evaluate_model med lämplig parametergrid
3. Sparar resultaten i "evaluation_scores.csv"

## Datahantering
Koden förutsätter att data redan är förbehandlad och uppdelad i tränings-, validerings- och testuppsättningar för både DF1 och DF2 dataset. Alla modeller tränas och utvärderas på båda dataseten för att möjliggöra jämförelse.

## Resultatlagring
Resultaten från modellträningen sparas i "evaluation_scores.csv" med följande kolumner:
- Model: Modellnamn och vilket dataset som användes
- Best Parameters: De optimala hyperparametrarna som hittades med GridSearchCV
- Validation Accuracy: Noggrannhet på valideringsdata
- Test Accuracy: Noggrannhet på testdata

Denna systematiska utvärdering av modeller möjliggör en objektiv jämförelse av olika algoritmer och identifiering av den bästa modellen.