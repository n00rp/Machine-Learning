# Model Training - Kardiovaskulär sjukdomsprediktion

Denna fil innehåller kod för att träna och utvärdera olika maskininlärningsmodeller för prediktion av kardiovaskulära sjukdomar. Koden använder GridSearchCV för att hitta optimala hyperparametrar för varje modell och utvärderar modellernas prestanda på både validerings- och testdata.

## Modeller och parametergrids

### Logistic Regression
- **C**: [0.1, 0.5, 1, 10, 100] - Regulariseringsstyrka (lägre värde = starkare regularisering)
- **penalty**: ['l1', 'l2'] - Typ av regularisering (L1 eller L2)

### XGBoost
- **learning_rate**: [0.001, 0.01, 0.1, 0.2] - Inlärningstakt för modellen
- **max_depth**: [3, 5, 7, 9] - Maximalt djup för varje beslutsträd
- **n_estimators**: [50, 100, 200] - Antal träd i ensemblen

### K-Nearest Neighbors (KNN)
- **n_neighbors**: [5, 7, 9, 11, 13] - Antal grannar att beakta
- **weights**: ['uniform', 'distance'] - Viktning av grannar
- **p**: [1, 2] - Distansmått (1 = Manhattan, 2 = Euklidisk)

### Decision Tree
- **max_depth**: [4, 5, 6, 7, 8, 9, 10] - Maximalt djup för trädet
- **min_samples_split**: [3, 5, 6, 7, 10, 20] - Minsta antal sampel för att dela en nod
- **min_samples_leaf**: [2, 3, 4, 5, 6, 7, 8] - Minsta antal sampel i ett löv

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

Denna systematiska utvärdering av modeller möjliggör en objektiv jämförelse av olika algoritmer och identifiering av den bästa modellen för kardiovaskulär sjukdomsprediktion.