# Sammanfattning av Modellträning och Utvärdering

## Översikt

Detta dokument sammanfattar funktionaliteten hos två filer som används för att träna och utvärdera maskininlärningsmodeller för klassificering:

1. **ensemble.py** - Hanterar träning av den bästa modellen på kombinerad data
2. **evaluate_best_model.py** - Utvärderar olika modeller för att hitta den bästa

Båda filerna arbetar med samma uppsättning modeller och hyperparametrar som har optimerats för DF2-datasetet.

## Gemensamma komponenter

Båda filerna delar flera gemensamma komponenter:

- **BEST_PARAMS**: Ett dictionary med optimala hyperparametrar för varje modell baserat på tidigare tester på DF2
- **create_models()**: Skapar modeller med de optimala hyperparametrarna utifrån tidigare tester
- **create_voting_classifier()**: Skapar en ensemble-modell (VotingClassifier) från de individuella modellerna
- **combine_data()**: Kombinerar tränings- och valideringsdata för att maximera mängden träningsdata

## evaluate_best_model.py

Denna fil fokuserar på att utvärdera och jämföra olika modeller för att hitta den bästa.

### Huvudfunktioner evaluate_best_model.py:

1. **evaluate_model()**: 
   - Utvärderar en modell på testdata och returnerar noggrannheten
   - Används för att jämföra prestanda mellan olika modeller

2. **find_best_model()**:
   - Tränar flera olika modeller på kombinerad tränings- och valideringsdata
   - Utvärderar varje modell på testdata
   - Jämför resultaten och identifierar den bästa modellen
   - Skapar visualiseringar av modellernas prestanda
   - Returnerar namnet på den bästa modellen, alla tränade modeller och en dataframe med resultat


## ensemble.py

Denna fil fokuserar på att träna den bästa modellen på kombinerad data.

### Huvudfunktioner ensemble.py:

1. **combine_data()**:
   - Kombinerar tränings- och valideringsdata för att maximera mängden träningsdata
   - Används för att skapa ett större dataset för träning

2. **evaluate_model()**:
   - Utvärderar en modell på testdata
   - Skapar en confusion matrix och klassificeringsrapport
   - Visualiserar resultaten med en heatmap

3. **train_best_model()**:
   - Tränar den bästa modellen (eller en specifik modell) på kombinerad data
   - Utvärderar modellen på testdata
   - Returnerar den tränade modellen


## Viktiga observationer

1. **Testdata används endast för utvärdering**, aldrig för träning
2. **Träning sker på kombinerad tränings- och valideringsdata** för att maximera mängden träningsdata
3. **Båda filerna använder samma optimala hyperparametrar** för att säkerställa konsistenta resultat
4. **Modellerna utvärderas med noggrannhet (accuracy)** som huvudmått
5. **Visualiseringar hjälper till att förstå modellernas prestanda**

## Slutsats

Dessa två filer erbjuder en komplett pipeline för att:
1. Utvärdera och jämföra olika modeller
2. Hitta den bästa modellen
3. Träna den bästa modellen på maximalt tillgänglig data
4. Utvärdera den slutliga modellen

Genom att separera utvärdering och träning i olika filer blir koden mer modulär och lättare att underhålla.
