import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

"""
Det skall sägas att jag tagit mycket hjälp med syntaxen i denna filen, dock har jag inte kopierat något 
direkt in i filen. Både från klasskamrater och på nätet.

"""



def load_best_parameters():
    """
    Laddar in de bästa parametrarna från den sparade CSV-filen
    """
    results_df = pd.read_csv("evaluation_scores.csv")
    
    # Konvertera strängar till dictionaries för Best Parameters-kolumnen
    results_df['Best Parameters'] = results_df['Best Parameters'].apply(ast.literal_eval)
    
    # Hitta de bästa modellerna baserat på valideringsnoggrannhet
    best_models = {}
    
    # Hitta bästa XGBoost-modellen
    xgb_models = results_df[results_df['Model'].str.contains('XGBoost')]
    best_xgb = xgb_models.loc[xgb_models['Validation Accuracy'].idxmax()] # Tog hjälp av Stackoverflow för att förstå idxmax
    best_models['XGBoost'] = best_xgb['Best Parameters']
    
    # Hitta bästa Logistic Regression-modellen
    log_reg_models = results_df[results_df['Model'].str.contains('Logistic Regression')]
    best_log_reg = log_reg_models.loc[log_reg_models['Validation Accuracy'].idxmax()]
    best_models['Logistic Regression'] = best_log_reg['Best Parameters']
    
    # Hitta bästa KNN-modellen
    knn_models = results_df[results_df['Model'].str.contains('KNN')]
    best_knn = knn_models.loc[knn_models['Validation Accuracy'].idxmax()]
    best_models['KNN'] = best_knn['Best Parameters']
    
    # Hitta bästa Decision Tree-modellen
    dt_models = results_df[results_df['Model'].str.contains('Decision Tree')]
    best_dt = dt_models.loc[dt_models['Validation Accuracy'].idxmax()]
    best_models['Decision Tree'] = best_dt['Best Parameters']
    
    print("Bästa parametrar laddade:")
    for model, params in best_models.items():
        print(f"{model}: {params}")
    
    return best_models

def create_ensemble_model(best_params):
    """
    Skapar en ensemble-modell med de bästa parametrarna
    """
    # XGBoost med optimala parametrar
    xgb_model = XGBClassifier(random_state=42, **best_params['XGBoost'])
    
    # Logistic Regression med optimala parametrar
    log_reg_model = LogisticRegression(random_state=42, solver='liblinear', **best_params['Logistic Regression'])
    
    # KNN med optimala parametrar
    knn_model = KNeighborsClassifier(**best_params['KNN'])
    
    # Decision Tree med optimala parametrar
    dt_model = DecisionTreeClassifier(random_state=42, **best_params['Decision Tree'])
    
    # Skapa VotingClassifier med alla modeller
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('log_reg', log_reg_model),
            ('knn', knn_model),
            ('dt', dt_model)
        ],
        voting='soft'  # Använd 'soft' för att använda sannolikheter
    )
    
    return ensemble_model, {
        'XGBoost': xgb_model,
        'Logistic Regression': log_reg_model,
        'KNN': knn_model,
        'Decision Tree': dt_model
    }

def train_and_evaluate_df2(X2_train_scaled, X2_val_scaled, X2_test_scaled, y2_train, y2_val, y2_test):
    """
    Tränar och utvärderar ensemble-modellen på DF2-datasetet
    """
    # Ladda bästa parametrar
    best_params = load_best_parameters()
    
    # Skapa ensemble-modellen med bästa parametrar
    ensemble_model, individual_models = create_ensemble_model(best_params)
    
    print("Tränar ensemble-modellen på DF2-datasetet...")
    # Träna modellen på all träningsdata (inklusive valideringsdata)
    X2_train_full = np.vstack((X2_train_scaled, X2_val_scaled))
    y2_train_full = np.concatenate((y2_train, y2_val))
    
    ensemble_model.fit(X2_train_full, y2_train_full)
    
    # Utvärdera modellen på testdata
    y2_pred = ensemble_model.predict(X2_test_scaled)
    accuracy = accuracy_score(y2_test, y2_pred)
    
    print(f"Ensemble-modellens noggrannhet på DF2-testdata: {accuracy:.4f}")
    print("\nKlassificeringsrapport för DF2:")
    print(classification_report(y2_test, y2_pred))
    
    # Skapa confusion matrix
    cm = confusion_matrix(y2_test, y2_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix för Ensemble-modellen (DF2)')
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutsagt värde')
    plt.show()
    
    # Jämför med individuella modeller
    print("\nUtvärderar individuella modeller på DF2-datasetet...")
    
    results = {'Modell': [], 'Noggrannhet': []}
    
    for name, model in individual_models.items():
        model.fit(X2_train_full, y2_train_full)
        y2_pred = model.predict(X2_test_scaled)
        acc = accuracy_score(y2_test, y2_pred)
        results['Modell'].append(name)
        results['Noggrannhet'].append(acc)
        print(f"{name}: {acc:.4f}")
    
    # Lägg till ensemble-resultatet
    results['Modell'].append('Ensemble (Voting)')
    results['Noggrannhet'].append(accuracy)
    
    # Skapa jämförelsediagram
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modell', y='Noggrannhet', data=pd.DataFrame(results))
    plt.title('Jämförelse av modellernas noggrannhet på DF2-datasetet')
    plt.ylim(0.5, 1.0)  # Sätt y-axelns gränser för bättre visualisering
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Spara resultaten till CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('ensemble_results_df2.csv', index=False)
    
    return ensemble_model, individual_models, results_df

def train_best_individual_model(X2_train_scaled, X2_val_scaled, X2_test_scaled, y2_train, y2_val, y2_test, results_df):
    """
    Väljer den bästa individuella modellen och tränar den på all DF2-data förutom testdata
    """
    # Hitta den bästa individuella modellen (exklusive ensemble)
    individual_results = results_df[results_df['Modell'] != 'Ensemble (Voting)']
    best_model_name = individual_results.loc[individual_results['Noggrannhet'].idxmax()]['Modell']
    
    print(f"\nBästa individuella modell för DF2: {best_model_name}")
    
    # Ladda bästa parametrar
    best_params = load_best_parameters()
    
    # Skapa modellen med bästa parametrar
    if best_model_name == 'XGBoost':
        best_model = XGBClassifier(random_state=42, **best_params['XGBoost'])
    elif best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(random_state=42, solver='liblinear', **best_params['Logistic Regression'])
    elif best_model_name == 'KNN':
        best_model = KNeighborsClassifier(**best_params['KNN'])
    elif best_model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(random_state=42, **best_params['Decision Tree'])
    
    # Träna modellen på all träningsdata (inklusive valideringsdata)
    X2_train_full = np.vstack((X2_train_scaled, X2_val_scaled))
    y2_train_full = np.concatenate((y2_train, y2_val))
    
    print(f"Tränar {best_model_name} på all DF2-data förutom testdata...")
    best_model.fit(X2_train_full, y2_train_full)
    
    # Utvärdera modellen på testdata
    y2_pred = best_model.predict(X2_test_scaled)
    accuracy = accuracy_score(y2_test, y2_pred)
    
    print(f"{best_model_name} noggrannhet på DF2-testdata: {accuracy:.4f}")
    print("\nKlassificeringsrapport för DF2:")
    print(classification_report(y2_test, y2_pred))
    
    # Skapa confusion matrix
    cm = confusion_matrix(y2_test, y2_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix för {best_model_name} på DF2-datasetet')
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutsagt värde')
    plt.show()
    
    return best_model
