import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier



# Bästa parametrarna för respektive modell utifrån tidigare tester på DF2
BEST_PARAMS = {
    'XGBoost': {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100},
    'Logistic Regression': {'C': 0.1, 'penalty': 'l1'},
    'KNN': {'n_neighbors': 13, 'p': 1, 'weights': 'uniform'},
    'Decision Tree': {'max_depth': 7, 'min_samples_leaf': 3, 'min_samples_split': 3}
}

def create_models():
    """Skapar modeller med de bästa parametrarna"""
    return {
        'XGBoost': XGBClassifier(random_state=42, **BEST_PARAMS['XGBoost']),
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', **BEST_PARAMS['Logistic Regression']),
        'KNN': KNeighborsClassifier(**BEST_PARAMS['KNN']),
        'Decision Tree': DecisionTreeClassifier(random_state=42, **BEST_PARAMS['Decision Tree'])
    }

def create_voting_classifier(models):
    """Skapar en VotingClassifier från modellerna"""
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting='soft')

def combine_data(X_train_scaled, X_val_scaled, y_train, y_val):
    """Kombinerar tränings- och valideringsdata"""
    return np.vstack((X_train_scaled, X_val_scaled)), np.concatenate((y_train, y_val))

def evaluate_model(model, X_test_scaled, y_test, model_name="Model"):
    """Utvärderar en modell och returnerar noggrannheten"""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def find_best_model(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test):
    # Skapa modeller
    models = create_models()
    
    # Skapa ensemble
    voting_clf = create_voting_classifier(models)
    
    # Kombinera tränings- och valideringsdata
    X_combined, y_combined = combine_data(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Träna alla modeller
    print("Tränar och utvärderar alla modeller...")
    for name, model in models.items():
        model.fit(X_combined, y_combined)
    
    voting_clf.fit(X_combined, y_combined)
    
    # Utvärdera alla modeller
    results = {'Modell': [], 'Noggrannhet': []}
    
    # Utvärdera individuella modeller
    for name, model in models.items():
        acc = evaluate_model(model, X_test_scaled, y_test, name)
        results['Modell'].append(name)
        results['Noggrannhet'].append(acc)
    
    # Utvärdera ensemble
    ensemble_acc = evaluate_model(voting_clf, X_test_scaled, y_test, "Voting Classifier")
    results['Modell'].append("Voting Classifier")
    results['Noggrannhet'].append(ensemble_acc)
    
    # Skapa dataframe med resultat
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Noggrannhet', ascending=False)
    
    # Visa resultaten
    print("\nRanking av modeller baserat på noggrannhet:")
    print(results_df)
    
    # Hitta bästa modellen
    best_model_name = results_df.iloc[0]['Modell']
    best_accuracy = results_df.iloc[0]['Noggrannhet']
    
    print(f"\n----- RESULTAT -----")
    print(f"Bästa modell: {best_model_name}")
    print(f"Noggrannhet: {best_accuracy:.4f}")
    
    # Skapa en visualisering av resultaten
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Noggrannhet', y='Modell', data=results_df, hue='Modell', palette='viridis', legend=False)
    plt.title('Jämförelse av modellernas noggrannhet')
    plt.xlabel('Noggrannhet')
    plt.ylabel('Modell')
    plt.tight_layout()
    plt.show()
    
    # Lägg till modellerna i ett dictionary för att kunna returnera dem
    all_models = models.copy()
    all_models["Voting Classifier"] = voting_clf
    
    return best_model_name, all_models, results_df
