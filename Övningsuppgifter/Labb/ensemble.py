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
    """Skapar en confusion matrix och klassificeringsrapport för modellen"""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n----- Utvärdering av {model_name} -----")
    print(f"Noggrannhet: {accuracy:.4f}")
    
    # Skriver ut klassificeringsrapport
    print("\nKlassificeringsrapport:")
    print(classification_report(y_test, y_pred))
    
    # Skriver ut resultatet i en confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix för {model_name}')
    plt.ylabel('Faktiskt värde')
    plt.xlabel('Förutsagt värde')
    plt.show()

def train_best_model(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, best_model_name="XGBoost"):

    # Skapa modeller med bästa parametrarna
    models = create_models()
    
    # Skapa en ensemble om det valda modell är "Voting Classifier"
    if best_model_name == "Voting Classifier":
        best_model = create_voting_classifier(models)
    elif best_model_name in models:
        best_model = models[best_model_name]
    else:
        raise ValueError(f"Unknown model: {best_model_name}. Choose from: {list(models.keys())} or 'Voting Classifier'")
    
    # Kombinerar tränings- och valideringsdata
    print(f"Kombinerar tränings- och valideringsdata...")
    X_combined, y_combined = combine_data(X_train_scaled, X_val_scaled, y_train, y_val)
    print(f"Storlek på kombinerad träningsdata: {X_combined.shape}, {y_combined.shape}") 
    
    # Tränar modellen på all data förutom testdata
    print(f"\nTränar {best_model_name} på all data förutom testdata...")
    

    if best_model_name == "Voting Classifier":
        for name, model in models.items():
            print(f"  Tränar {name}...")
            model.fit(X_combined, y_combined)
        best_model = create_voting_classifier(models)
    
    # Tränar modellen på all data förutom testdata
    best_model.fit(X_combined, y_combined)
    
    # Skriver ut slutlig utvärdering av modellen
    print(f"\nSlutlig utvärdering av {best_model_name}:")
    evaluate_model(best_model, X_test_scaled, y_test, best_model_name)
    
    return best_model