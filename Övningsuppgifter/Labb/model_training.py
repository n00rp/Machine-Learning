

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import os


"""
Delen med hyperparametrar har jag tagit inspiration från sklearn dokumentationen och StackOverflow
för att hitta rimliga nivåer.

"""

# Logistic regression

param_grid_linearregression = {
    'C': [0.1, 0.5, 1, 10, 100], # Ett mindre värde på C innebär starkare regularisering
    'penalty': ['l1', 'l2'] # Bestämmer om koefficienterna kan bli satta till 0 eller inte
}

# XGBoost (eXtreme Gradient Boosting)

param_grid_xgboost = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2], # Inlärningstakten, hur snabbt modellen anpassar sig under träning
    'max_depth': [3, 5, 7, 9], # Hur djupt varje träd är och hur komplext det får vara
    'n_estimators': [50, 100, 200] # Antalet träd som det får vara
}

# KNN - KNeighborsClassifier

param_grid_knn = {
    'n_neighbors': [5, 7, 9, 11, 13],  # Antal grannar
    'weights': ['uniform', 'distance'],  # Vikter för grannar
    'p': [1, 2]  # Distansmått (1 = Manhattan, 2 = Euclidean)
}

param_grid_dt = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],  # Maximalt djup för trädet
    'min_samples_split': [3, 5, 6, 7,10, 20],  # Minsta antal sampel för att dela en nod
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8]  # Minsta antal sampel i ett löv
    
}





# Definiera träningsfunktionen
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, model_name, save_results=False, filename="evaluation_scores.csv"):

    # Använd GridSearchCV för att hitta bästa hyperparametrar
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    
    # Visa bästa parametrar
    print(f"Bästa parametrar för {model_name}:", grid_search.best_params_)
    
    # Gör förutsägelser på valideringsdata
    y_val_pred = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Gör förutsägelser på testdata
    y_test_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Spara resultaten om save_results är True
    if save_results:
        results = {
            'Model': [model_name],
            'Best Parameters': [grid_search.best_params_],
            'Validation Accuracy': [val_accuracy],
            'Test Accuracy': [test_accuracy]
        }
        results_df = pd.DataFrame(results)
        
        # Kontrollera om filen redan finns
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            # Kontrollera om modellen redan finns i filen
            if model_name not in existing_df['Model'].values:
                # Lägg till nya resultat om modellen inte redan finns
                results_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                print(f"Modellen '{model_name}' finns redan i filen. Inga nya resultat sparades.")
        else:
            # Skapa en ny fil om den inte finns
            results_df.to_csv(filename, mode='w', header=True, index=False)
            print(f"Ny fil skapad: {filename}")
    
    return grid_search.best_estimator_, val_accuracy, test_accuracy


# Träna modeller


def train_models(X_train_scaled, X2_train_scaled, y_train, y2_train, X_val_scaled, 
                y_val, y2_val, X2_test_scaled, y_test, y2_test, X_test_scaled, X2_val_scaled):
    # XGBoost
    
    # XGBoost med data från DF1
    xgb_model = XGBClassifier(random_state=42)
    best_xgb_model_norm, xgb_val_accuracy_norm, xgb_test_accuracy_norm = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name="XGBoost med data från DF1",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Logistic Regression med data från DF1

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name="Logistic Regression med data från DF1",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # KNN - KNeighborsClassifier med data från DF1

    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name="KNN med data Från DF1",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # Decision Tree med data från DF1

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        model_name="Decision Tree med data DF1",
        save_results=True,
        filename="evaluation_scores.csv"
    )


# Samma modeller fast för DF2

    # XGBoost med data för DF2

    best_xgb_model_norm, xgb_val_accuracy_norm, xgb_test_accuracy_norm = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X2_train_scaled,
        y_train=y2_train,
        X_val=X2_val_scaled,
        y_val=y2_val,
        X_test=X2_test_scaled,
        y_test=y2_test,
        model_name="XGBoost med data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # Logistic Regression med data för DF2

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X2_train_scaled,
        y_train=y2_train,
        X_val=X2_val_scaled,
        y_val=y2_val,
        X_test=X2_test_scaled,
        y_test=y2_test,
        model_name="Logistic Regression med data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


# KNN - KNeighborsClassifier med data för DF2

    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_scaled,
        y_train=y2_train,
        X_val=X2_val_scaled,
        y_val=y2_val,
        X_test=X2_test_scaled,
        y_test=y2_test,
        model_name="KNN med data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # Decision Tree med data för DF2

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X2_train_scaled,
        y_train=y2_train,
        X_val=X2_val_scaled,
        y_val=y2_val,
        X_test=X2_test_scaled,
        y_test=y2_test,
        model_name="Decision Tree med data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

