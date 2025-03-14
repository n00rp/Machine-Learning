

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Logistic regression

param_grid_linearregression = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# XGBoost (eXtreme Gradient Boosting)

param_grid_xgboost = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [50, 100, 200]
}

# KNN - KNeighborsClassifier

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],  # Antal grannar
    'weights': ['uniform', 'distance'],  # Vikter för grannar
    'p': [1, 2]  # Distansmått (1 = Manhattan, 2 = Euclidean)
}

param_grid_dt = {
    'max_depth': [3, 5, 7, 10, 15, 20],  # Maximalt djup för trädet
    'min_samples_split': [2, 5, 10, 20],  # Minsta antal sampel för att dela en nod
    'min_samples_leaf': [1, 2, 4, 8]  # Minsta antal sampel i ett löv
    
}



# Lägg till fler parametergrids för andra modeller...

# Definiera träningsfunktionen
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, model_name, save_results=False, filename="evaluation_scores.csv"):

    """
    Funktion för att träna och utvärdera en modell med GridSearchCV.
    
    :param model: Modell som ska tränas (t.ex. RandomForestClassifier, LogisticRegression)
    :param param_grid: Hyperparametrar för GridSearchCV
    :param X_train, y_train: Träningsdata
    :param X_val, y_val: Valideringsdata
    :param X_test, y_test: Testdata
    :param model_name: Namn på modellen (t.ex. "Random Forest", "Logistic Regression")
    :param save_results: Om True, spara resultaten i en CSV-fil
    :param filename: Namn på filen där resultaten ska sparas
    :return: Bästa modellen, accuracy på valideringsdata, accuracy på testdata
    """
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
def train_models(X_train_std, X_train_norm, X2_train_std, X2_train_norm, y_train, y2_train, X_val_std, X_val_norm, 
                X2_val_std, X2_val_norm, y_val, y2_val, X_test_std, X_test_norm, X2_test_std, X2_test_norm, y_test, y2_test):
    # XGBoost
    xgb_model = XGBClassifier(random_state=42)
    best_xgb_model, xgb_val_accuracy, xgb_test_accuracy = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X_train_std,
        y_train=y_train,
        X_val=X_val_std,
        y_val=y_val,
        X_test=X_test_std,
        y_test=y_test,
        model_name="XGBoost med standardiserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )
    
    # XGBoost på normaliserad data

    best_xgb_model_norm, xgb_val_accuracy_norm, xgb_test_accuracy_norm = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        model_name="XGBoost med normaliserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )
    # Logistic Regression - Standardiserad data
    

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X_train_std,
        y_train=y_train,
        X_val=X_val_std,
        y_val=y_val,
        X_test=X_test_std,
        y_test=y_test,
        model_name="Logistic Regression med standardiserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Logistic Regression på normaliserad data

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        model_name="Logistic Regression med normaliserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # KNN - KNeighborsClassifier - Standardiserad data


    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X_train_std,
        y_train=y_train,
        X_val=X_val_std,
        y_val=y_val,
        X_test=X_test_std,
        y_test=y_test,
        model_name="KNN med standardiserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # KNN - KNeighborsClassifier på normaliserad data

    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        model_name="KNN med normaliserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Decision Tree - Standardiserad data

    from sklearn.tree import DecisionTreeClassifier

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X_train_std,
        y_train=y_train,
        X_val=X_val_std,
        y_val=y_val,
        X_test=X_test_std,
        y_test=y_test,
        model_name="Decision Tree med standardiserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Decision Tree på normaliserad data

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        model_name="Decision Tree med normaliserad data",
        save_results=True,
        filename="evaluation_scores.csv"
    )


# Samma modeller fast för DF2

    
    # XGBoost - Standardiserad data för DF2
    

    xgb_model = XGBClassifier(random_state=42)
    best_xgb_model, xgb_val_accuracy, xgb_test_accuracy = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X2_train_std,
        y_train=y2_train,
        X_val=X2_val_std,
        y_val=y2_val,
        X_test=X2_test_std,
        y_test=y2_test,
        model_name="XGBoost med standardiserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # XGBoost på normaliserad data för DF2

    best_xgb_model_norm, xgb_val_accuracy_norm, xgb_test_accuracy_norm = train_and_evaluate_model(
        model=xgb_model,
        param_grid=param_grid_xgboost,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="XGBoost med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Logistic Regression - Standardiserad data för DF2

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X2_train_std,
        y_train=y2_train,
        X_val=X2_val_std,
        y_val=y2_val,
        X_test=X2_test_std,
        y_test=y2_test,
        model_name="Logistic Regression med standardiserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

      # Logistic Regression på normaliserad data för DF2

    best_log_reg_model_norm, log_reg_val_accuracy_norm, log_reg_test_accuracy_norm = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="Logistic Regression med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # Logistic Regression - Standardiserad data för DF2

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X2_train_std,
        y_train=y2_train,
        X_val=X2_val_std,
        y_val=y2_val,
        X_test=X2_test_std,
        y_test=y2_test,
        model_name="Logistic Regression med standardiserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Logistic Regression på normaliserad data för DF2

    log_reg_model = LogisticRegression(random_state=42, solver='liblinear')
    best_log_reg_model, log_reg_val_accuracy, log_reg_test_accuracy = train_and_evaluate_model(
        model=log_reg_model,
        param_grid=param_grid_linearregression,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="Logistic Regression med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


# KNN - KNeighborsClassifier - Standardiserad data för DF2

    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_std,
        y_train=y2_train,
        X_val=X2_val_std,
        y_val=y2_val,
        X_test=X2_test_std,
        y_test=y2_test,
        model_name="KNN med standardiserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

# KNN - KNeighborsClassifier på normaliserad data för DF2

    knn_model = KNeighborsClassifier()
    best_knn_model, knn_val_accuracy, knn_test_accuracy = train_and_evaluate_model(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="KNN med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )


    # Decision Tree - Standardiserad data för DF2

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X2_train_std,
        y_train=y2_train,
        X_val=X2_val_std,
        y_val=y2_val,
        X_test=X2_test_std,
        y_test=y2_test,
        model_name="Decision Tree med standardiserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

    # Decision Tree på normaliserad data för DF2

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="Decision Tree med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )

# Decision Tree på normaliserad data för DF2

    dt_model = DecisionTreeClassifier(random_state=42)
    best_dt_model, dt_val_accuracy, dt_test_accuracy = train_and_evaluate_model(
        model=dt_model,
        param_grid=param_grid_dt,
        X_train=X2_train_norm,
        y_train=y2_train,
        X_val=X2_val_norm,
        y_val=y2_val,
        X_test=X2_test_norm,
        y_test=y2_test,
        model_name="Decision Tree med normaliserad data från DF2",
        save_results=True,
        filename="evaluation_scores.csv"
    )