# Imports de bibliotecas padrão do Python
import sys
import os
import joblib

# Imports de bibliotecas externas
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Imports internos do projeto
# Adiciona a pasta raiz ao sys.path para imports locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.load_data.load_data import loadDatasetRaw
from src.utils.data_prep import preprocess_and_split
from src.utils.evaluation import evaluate_model

def train_LogisticRegression(save_model: bool = False):
    """
    Treina um modelo de Logistic Regression padrão (baseline) 
    e avalia com métricas completas.
    """
    dataset = loadDatasetRaw()
    if dataset is None or dataset.empty:
        print("Erro: Dataset não foi carregado ou está vazio.")
        return

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(dataset)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Avaliação do modelo
    evaluate_model(
        pipeline,
        X_test,
        y_test,
        title='Modelo LogisticRegression',
        dataset_name='Bank Dataset',
        model_name='LR Baseline',
        tuning=False
    )

    if save_model:
        joblib.dump(pipeline, "logistic_model_baseline.pkl")
        print("Modelo salvo como 'logistic_model_baseline.pkl'")


def train_LogisticRegression_tuning(save_model: bool = True):
    """
    Treina um modelo de Logistic Regression otimizado com GridSearchCV.
    Explora hiperparâmetros, faz cross-validation e avalia métricas completas.
    """
    dataset = loadDatasetRaw()
    if dataset is None or dataset.empty:
        print("Erro ao ler o dataset")
        return

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(dataset)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Grid de hiperparâmetros separado por penalty
    param_grid = [
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__class_weight": [None, "balanced"]
        },
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1"],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"]
        }
    ]

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nMelhor score CV: {grid_search.best_score_ * 100:.2f}%")
    print("Melhores parâmetros:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Avaliação do modelo ajustado
    evaluate_model(
        best_model,
        X_test,
        y_test,
        title='Modelo LogisticRegression - Tuning',
        dataset_name='Bank Dataset',
        model_name='Bank Dataset',
        tuning=True
    )

    if save_model:
        joblib.dump(best_model, "logistic_model_tuned.pkl")
        print("\nModelo salvo como 'logistic_model_tuned.pkl'")

train_LogisticRegression_tuning()