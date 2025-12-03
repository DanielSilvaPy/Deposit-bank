# Bibliotecas padrão
import sys
import os
import joblib

# Bibliotecas de terceiros
import sys
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Imports internos do projeto
# Adiciona a pasta raiz ao sys.path para imports locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.load_data.load_data import loadDatasetRaw
from src.utils.data_prep import preprocess_and_split
from src.utils.evaluation import evaluate_model

from src.load_data.load_data import loadDatasetRaw

def train_DecisionTreeClassifier(save_model: bool = False):
    """
    Treina um modelo de DecisionTreeClassifier padrão (baseline) 
    e avalia com métricas completas.
    """

    dataset = loadDatasetRaw()
    if dataset is None or dataset.empty:
        print('Erro: Dataset não foi carregado ou está vazio.')
        return

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(dataset)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    
    # Avaliação do modelo
    evaluate_model(
        pipeline, 
        X_test,
        y_test,
        title='Modelo Decision Tree Classifier',
        dataset_name='Bank Dataset',
        model_name='DTC Baseline',
        tuning=False
    )

    if save_model:
        joblib.dump(pipeline, 'decision_tree_classifier.pkl')
        print("Modelo Salvo como 'decision_tree_classifier.pkl'")


def train_DecisionTreeClassifier_tuning(save_model: bool = False):
    """
    Treina um modelo Decision Tree com otimização de hiperparâmetros (GridSearchCV).
    Avalia métricas completas e, se desejado, salva o melhor modelo.
    """
    dataset = loadDatasetRaw()
    if dataset is None or dataset.empty:
        print("Erro: Dataset não foi carregado ou está vazio.")
        return

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(dataset)

    # pipeline: preprocessamento + modelo
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    # grade de hiperparâmetros para tuning
    param_grid = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__splitter": ["best", "random"],
        "classifier__max_depth": [None, 5, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nMelhor score CV: {grid_search.best_score_ * 100:.2f}%")
    print("Melhores parâmetros:", grid_search.best_params_)

    # melhor modelo encontrado
    best_model = grid_search.best_estimator_

    # avaliação no conjunto de teste
    evaluate_model(
        best_model, 
        X_test, 
        y_test, 
        title="Modelo Decision Tree Classifier - Tuning", 
        dataset_name="Bank Dataset", 
        model_name="DecisionTree Tuning", 
        tuning=True)

    # salvar modelo, se desejado
    if save_model:
        joblib.dump(best_model, "decision_tree_tuned.pkl")
        print("\nModelo salvo como 'decision_tree_tuned.pkl'")


