# Imports de bibliotecas padrão do Python
import sys
import os

# Configuração do path para permitir imports internos
# Adiciona a pasta raiz ao sys.path para imports locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports internos do projeto
from src.utils.save_results import save_metrics

def evaluate_model(
    model,
    X_test,
    y_test,
    title: str = 'Modelo',
    dataset_name: str = 'Desconhecido',
    model_name: str = 'Desconhecido',
    tuning: bool = False,
    save_path: str = "model_metrics.csv"  # Pode ser sobrescrito se necessário
):
    """
    Avalia o modelo com métricas completas, imprime resultados e salva em CSV.
    """
    # Importações específicas de métricas (scikit-learn)
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
        roc_auc_score, precision_score, recall_score, f1_score
    )

    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Impressão de resultados
    print(f"\n{title} - Dataset: {dataset_name}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

    # Preparar métricas para salvar
    metrics = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Tuning": tuning,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
    }

    # Salvar métricas no CSV (caminho controlado pelo save_metrics)
    save_metrics(metrics)
