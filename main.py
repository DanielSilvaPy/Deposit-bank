# Imports de bibliotecas padrão
import sys
import os
import pandas as pd

# Configuração de path para permitir imports internos
# Adiciona a pasta raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Imports internos do projeto
from src.load_data.load_data import loadDatasetRaw
from src.utils.data_prep import preprocess_and_split
from src.utils.evaluation import evaluate_model
from src.model.train_LogisticRegression import train_LogisticRegression, train_LogisticRegression_tuning
from src.model.train_DecisionTreeClassifier import train_DecisionTreeClassifier, train_DecisionTreeClassifier_tuning


df_bank = loadDatasetRaw()
if df_bank is None or df_bank.empty:
    print("Erro: dataset não carregado ou vazio.")
else:

    # 01 LogisticRegression
    train_LogisticRegression(save_model=True)
    train_LogisticRegression_tuning(save_model=True)

    # 02 DecisionTreeClassifier
    train_DecisionTreeClassifier(save_model=True)
    train_DecisionTreeClassifier_tuning(save_model=True)

    # 03