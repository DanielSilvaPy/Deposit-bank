import pandas as pd
import os

# Caminho base (ajustado para o seu projeto)
BASE_DIR = r"C:\Users\Daniel\Desktop\Project DNC\data\data_model"
SAVE_PATH = os.path.join(BASE_DIR, "model_metrics.csv")

def save_metrics(metrics: dict, save_path: str = SAVE_PATH):
    """
    Salva as métricas do modelo em um CSV.
    Se o arquivo existir, adiciona a nova linha. Caso contrário, cria um novo.
    """
    # Garante que a pasta existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Cria DataFrame da nova linha
    df_new = pd.DataFrame([metrics])
    
    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(save_path, index=False)
    else:
        df_new.to_csv(save_path, index=False)

