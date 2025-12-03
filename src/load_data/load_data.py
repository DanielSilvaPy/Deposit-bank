# Imports de bibliotecas padrão do Python
import pandas as pd

def loadDatasetRaw(file_path=r"C:\Users\Daniel\Desktop\Project DNC\data\raw\bank.csv"):
    """
    Carrega um dataset CSV em um DataFrame do pandas.

    Args:
        file_path (str): Caminho do arquivo CSV.

    Returns:
        pd.DataFrame | None: DataFrame com os dados carregados, ou None se não encontrar o arquivo.
    """
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None