import sys, os
import pandas as pd

# adiciona a pasta raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.load_data.load_data import loadDatasetRaw

def data_clean(dataset: pd.DataFrame, how: str = "any", inplace: bool = False) -> pd.DataFrame:
    """
    Apaga as linhas nulas do dataset.
    
    Parâmetros:
        dataset (pd.DataFrame): DataFrame de entrada.
        how (str): 'any' -> remove se QUALQUER valor na linha for NaN.
                   'all' -> remove apenas se TODOS os valores da linha forem NaN.
        inplace (bool): Se True, altera o DataFrame original. Se False, retorna um novo.
    
    Retorna:
        pd.DataFrame: DataFrame limpo.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("O parâmetro 'dataset' deve ser um pandas DataFrame.")

    linhas_antes = dataset.shape[0]
    dataset_limpo = dataset.dropna(how=how, inplace=inplace)
    linhas_depois = dataset.shape[0] if inplace else dataset_limpo.shape[0]

    print(f"✅ {linhas_antes - linhas_depois} linhas removidas.")
    
    return dataset if inplace else dataset_limpo

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza geral do DataFrame:
    - Remove duplicados
    - Remove colunas totalmente nulas
    - Preenche valores nulos (numéricos com mediana, categóricos com moda)
    - Reseta o índice
    """
    if dataset is None or dataset.empty:
        print("⚠️ DataFrame vazio ou inválido.")
        return dataset
    
    df = dataset.copy()
    
    # 1. Remover duplicados
    df.drop_duplicates(inplace=True)
    
    # 2. Remover colunas totalmente nulas
    df.dropna(axis=1, how="all", inplace=True)
    
    # 3. Preencher valores nulos
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:  # Numéricos
            df[col].fillna(df[col].median(), inplace=True)
        else:  # Categóricos
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 4. Resetar índice
    df.reset_index(drop=True, inplace=True)
    
    return df