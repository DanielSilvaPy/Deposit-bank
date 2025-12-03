# Imports de bibliotecas padrão do Python
import pandas as pd


# Imports de bibliotecas externas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_and_split(dataset: pd.DataFrame, target_col: str = 'deposit', test_size: float = 0.2, random_state: int = 42):
    """
    Função auxiliar para preparar os dados: remove colunas problemáticas, separa features/target,
    identifica colunas numéricas e categóricas, e divide em treino/teste.

    Retorna: X_train, X_test, y_train, y_test, preprocessor
    """
    dataset = dataset.drop(columns=['duration', 'day', 'month'], errors='ignore')

    # Features e target
    X = dataset.drop(columns=target_col)
    y = dataset[target_col].map({'yes': 1, 'no': 0})

    # Identificar colunas numéricas e categóricas
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Pré-processamento
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor