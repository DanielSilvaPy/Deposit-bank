import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_model_false_tuning():
    """
    Plota gráfico comparando modelos que não passaram por Tuning
    """
    # Lendo os dados
    df_model = pd.read_csv('../data/data_model/model_metrics.csv')

    # Filtrando modelos sem Tuning
    df_model_false_tuning = df_model[df_model['Tuning'] == False]

    # Se não houver desvio, podemos colocar yerr=0
    yerr = [0]*len(df_model_false_tuning)

    # Gráfico
    plt.figure(figsize=(16,6))
    plt.errorbar(df_model_false_tuning['Model'], df_model_false_tuning['Accuracy'], yerr=yerr, fmt='o', capsize=5, linestyle='None')

    plt.ylabel('Acurácia')
    plt.xlabel('Modelo')
    plt.title('Comparação de Modelos - Sem Tuning')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

plot_model_false_tuning()

def plot_model_metrics():
    """
    Plota métricas comparativas para os modelos sem tuning.
    """
    # Lendo os dados
    df_model = pd.read_csv(r'C:\Users\Daniel\Desktop\Project DNC\data\data_model\model_metrics.csv')

    # Selecionando os modelos sem tuning
    df_model_false_tuning = df_model[df_model['Tuning'] == False]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = df_model_false_tuning['Model'].tolist()
    
    # Preparando posições para barras
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(16, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df_model_false_tuning[metric], width=width, label=metric)

    plt.xticks(x + width*1.5, models, rotation=45)
    plt.ylabel('Score')
    plt.xlabel('Modelos')
    plt.title('Comparação de Métricas dos Modelos (sem Tuning)')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Chama a função
plot_model_metrics()
