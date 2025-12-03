import os
import pandas as pd
import streamlit as st
import plotly.express as px

# Caminho absoluto para a raiz do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Caminho para o CSV
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "bank.csv")

# Layout mais largo
st.set_page_config(layout="wide")
st.title('Visualização dos Clientes')

# Carrega o dataset
df_bank = pd.read_csv(CSV_PATH)
st.dataframe(df_bank)  # tabela interativa

# Preparar dados
age_counts = df_bank['age'].value_counts().sort_index()
job_counts = df_bank['job'].value_counts().sort_values(ascending=True)

# Gráfico de idade dos clientes
fig_age = px.bar(
    x=age_counts.index,
    y=age_counts.values,
    labels={'x': 'Idade', 'y': 'Quantidade de Clientes'},
    text=age_counts.values,
    title='Distribuição de Idade dos Clientes',
    color=age_counts.values,
    color_continuous_scale='Viridis'
)
fig_age.update_layout(yaxis_title='Número de Clientes', xaxis_title='Idade')
fig_age.update_traces(textposition='outside')

# Gráfico de empregos dos clientes
fig_job = px.bar(
    x=job_counts.values,
    y=job_counts.index,
    orientation='h',
    labels={'x': 'Quantidade de Clientes', 'y': 'Profissão'},
    text=job_counts.values,
    title='Distribuição de Profissões dos Clientes',
    color=job_counts.values,
    color_continuous_scale='Cividis'
)
fig_job.update_traces(textposition='outside')

# Colunas no Streamlit
col1, col2 = st.columns(2)
col1.plotly_chart(fig_age, use_container_width=True)
col2.plotly_chart(fig_job, use_container_width=True)

# Criar faixas de idade
bins = [0, 29, 39, 49, 59, 100]
labels = ['<30', '30-39', '40-49', '50-59', '60+']
df_bank['faixa_idade'] = pd.cut(df_bank['age'], bins=bins, labels=labels, right=False)

# Gráfico interativo de depósitos por faixa etária
fig_deposit = px.histogram(
    df_bank,
    x='faixa_idade',
    color='deposit',
    barmode='group',  # barras lado a lado
    text_auto=True,   # mostra valores nas barras
    labels={'faixa_idade':'Faixa de Idade', 'count':'Quantidade de Clientes', 'deposit':'Depósito'},
    title='Depósitos aceitos e não aceitos por faixa etária',
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig_deposit.update_layout(
    xaxis_title='Faixa de Idade',
    yaxis_title='Quantidade de Clientes',
    legend_title='Depósito'
)
st.plotly_chart(fig_deposit, use_container_width=True)


# Gráfico interativo dos depósitos realizados por faixa educacional
fig_education_deposit = px.histogram(
    df_bank,
    x='education',
    color='deposit',
    barmode='group',
    text_auto=True,
    labels={'education': 'Nível Educacional', 'count': 'Quantidade de Clientes', 'deposit': 'Depósito'},
    title='Depósitos aceitos e não aceitos por níveis educacionais',
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig_education_deposit.update_layout(
    xaxis_title='Nível Educacional dos Clientes',
    yaxis_title='Quantidade de Clientes',
    legend_title='Depósito'
)
st.plotly_chart(fig_education_deposit, use_container_width=True)

# BoxPlot do saldo
fig_box = px.box(
    df_bank,
    y='balance',
    title='Distribuição do Saldo',
    points='outliers',
    color_discrete_sequence=['lightgreen']
)
fig_box.update_layout(
    xaxis_title='Distribuição do Saldo',
    yaxis_title='Saldo',
    legend_title='Variação'
)
st.plotly_chart(fig_box, use_container_width=True)


# Contagem de depósitos
deposit_counts = df_bank['deposit'].value_counts()

fig_pie = px.pie(
    names=deposit_counts.index,  # categorias (yes/no)
    values=deposit_counts.values, # quantidade de cada categoria
    title='Proporção de Depósitos Aceitos e Não Aceitos',
    color=deposit_counts.index,
    color_discrete_map={'yes':'green', 'no':'red'}
)

fig_pie.update_traces(textposition='inside', textinfo='percent+label')  # mostra percentual + label

# Colunas no Streamlit
col4, col5 = st.columns(2)
col4.plotly_chart(fig_pie, use_container_width=True)