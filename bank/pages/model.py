import streamlit as st
import pandas as pd
import joblib

# Caminho do modelo
model_path = r"C:\Users\Daniel\Desktop\Project DNC\logistic_model_tuned.pkl"
model = joblib.load(model_path)

st.title("Simulação do Modelo Logistic Regression Tuned")

# Dividindo os inputs em colunas
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Idade", min_value=0, max_value=120, value=30)
    marital = st.selectbox("Estado Civil", ['married', 'single', 'divorced'])
    housing = st.selectbox("Empréstimo Habitacional?", ['yes', 'no'])

with col2:
    job = st.selectbox("Profissão", ['admin.', 'blue-collar', 'technician', 'services', 'management'])
    education = st.selectbox("Educação", ['secondary', 'tertiary', 'primary'])
    loan = st.selectbox("Empréstimo Pessoal?", ['yes', 'no'])

with col3:
    default = st.selectbox("Crédito default?", ['yes', 'no'])
    balance = st.number_input("Saldo Bancário", value=1000)

# Criar DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan]
})

# Colunas faltantes com valor padrão
missing_cols = ['contact', 'pdays', 'poutcome', 'previous', 'campaign', 'day', 'month', 'duration']
for col in missing_cols:
    if col not in input_df.columns:
        input_df[col] = 0 if col in ['pdays', 'previous', 'campaign', 'day', 'month', 'duration'] else 'unknown'

st.subheader("Dados do Cliente")
st.dataframe(input_df)

# Simulação
if st.button("Simular Depósito"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]
    prob_percent = round(prediction_proba[0] * 100, 2)  # transforma em %

    if prediction[0] == 1:
        st.success(f"O cliente provavelmente irá aceitar o depósito! Probabilidade: {prob_percent}%")
    else:
        st.warning(f"O cliente provavelmente não irá aceitar o depósito. Probabilidade: {prob_percent}%")

