import streamlit as st
import pandas as pd
import joblib
import os

# ================================
# CARREGAMENTO SEGURO DO MODELO
# ================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = os.path.join(BASE_DIR, "logistic_model_tuned.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Modelo não encontrado em: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ================================
# INTERFACE
# ================================
st.title("Simulação do Modelo - Logistic Regression Tuned")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Idade", min_value=0, max_value=120, value=30)
    marital = st.selectbox("Estado Civil", ['married', 'single', 'divorced'])
    housing = st.selectbox("Empréstimo Habitacional?", ['yes', 'no'])

with col2:
    job = st.selectbox(
        "Profissão",
        ['admin.', 'blue-collar', 'technician', 'services', 'management']
    )
    education = st.selectbox("Educação", ['secondary', 'tertiary', 'primary'])
    loan = st.selectbox("Empréstimo Pessoal?", ['yes', 'no'])

with col3:
    default = st.selectbox("Crédito default?", ['yes', 'no'])
    balance = st.number_input("Saldo Bancário", value=1000)

# ================================
# CRIAÇÃO DO DATAFRAME
# ================================
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

# ================================
# COLUNAS QUE O MODELO ESPERA
# ================================
missing_cols = [
    'contact', 'pdays', 'poutcome',
    'previous', 'campaign', 'day',
    'month', 'duration'
]

for col in missing_cols:
    if col not in input_df.columns:
        if col in ['pdays', 'previous', 'campaign', 'day', 'month', 'duration']:
            input_df[col] = 0
        else:
            input_df[col] = 'unknown'

st.subheader("Dados do Cliente")
st.dataframe(input_df, use_container_width=True)

# ================================
# SIMULAÇÃO
# ================================
if st.button("Simular Depósito"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]
        prob_percent = round(float(prediction_proba[0]) * 100, 2)

        if prediction[0] == 1:
            st.success(
                f"✅ O cliente provavelmente irá aceitar o depósito!\n\n"
                f"📊 Probabilidade: **{prob_percent}%**"
            )
        else:
            st.warning(
                f"⚠️ O cliente provavelmente NÃO irá aceitar o depósito.\n\n"
                f"📊 Probabilidade: **{prob_percent}%**"
            )

    except Exception as e:
        st.error(f"Erro ao fazer a predição: {e}")
