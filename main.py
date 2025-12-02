import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import xgboost

# Configuração da página
st.set_page_config(page_title="Previsão de Obesidade")

# Título
st.title('Previsão de Obesidade')

st.write("Por favor, insira os dados:")

# -------------------------------
# Carregar modelo e encoder
# -------------------------------
modelo_path = 'model/modelo_obesidade.pkl'
encoder_path = "model/label_encoder.pkl"

modelo = joblib.load(modelo_path)
le_y = joblib.load(encoder_path)


# Tradução das classes
TRADUCAO_LABELS = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III"
}

# Ordem das features usadas no modelo
FEATURES = [
    'genero', 'idade', 'altura', 'peso',
    'frequencia_alimentos_caloricos', 'consumo_vegetais',
    'qtd_refeicoes_diarias', 'fumante', 'consumo_agua',
    'monitora_calorias', 'frequencia_atividade_fisica',
    'tempo_tecnologia', 'frequencia_alcool',
    'historico_familiar_no', 'historico_familiar_yes',
    'lanches_entre_refeicoes_Always', 'lanches_entre_refeicoes_Frequently',
    'lanches_entre_refeicoes_Sometimes', 'lanches_entre_refeicoes_no',
    'transporte_Automobile', 'transporte_Bike', 'transporte_Motorbike',
    'transporte_Public_Transportation', 'transporte_Walking'
]

# -------------------------------
# Função de previsão
# -------------------------------
def prever_obesidade(dados):
    df = pd.DataFrame([dados])[FEATURES]
    pred_num = model.predict(df)[0]
    pred_en = le.inverse_transform([pred_num])[0]
    pred_pt = TRADUCAO_LABELS[pred_en]
    return pred_pt


# -------------------------------
# Interface Streamlit
# -------------------------------
st.title("🔍 Classificador de Nível de Obesidade")
st.write("Preencha os dados abaixo para obter a previsão:")

# Inputs
genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
genero = 1 if genero == "Masculino" else 0

idade = st.number_input("Idade", min_value=10, max_value=100, value=25)
altura = st.number_input("Altura (m)", min_value=1.20, max_value=2.20, value=1.70)
peso = st.number_input("Peso (kg)", min_value=30, max_value=250, value=70)

frequencia_alimentos_caloricos = st.slider("Frequência de alimentos calóricos (0–3)", 0, 3, 1)
consumo_vegetais = st.slider("Consumo de vegetais (1–3)", 1, 3, 2)
qtd_refeicoes_diarias = st.slider("Quantidade de refeições por dia", 1, 4, 3)

fumante = st.selectbox("Fumante", ["Não", "Sim"])
fumante = 1 if fumante == "Sim" else 0

consumo_agua = st.slider("Consumo de água (1–3)", 1, 3, 2)
monitora_calorias = st.selectbox("Monitora calorias?", ["Não", "Sim"])
monitora_calorias = 1 if monitora_calorias == "Sim" else 0

frequencia_atividade_fisica = st.slider("Frequência de atividade física (0–3)", 0, 3, 1)
tempo_tecnologia = st.slider("Tempo diário em tecnologia (0–2)", 0, 2, 1)
frequencia_alcool = st.slider("Frequência de consumo de álcool (0–3)", 0, 3, 1)

historico_familiar = st.selectbox("Histórico familiar de obesidade?", ["Não", "Sim"])
historico_familiar_no = 1 if historico_familiar == "Não" else 0
historico_familiar_yes = 1 if historico_familiar == "Sim" else 0

lanches = st.selectbox("Lanches entre refeições", 
                       ["Never", "Sometimes", "Frequently", "Always"])
lanches_entre_refeicoes_Always = 1 if lanches == "Always" else 0
lanches_entre_refeicoes_Frequently = 1 if lanches == "Frequently" else 0
lanches_entre_refeicoes_Sometimes = 1 if lanches == "Sometimes" else 0
lanches_entre_refeicoes_no = 1 if lanches == "Never" else 0

transporte = st.selectbox("Transporte", 
                          ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
transporte_Automobile = 1 if transporte == "Automobile" else 0
transporte_Bike = 1 if transporte == "Bike" else 0
transporte_Motorbike = 1 if transporte == "Motorbike" else 0
transporte_Public_Transportation = 1 if transporte == "Public_Transportation" else 0
transporte_Walking = 1 if transporte == "Walking" else 0

# -------------------------------
# Montar dicionário final
# -------------------------------
dados_paciente = {
    'genero': genero,
    'idade': idade,
    'altura': altura,
    'peso': peso,
    'frequencia_alimentos_caloricos': frequencia_alimentos_caloricos,
    'consumo_vegetais': consumo_vegetais,
    'qtd_refeicoes_diarias': qtd_refeicoes_diarias,
    'fumante': fumante,
    'consumo_agua': consumo_agua,
    'monitora_calorias': monitora_calorias,
    'frequencia_atividade_fisica': frequencia_atividade_fisica,
    'tempo_tecnologia': tempo_tecnologia,
    'frequencia_alcool': frequencia_alcool,
    'historico_familiar_no': historico_familiar_no,
    'historico_familiar_yes': historico_familiar_yes,
    'lanches_entre_refeicoes_Always': lanches_entre_refeicoes_Always,
    'lanches_entre_refeicoes_Frequently': lanches_entre_refeicoes_Frequently,
    'lanches_entre_refeicoes_Sometimes': lanches_entre_refeicoes_Sometimes,
    'lanches_entre_refeicoes_no': lanches_entre_refeicoes_no,
    'transporte_Automobile': transporte_Automobile,
    'transporte_Bike': transporte_Bike,
    'transporte_Motorbike': transporte_Motorbike,
    'transporte_Public_Transportation': transporte_Public_Transportation,
    'transporte_Walking': transporte_Walking
}

# -------------------------------
# Botão para prever
# -------------------------------
if st.button("🔮 Prever Nível de Obesidade"):
    pred = prever_obesidade(dados_paciente)
    st.success(f"**Previsão: {pred}**")