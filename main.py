import streamlit as st
import pickle
import numpy as np
import xgboost
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(page_title="Previsão de Obesidade teste")

# Título
st.title('Previsão de Obesidade')

st.write("Por favor, insira os dados:")

# Entradas do usuário
genero = st.selectbox('Gênero', ['Masculino', 'Feminino'])
frequencia_alimentos_caloricos = st.selectbox('Você consome frequentemente alimentos com alto teor calórico?', ['Sim', 'Não'])
fumante = st.selectbox('Você fuma?', ['Sim', 'Não'])
monitora_calorias = st.selectbox('Você monitora seu consumo de calorias?', ['Sim', 'Não'])
historico_familiar = st.selectbox('Algum membro da família sofreu ou sofre de excesso de peso?', ['Sim', 'Não'])
frequencia_alcool = st.selectbox('Com que frequência você consome bebidas alcoólicas?', ['Sempre', 'Frequentemente', 'Às vezes', 'Não'])
lanches_entre_refeicoes = st.selectbox('Você come alguma coisa entre as refeições?', ['Sempre', 'Frequentemente', 'Às vezes', 'Não'])
transporte = st.selectbox('Qual meio de transporte você costuma usar?', ['Carro', 'Bicicleta', 'Moto', 'Transporte público', 'A pé'])
idade = st.number_input('Idade', min_value=0)
altura = st.number_input('Altura (cm)', min_value=0.0)
peso = st.number_input('Peso (kg)', min_value=0.0)

# Consumo de vegetais
vegatais_opcoes = ['Às vezes', 'Sempre', 'Nunca']
vegatais_entrada = st.selectbox('Frequência de Consumo de Vegetais', vegatais_opcoes)
veg_map = {'Às vezes': 2, 'Sempre': 3, 'Nunca': 1}
consumo_vegetais = veg_map[vegatais_entrada]

# Número de refeições principais
nrp_opcoes = ['Mais de três', 'Entre 1 e 2', '3']
nrp_entrada = st.selectbox('Número de Refeições Principais', nrp_opcoes)
nrp_map = {'Mais de três': 3, 'Entre 1 e 2': 1, '3': 2}
qtd_refeicoes_diarias = nrp_map[nrp_entrada]

# Consumo de água
agua_opcoes = ['Entre 1L e 2L', 'Mais que 2L', 'Menos de um litro']
agua_entrada = st.selectbox('Consumo diário de água', agua_opcoes)
agua_map = {'Menos de um litro': 1, 'Entre 1L e 2L': 2, 'Mais que 2L': 3}
consumo_agua = agua_map[agua_entrada]

# Tempo usando tecnologia
tec_opcoes = ['Eu não uso', '0–2 horas', '3–5 horas']
tec_entrada = st.selectbox('Tempo de utilização de dispositivos tecnológicos (horas por dia)', tec_opcoes)
tec_map = {'Eu não uso': 1, '0–2 horas': 2, '3–5 horas': 3}
tempo_tecnologia = tec_map[tec_entrada]

# Frequência de atividade física
atv_opcoes = ['Eu não pratico', '1 ou 2 dias', '2 ou 4 dias', '4 ou 5 dias']
atv_entrada = st.selectbox('Frequência de atividade física (vezes por semana)', atv_opcoes)
atv_map = {'Eu não pratico': 1, '1 ou 2 dias': 2, '2 ou 4 dias': 3, '4 ou 5 dias': 4}
frequencia_atividade_fisica = atv_map[atv_entrada]

# Codificação numérica
genero = 1 if genero == 'Masculino' else 0
frequencia_alimentos_caloricos = 1 if frequencia_alimentos_caloricos == 'Sim' else 0
fumante = 1 if fumante == 'Sim' else 0
monitora_calorias = 1 if monitora_calorias == 'Sim' else 0
historico_familiar = 1 if historico_familiar == 'Sim' else 0

alcool_map = {'Sempre': 1, 'Frequentemente': 2, 'Às vezes': 3, 'Não': 4}
frequencia_alcool = alcool_map[frequencia_alcool]

lanches_entre_refeicoes_map = {'Sempre': 1, 'Frequentemente': 2, 'Às vezes': 3, 'Não': 4}
lanches_entre_refeicoes = lanches_entre_refeicoes_map[lanches_entre_refeicoes]

transporte_map = {'Carro': 0, 'Bicicleta': 1, 'Moto': 2, 'Transporte público': 3, 'A pé': 4}
transporte = transporte_map[transporte]

# Carregar modelo e scaler
modelo_path = 'model/modelo_obesidade.pkl'
scaler_path = 'model/scaler.pkl'
encoder_path = "model/label_y.pkl"

modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)
le_y = joblib.load(encoder_path)

if st.button('Prever'):

    dados = np.array([[  
        genero, idade, historico_familiar, frequencia_alimentos_caloricos,
        consumo_vegetais, qtd_refeicoes_diarias, lanches_entre_refeicoes, fumante,
        consumo_agua, monitora_calorias, frequencia_atividade_fisica, tempo_tecnologia,
        frequencia_alcool, transporte
    ]])

    dados_scaled = scaler.transform(dados)

    previsao = modelo.predict(dados_scaled)

    mapa_classes = {
        1: "Abaixo do peso",
        2: "Peso normal",
        3: "Sobrepeso",
        4: "Obesidade"
    }

    classe = mapa_classes.get(previsao[0], "Desconhecido")

    st.success(f'O nível de obesidade previsto é: **{classe}**')
