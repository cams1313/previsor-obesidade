import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


# Configuração da página
st.set_page_config(page_title="Previsão de Obesidade")

# Título
st.title('Previsão de Obesidade')

st.write("Por favor, insira os dados:")

# ==========================
# ENTRADAS DO USUÁRIO
# ==========================

genero = st.selectbox('Gênero', ['Masculino', 'Feminino'])
genero = 1 if genero == 'Masculino' else 0

frequencia_alimentos_caloricos = st.selectbox('Você consome frequentemente alimentos com alto teor calórico?', ['Sim', 'Não'])
frequencia_alimentos_caloricos = 1 if frequencia_alimentos_caloricos == 'Sim' else 0

fumante = st.selectbox('Você fuma?', ['Sim', 'Não'])
fumante = 1 if fumante == 'Sim' else 0

monitora_calorias = st.selectbox('Você monitora seu consumo de calorias?', ['Sim', 'Não'])
monitora_calorias = 1 if monitora_calorias == 'Sim' else 0

historico_familiar = st.selectbox('Algum membro da família sofreu ou sofre de excesso de peso?', ['Sim', 'Não'])
historico_familiar = 1 if historico_familiar == 'Sim' else 0

frequencia_alcool = st.selectbox('Com que frequência você consome bebidas alcoólicas?',
                                 ['Sempre', 'Frequentemente', 'Às vezes', 'Não'])
alcool_map = {'Sempre': 1, 'Frequentemente': 2, 'Às vezes': 3, 'Não': 4}
frequencia_alcool = alcool_map[frequencia_alcool]

lanches_entre_refeicoes = st.selectbox('Você come alguma coisa entre as refeições?',
                                       ['Sempre', 'Frequentemente', 'Às vezes', 'Não'])
lanches_map = {'Sempre': 1, 'Frequentemente': 2, 'Às vezes': 3, 'Não': 4}
lanches_entre_refeicoes = lanches_map[lanches_entre_refeicoes]

transporte = st.selectbox('Qual meio de transporte você costuma usar?',
                          ['Carro', 'Bicicleta', 'Moto', 'Transporte público', 'A pé'])
transporte_map = {'Carro': 0, 'Bicicleta': 1, 'Moto': 2, 'Transporte público': 3, 'A pé': 4}
transporte = transporte_map[transporte]

idade = st.number_input('Idade', min_value=0)

altura = st.number_input('Altura (cm)', min_value=0.0)
peso = st.number_input('Peso (kg)', min_value=0.0)

# → Consumo de vegetais
vegatais_opcoes = ['Às vezes', 'Sempre', 'Nunca']
vegatais_entrada = st.selectbox('Frequência de Consumo de Vegetais', vegatais_opcoes)
veg_map = {'Às vezes': 2, 'Sempre': 3, 'Nunca': 1}
consumo_vegetais = veg_map[vegatais_entrada]

# → Número de refeições
nrp_opcoes = ['Mais de três', 'Entre 1 e 2', '3']
nrp_entrada = st.selectbox('Número de Refeições Principais', nrp_opcoes)
nrp_map = {'Mais de três': 3, 'Entre 1 e 2': 1, '3': 2}
qtd_refeicoes_diarias = nrp_map[nrp_entrada]

# → Consumo de água
agua_opcoes = ['Entre 1L e 2L', 'Mais que 2L', 'Menos de um litro']
agua_entrada = st.selectbox('Consumo diário de água', agua_opcoes)
agua_map = {'Menos de um litro': 1, 'Entre 1L e 2L': 2, 'Mais que 2L': 3}
consumo_agua = agua_map[agua_entrada]

# → Tempo de tecnologia
tec_opcoes = ['Eu não uso', '0–2 horas', '3–5 horas']
tec_entrada = st.selectbox('Tempo de uso de tecnologia (horas/dia)', tec_opcoes)
tec_map = {'Eu não uso': 1, '0–2 horas': 2, '3–5 horas': 3}
tempo_tecnologia = tec_map[tec_entrada]

frequencia_atividade_fisica = st.selectbox(
    'Frequência de atividade física (vezes por semana)',
    ['Eu não pratico', '1 ou 2 dias', '2 ou 4 dias', '4 ou 5 dias']
)
atv_map = {'Eu não pratico': 1, '1 ou 2 dias': 2, '2 ou 4 dias': 3, '4 ou 5 dias': 4}
frequencia_atividade_fisica = atv_map[frequencia_atividade_fisica]


# ==========================
# CARREGAR MODELO E SCALER
# ==========================

modelo = joblib.load("model/modelo_obesidade.pkl")
scaler = joblib.load("model/scaler.pkl")
le_y = joblib.load("model/label_y.pkl")


# Mapa de rótulos do Y
label_map = {
    0: "Abaixo do peso",
    1: "Peso normal",
    2: "Sobrepeso",
    3: "Obesidade"
}


# ==========================
# PREVISÃO
# ==========================

if st.button('Prever'):

    # Ordem EXATA usada no treinamento
    dados = np.array([[  
        genero, idade, historico_familiar, frequencia_alimentos_caloricos,
        consumo_vegetais, qtd_refeicoes_diarias, lanches_entre_refeicoes, fumante,
        consumo_agua, monitora_calorias, frequencia_atividade_fisica,
        tempo_tecnologia, frequencia_alcool, transporte
    ]])

    # Padronização
    dados_scaled = scaler.transform(dados)

    pred = modelo.predict(dados_scaled)[0]

    resultado = label_map.get(pred, "Resultado desconhecido")

    st.success(f'O nível de obesidade previsto é: **{resultado}**')