import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelo y columnas
modelo = joblib.load("pipeline_final_desercion.pkl")
columnas_esperadas = joblib.load("columnas_esperadas.pkl")

st.set_page_config(page_title="Predicción de Deserción", layout="wide")
st.title("📚 Predicción de Deserción Universitaria")
st.markdown("Por favor, llena el formulario con los datos del estudiante:")

# === Entradas agrupadas ===
int_cols = [
    'Application order', 'Daytime/evening attendance', 'Displaced', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (without evaluations)'
]

float_cols = [
    'Previous qualification (grade)', 'Admission grade', 'Curricular units 2nd sem (grade)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

bool_cols = [col for col in columnas_esperadas if col not in int_cols + float_cols]

# Crear diccionario de entrada
input_data = {}

# === Entradas tipo int ===
st.subheader("📌 Variables numéricas enteras")
cols_int = st.columns(3)
for i, col in enumerate(int_cols):
    with cols_int[i % 3]:
        input_data[col] = st.number_input(col, value=0, step=1)

# === Entradas tipo float ===
st.subheader("📊 Variables numéricas decimales")
cols_float = st.columns(3)
for i, col in enumerate(float_cols):
    with cols_float[i % 3]:
        input_data[col] = st.number_input(col, format="%.2f")

# === Entradas tipo bool ===
st.subheader("✅ Variables categóricas (Sí/No)")
cols_bool = st.columns(3)
for i, col in enumerate(bool_cols):
    with cols_bool[i % 3]:
        input_data[col] = st.checkbox(col, value=False)

# === Predicción ===
if st.button("🔍 Predecir Deserción"):
    try:
        input_df = pd.DataFrame([input_data])[columnas_esperadas]
        pred = modelo.predict(input_df)[0]
        proba = modelo.predict_proba(input_df)[0][1]

        st.success(f"🎯 Resultado: {'DESERCIÓN' if pred == 1 else 'CONTINUIDAD'}")
        st.metric("Probabilidad de deserción", f"{proba:.2%}")
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

