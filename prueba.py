import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y las columnas esperadas
modelo = joblib.load("pipeline_final_desercion.pkl")
columnas_esperadas = joblib.load("columnas_esperadas.pkl")

st.set_page_config(page_title="Predicción de Deserción", layout="wide")
st.title("🎓 Predicción de Deserción Universitaria")

st.markdown("Sube un archivo `.csv` con las siguientes columnas en el mismo orden y formato del modelo.")

archivo = st.file_uploader("📁 Carga tu archivo CSV", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        if list(df.columns) != columnas_esperadas:
            st.error("❌ Las columnas del archivo no coinciden con las esperadas.")
            st.text("Orden esperado de columnas:")
            st.code("\n".join(columnas_esperadas))
        else:
            predicciones = modelo.predict(df)
            probabilidades = modelo.predict_proba(df)[:, 1]
            resultado = df.copy()
            resultado["ProbabilidadDesercion"] = probabilidades
            resultado["Prediccion"] = predicciones
            st.success("✅ Predicción realizada correctamente.")
            st.write(resultado)
            st.download_button(
                label="📥 Descargar resultados como CSV",
                data=resultado.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_desercion.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"⚠️ Ocurrió un error al procesar el archivo: {e}")
