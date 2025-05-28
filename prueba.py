import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y las columnas esperadas
modelo = joblib.load("pipeline_final_desercion.pkl")
columnas_esperadas = joblib.load("columnas_esperadas.pkl")

st.set_page_config(page_title="Predicción de Deserción", layout="wide")
st.title("🎓 Predicción de Deserción Universitaria")

st.markdown("Sube un archivo `.csv` con las columnas en el mismo orden y formato del modelo.")

archivo = st.file_uploader("📁 Carga tu archivo CSV", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip()  # Elimina espacios ocultos

        columnas_subidas = list(df.columns)
        
        # Comparar columna por columna
        columnas_diferentes = [(i, col1, col2) for i, (col1, col2) in enumerate(zip(columnas_subidas, columnas_esperadas)) if col1 != col2]

        if columnas_diferentes:
            st.error("❌ Las columnas del archivo no coinciden con las esperadas.")
            st.write("❗ Columnas distintas en estas posiciones:")
            for i, col1, col2 in columnas_diferentes:
                st.write(f"{i+1}. Subida: `{col1}` — Esperada: `{col2}`")
            st.text("Orden esperado:")
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


