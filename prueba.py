import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
from sklearn.exceptions import NotFittedError

# Configuración inicial
st.set_page_config(page_title="Predicción de Deserción", layout="wide")

# Limpiar cache si hay problemas
if st.button("🗑️ Limpiar Cache", help="Presiona si hay problemas con el modelo"):
    st.cache_resource.clear()
    st.rerun()

# Cargar el modelo (pipeline completo)
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("pipeline_final_desercion.pkl")
        
        # Verificar que el modelo esté correctamente cargado
        if not hasattr(modelo, 'named_steps'):
            raise AttributeError("El modelo no tiene la estructura de pipeline esperada")
            
        # Verificar que tenga los feature names
        try:
            _ = modelo.feature_names_in_
        except NotFittedError:
            raise AttributeError("El modelo no tiene feature_names_in_. ¿Está correctamente entrenado?")
            
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        st.stop()

# Cargar modelo
try:
    modelo = cargar_modelo()
    columnas_esperadas = modelo.feature_names_in_.tolist()
    
    # Información del modelo en el sidebar
    st.sidebar.subheader("🔍 Información del Modelo")
    st.sidebar.write("Tipo de modelo:", type(modelo).__name__)
    st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
    st.sidebar.write("Número de features esperadas:", len(columnas_esperadas))
    
    # Verificar compatibilidad de versiones
    import sklearn
    import xgboost as xgb
    st.sidebar.info(f"Scikit-learn: {sklearn.__version__}")
    st.sidebar.info(f"XGBoost: {xgb.__version__}")
    
    # Información del preprocessor
    preprocessor = modelo.named_steps['preprocessor']
    st.sidebar.write("Transformers del preprocessor:")
    for name, transformer, columns in preprocessor.transformers_:
        col_count = len(columns) if hasattr(columns, '__len__') else 'N/A'
        st.sidebar.write(f"- {name}: {type(transformer).__name__ if not isinstance(transformer, str) else transformer} ({col_count} columnas)")

except Exception as e:
    st.error(f"Error crítico al inicializar la aplicación: {str(e)}")
    st.stop()

# Interfaz principal
st.title("🎓 Predicción de Deserción Estudiantil")
st.markdown("""
Complete los datos del estudiante para predecir si existe riesgo de deserción.
Los campos marcados con * son obligatorios.
""")

# Definición de variables y opciones
MARITAL_OPTIONS = ["Divorced", "FactoUnion", "Separated", "Single"]
APP_MODE_OPTIONS = ["Admisión Especial", "Admisión Regular", "Admisión por Ordenanza",
                   "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"]
COURSE_OPTIONS = ["Agricultural & Environmental Sciences", "Arts & Design", "Business & Management",
                 "Communication & Media", "Education", "Engineering & Technology",
                 "Health Sciences", "Social Sciences"]

# Formulario de entrada
with st.form("student_form"):
    st.subheader("🧑‍🎓 Información General del Estudiante")
    
    col1, col2 = st.columns(2)
    
    with col1:
        application_order = st.slider("Orden de aplicación*", 0, 9, 1,
                                    help="0 = Primera opción de carrera, 9 = Novena opción")
        attendance = st.radio("Horario de clases*", ["Diurno", "Vespertino"])
        prev_grade = st.number_input("Nota formación previa*", 0.0, 200.0, 120.0, step=0.1)
        admission_grade = st.number_input("Nota de admisión*", 0.0, 200.0, 130.0, step=0.1)
        age = st.slider("Edad al ingresar*", 17, 60, 20)
        gender = st.radio("Género*", ["Mujer", "Hombre"])
    
    with col2:
        displaced = st.radio("Situación de desplazamiento*", ["No", "Sí"])
        debtor = st.radio("Estado de morosidad*", ["No", "Sí"])
        tuition_paid = st.radio("Pago de matrícula*", ["Al día", "En mora"])
        scholarship = st.radio("Beca*", ["No", "Sí"])
        unemployment = st.slider("Tasa de desempleo regional*", 0.0, 25.0, 7.5, step=0.1)
        inflation = st.slider("Tasa de inflación*", 0.0, 15.0, 3.0, step=0.1)
        gdp = st.slider("Crecimiento del PIB*", -5.0, 20.0, 2.5, step=0.1)

    st.subheader("📚 Rendimiento Académico")
    
    col3, col4 = st.columns(2)
    
    with col3:
        eval1 = st.number_input("Evaluaciones 1er semestre*", 0, 20, 5)
        noeval1 = st.number_input("Cursos sin evaluar 1er semestre", 0, 10, 0)
        eval2 = st.number_input("Evaluaciones 2do semestre*", 0, 20, 5)
        noeval2 = st.number_input("Cursos sin evaluar 2do semestre", 0, 10, 0)
    
    with col4:
        credited2 = st.number_input("Créditos reconocidos 2do semestre", 0, 20, 0)
        enrolled2 = st.number_input("Cursos inscritos 2do semestre*", 0, 20, 6)
        approved2 = st.number_input("Cursos aprobados 2do semestre*", 0, 20, 4)
        grade2 = st.number_input("Promedio 2do semestre*", 0.0, 20.0, 13.0, step=0.1)

    st.subheader("📌 Información Categórica")
    
    marital = st.selectbox("Estado civil*", MARITAL_OPTIONS)
    app_mode = st.selectbox("Modalidad de ingreso*", APP_MODE_OPTIONS)
    course = st.selectbox("Programa académico*", COURSE_OPTIONS)
    
    submit = st.form_submit_button("🔮 Predecir Riesgo de Deserción")

# Procesamiento después de enviar el formulario
if submit:
    try:
        # Crear DataFrame con todas las columnas en el orden esperado
        input_data = pd.DataFrame(0, index=[0], columns=columnas_esperadas)
        
        # Mapeo de valores del formulario a las columnas del modelo
        mappings = {
            # Información general
            "Application order": application_order,
            "Daytime/evening attendance": 1 if attendance == "Diurno" else 0,
            "Previous qualification (grade)": prev_grade,
            "Admission grade": admission_grade,
            "Age at enrollment": age,
            "Gender": 1 if gender == "Hombre" else 0,
            "Displaced": 1 if displaced == "Sí" else 0,
            "Debtor": 1 if debtor == "Sí" else 0,
            "Tuition fees up to date": 1 if tuition_paid == "Al día" else 0,
            "Scholarship holder": 1 if scholarship == "Sí" else 0,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
            
            # Rendimiento académico
            "Curricular units 1st sem (evaluations)": eval1,
            "Curricular units 1st sem (without evaluations)": noeval1,
            "Curricular units 2nd sem (credited)": credited2,
            "Curricular units 2nd sem (enrolled)": enrolled2,
            "Curricular units 2nd sem (evaluations)": eval2,
            "Curricular units 2nd sem (approved)": approved2,
            "Curricular units 2nd sem (grade)": grade2,
            "Curricular units 2nd sem (without evaluations)": noeval2,
            
            # Variables categóricas (one-hot encoded)
            f"Marital status_{marital}": 1,
            f"Application mode_{app_mode}": 1,
            f"Course_{course}": 1
        }
        
        # Aplicar los mapeos al DataFrame
        for col, val in mappings.items():
            if col in input_data.columns:
                input_data[col] = val

        # Validación final de datos
        if input_data.isna().any().any():
            st.error("Error: Hay valores nulos en los datos de entrada")
            st.stop()

        # Realizar predicción
        with st.spinner("Analizando datos..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = modelo.predict_proba(input_data)[0][1]
                prediction = modelo.predict(input_data)[0]

        # Mostrar resultados
        st.subheader("📊 Resultados de la Predicción")
        
        # Visualización principal
        risk_color = "red" if proba > 0.7 else "orange" if proba > 0.5 else "green"
        risk_text = "ALTO" if proba > 0.7 else "MODERADO" if proba > 0.5 else "BAJO"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicción", 
                     value="RIESGO DE DESERCIÓN" if prediction == 1 else "NO HAY RIESGO",
                     delta=f"Probabilidad: {proba:.1%}")
        
        with col2:
            st.metric("Nivel de Riesgo", 
                     value=risk_text,
                     delta_color="off")
        
        with col3:
            st.write("**Interpretación:**")
            if proba > 0.7:
                st.error("""Se recomienda intervención inmediata. 
                        El estudiante tiene alta probabilidad de abandonar.""")
            elif proba > 0.5:
                st.warning("""Se recomienda monitoreo cercano. 
                          El estudiante muestra señales de riesgo.""")
            else:
                st.success("""El estudiante tiene baja probabilidad de deserción. 
                          Continuar con seguimiento normal.""")

        # Gráfico de probabilidad
        st.progress(float(proba))
        st.caption(f"Probabilidad estimada de deserción: {proba:.1%}")

        # Sección de análisis detallado
        with st.expander("🔍 Detalles Técnicos", expanded=False):
            st.write("**Datos enviados al modelo:**")
            st.dataframe(input_data.T.style.background_gradient(cmap="Blues"), height=300)
            
            st.write("**Variables más influyentes:**")
            try:
                # Intentar obtener importancia de características si es XGBoost
                import xgboost
                if hasattr(modelo.named_steps['classifier'], 'feature_importances_'):
                    importances = modelo.named_steps['classifier'].feature_importances_
                    features = modelo.feature_names_in_
                    importance_df = pd.DataFrame({
                        'Variable': features,
                        'Importancia': importances
                    }).sort_values('Importancia', ascending=False).head(10)
                    
                    st.bar_chart(importance_df.set_index('Variable'))
            except Exception:
                st.warning("No se pudo obtener información de importancia de características")

    except Exception as e:
        st.error(f"❌ Error al procesar la solicitud: {str(e)}")
        
        # Modo debug para desarrolladores
        with st.expander("⚙️ Modo Debug (para desarrolladores)"):
            st.write("**Información del error:**")
            st.code(str(e), language='python')
            
            st.write("**Estado del modelo:**")
            st.json({
                "Tipo": str(type(modelo)),
                "Pasos": list(modelo.named_steps.keys()),
                "Feature names": list(modelo.feature_names_in_[:5]) + ["..."] if hasattr(modelo, 'feature_names_in_') else "No disponible"
            })
            
            st.write("**Datos de entrada generados:**")
            try:
                st.write(input_data)
            except:
                st.write("No se pudo generar el dataframe de entrada")

# Información adicional en el footer
st.markdown("---")
st.markdown("""
**Notas:**
- *Sistema predictivo basado en aprendizaje automático*
- *Los resultados son estimaciones probabilísticas*
- *Versión del modelo: 1.0.0*
""")
