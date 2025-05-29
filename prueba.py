import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide"
)

# Limpiar cache si hay cambios
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🗑️ Limpiar Cache", help="Presiona si hay problemas con el modelo"):
        st.cache_resource.clear()
        st.rerun()

# Cargar el modelo (pipeline completo)
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("pipeline_final_desercion.pkl")
        return modelo, None
    except Exception as e:
        return None, str(e)

modelo, error_carga = cargar_modelo()

if modelo is None:
    st.error(f"❌ Error al cargar el modelo: {error_carga}")
    st.stop()

# Información del modelo en el sidebar
st.sidebar.header("🔍 Información del Modelo")

try:
    # Información básica del pipeline
    st.sidebar.subheader("Pipeline")
    pasos = list(modelo.named_steps.keys())
    for i, paso in enumerate(pasos, 1):
        st.sidebar.write(f"{i}. {paso}")
    
    # Información del preprocessor
    st.sidebar.subheader("Preprocessor")
    preprocessor = modelo.named_steps['preprocessor']
    
    # Obtener información de las features de manera segura
    try:
        total_features = len(modelo.feature_names_in_)
        st.sidebar.metric("Total de Features", total_features)
        
        # Información de los transformers
        if hasattr(preprocessor, 'transformers_'):
            st.sidebar.write("**Transformers:**")
            for i, (name, transformer, features) in enumerate(preprocessor.transformers_):
                if name == 'remainder':
                    st.sidebar.write(f"• {name}: {len(features)} features (passthrough)")
                else:
                    transformer_name = type(transformer).__name__
                    st.sidebar.write(f"• {name}: {transformer_name} ({len(features)} features)")
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error al obtener info del modelo: {str(e)}")
    
    # Información de versión
    import sklearn
    st.sidebar.subheader("Versiones")
    st.sidebar.info(f"Scikit-learn: {sklearn.__version__}")
    
except Exception as e:
    st.sidebar.error(f"Error en sidebar: {str(e)}")

# Título principal
st.title("🎓 Predicción de Deserción Estudiantil")
st.markdown("Completa los datos del estudiante para predecir si existe riesgo de deserción académica.")

# Crear las listas de opciones
marital_options = ["Divorced", "FactoUnion", "Separated", "Single"]
app_mode_options = ["Admisión Especial", "Admisión Regular", "Admisión por Ordenanza",
                    "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"]
course_options = ["Agricultural & Environmental Sciences", "Arts & Design", "Business & Management",
                  "Communication & Media", "Education", "Engineering & Technology",
                  "Health Sciences", "Social Sciences"]
prev_qual_options = ["Higher Education", "Other", "Secondary Education", "Technical Education"]
nacionality_options = ["Colombian", "Cuban", "Dutch", "English", "German", "Italian", "Lithuanian",
                       "Moldovan", "Mozambican", "Portuguese", "Romanian", "Santomean", "Turkish"]
mq_options = ["Basic_or_Secondary", "Other_or_Unknown", "Postgraduate", "Technical_Education"]
fq_options = ["Basic_or_Secondary", "Other_or_Unknown", "Postgraduate"]
mo_options = ["Administrative/Clerical", "Skilled Manual Workers", "Special Cases",
              "Technicians/Associate Professionals", "Unskilled Workers"]
fo_options = ["Administrative/Clerical", "Professionals", "Skilled Manual Workers",
              "Special Cases", "Technicians/Associate Professionals"]

# Formulario principal
with st.form("formulario_prediccion"):
    # Información general
    st.subheader("👤 Información General")
    col1, col2, col3 = st.columns(3)

    with col1:
        application_order = st.slider("Orden de solicitud (0=1ra opción)", 0, 9, 1, 
                                    help="0 significa que fue la primera opción del estudiante")
        age = st.slider("Edad al ingresar", 17, 60, 22)
        gender = st.radio("Género", ["Mujer", "Hombre"])
        marital = st.selectbox("Estado civil", marital_options)

    with col2:
        attendance = st.radio("Horario de clases", ["Diurno", "Vespertino"])
        displaced = st.radio("¿Es desplazado?", ["No", "Sí"])
        debtor = st.radio("¿Tiene deudas?", ["No", "Sí"])
        tuition_paid = st.radio("¿Matrícula al día?", ["Sí", "No"])

    with col3:
        scholarship = st.radio("¿Tiene beca?", ["No", "Sí"])
        app_mode = st.selectbox("Modalidad de ingreso", app_mode_options)
        nacionality = st.selectbox("Nacionalidad", nacionality_options)
        prev_qual = st.selectbox("Formación previa", prev_qual_options)

    # Calificaciones
    st.subheader("📊 Calificaciones y Notas")
    col4, col5 = st.columns(2)

    with col4:
        prev_grade = st.number_input("Nota de formación previa", 0.0, 200.0, 120.0,
                                   help="Nota obtenida en la formación previa")
        admission_grade = st.number_input("Nota de admisión", 0.0, 200.0, 130.0,
                                        help="Nota obtenida en el examen de admisión")

    with col5:
        course = st.selectbox("Área de estudio", course_options)

    # Información familiar
    st.subheader("👨‍👩‍👧‍👦 Información Familiar")
    col6, col7 = st.columns(2)

    with col6:
        st.write("**Información de la Madre**")
        mq = st.selectbox("Nivel educativo de la madre", mq_options)
        mo = st.selectbox("Ocupación de la madre", mo_options)

    with col7:
        st.write("**Información del Padre**")
        fq = st.selectbox("Nivel educativo del padre", fq_options)
        fo = st.selectbox("Ocupación del padre", fo_options)

    # Rendimiento académico
    st.subheader("📚 Rendimiento Académico")
    col8, col9 = st.columns(2)

    with col8:
        st.write("**Primer Semestre**")
        eval1 = st.number_input("Evaluaciones 1er semestre", 0, 20, 5)
        noeval1 = st.number_input("Sin evaluación 1er semestre", 0, 10, 0)

    with col9:
        st.write("**Segundo Semestre**")
        credited2 = st.number_input("Créditos obtenidos 2do semestre", 0, 20, 6)
        enrolled2 = st.number_input("Materias inscritas 2do semestre", 0, 20, 6)
        eval2 = st.number_input("Evaluaciones 2do semestre", 0, 20, 5)
        approved2 = st.number_input("Materias aprobadas 2do semestre", 0, 20, 4)
        grade2 = st.number_input("Nota promedio 2do semestre", 0.0, 20.0, 13.0)
        noeval2 = st.number_input("Sin evaluación 2do semestre", 0, 10, 0)

    # Indicadores económicos
    st.subheader("📈 Indicadores Socioeconómicos")
    col10, col11, col12 = st.columns(3)

    with col10:
        unemployment = st.slider("Tasa de desempleo (%)", 0.0, 25.0, 7.5)
    with col11:
        inflation = st.slider("Tasa de inflación (%)", 0.0, 15.0, 3.0)
    with col12:
        gdp = st.slider("PIB - GDP (%)", 0.0, 20.0, 5.0)

    # Botón de predicción
    st.markdown("---")
    submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary")

# Procesamiento cuando se envía el formulario
if submitted:
    with st.spinner("🔄 Procesando datos y realizando predicción..."):
        try:
            # Crear diccionario base con todas las features del modelo
            datos = {}
            
            # Inicializar todas las features con 0
            for feature in modelo.feature_names_in_:
                datos[feature] = 0
            
            # Asignar valores numéricos directos
            datos.update({
                "Application order": application_order,
                "Daytime/evening attendance": 1 if attendance == "Diurno" else 0,
                "Previous qualification (grade)": prev_grade,
                "Admission grade": admission_grade,
                "Displaced": 1 if displaced == "Sí" else 0,
                "Debtor": 1 if debtor == "Sí" else 0,
                "Tuition fees up to date": 1 if tuition_paid == "Sí" else 0,
                "Gender": 1 if gender == "Hombre" else 0,
                "Scholarship holder": 1 if scholarship == "Sí" else 0,
                "Age at enrollment": age,
                "Curricular units 1st sem (evaluations)": eval1,
                "Curricular units 1st sem (without evaluations)": noeval1,
                "Curricular units 2nd sem (credited)": credited2,
                "Curricular units 2nd sem (enrolled)": enrolled2,
                "Curricular units 2nd sem (evaluations)": eval2,
                "Curricular units 2nd sem (approved)": approved2,
                "Curricular units 2nd sem (grade)": grade2,
                "Curricular units 2nd sem (without evaluations)": noeval2,
                "Unemployment rate": unemployment,
                "Inflation rate": inflation,
                "GDP": gdp
            })

            # Mapeo de variables categóricas
            category_mappings = {
                "Marital status": (marital_options, marital),
                "Application mode": (app_mode_options, app_mode),
                "Course": (course_options, course),
                "Previous qualification": (prev_qual_options, prev_qual),
                "Nacionality": (nacionality_options, nacionality),
                "Mother's qualification": (mq_options, mq),
                "Father's qualification": (fq_options, fq),
                "Mother's occupation": (mo_options, mo),
                "Father's occupation": (fo_options, fo)
            }

            # Aplicar encoding one-hot para variables categóricas
            for base_name, (options, selected) in category_mappings.items():
                for option in options:
                    col_name = f"{base_name}_{option}"
                    if col_name in modelo.feature_names_in_:
                        datos[col_name] = 1 if option == selected else 0

            # Crear DataFrame con el orden correcto de features
            X = pd.DataFrame([datos], columns=modelo.feature_names_in_)
            
            # Asegurar tipos de datos correctos
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except:
                    X[col] = 0
            
            # Verificar que no hay valores nulos
            if X.isnull().any().any():
                st.error("❌ Error: Se detectaron valores nulos en los datos")
                st.stop()

            # Realizar predicción con manejo de warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediccion = modelo.predict(X)[0]
                probabilidades = modelo.predict_proba(X)[0]
                prob_desercion = probabilidades[1]

            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultados de la Predicción")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediccion == 1:
                    st.error("🚨 **ALTO RIESGO DE DESERCIÓN**")
                    st.markdown(f"**Probabilidad de deserción: {prob_desercion:.1%}**")
                    
                    # Consejos para estudiantes en riesgo
                    st.subheader("💡 Recomendaciones")
                    st.info("""
                    **Acciones sugeridas:**
                    - Contactar al tutor académico inmediatamente
                    - Solicitar apoyo del centro de bienestar estudiantil
                    - Evaluar opciones de financiamiento adicional
                    - Considerar tutoría académica personalizada
                    - Revisar la carga académica del siguiente semestre
                    """)
                else:
                    st.success("✅ **BAJO RIESGO DE DESERCIÓN**")
                    st.markdown(f"**Probabilidad de deserción: {prob_desercion:.1%}**")
                    
                    # Consejos para mantener el buen rendimiento
                    st.subheader("🎯 Recomendaciones")
                    st.info("""
                    **Para mantener el buen rendimiento:**
                    - Continuar con los hábitos de estudio actuales
                    - Participar en actividades extracurriculares
                    - Mantener comunicación regular con profesores
                    - Considerar roles de liderazgo estudiantil
                    """)
            
            with col_res2:
                # Gráfico de probabilidades
                st.subheader("📈 Distribución de Probabilidades")
                
                prob_no_desercion = probabilidades[0]
                
                # Crear datos para el gráfico
                data_prob = {
                    'Resultado': ['No Deserción', 'Deserción'],
                    'Probabilidad': [prob_no_desercion, prob_desercion],
                    'Color': ['#00ff00' if prob_desercion < 0.5 else '#ff6b6b', 
                             '#ff6b6b' if prob_desercion >= 0.5 else '#00ff00']
                }
                
                # Mostrar métricas
                st.metric("Probabilidad de No Deserción", f"{prob_no_desercion:.1%}")
                st.metric("Probabilidad de Deserción", f"{prob_desercion:.1%}")
                
                # Interpretación del riesgo
                if prob_desercion < 0.3:
                    riesgo = "🟢 Bajo"
                elif prob_desercion < 0.7:
                    riesgo = "🟡 Medio"
                else:
                    riesgo = "🔴 Alto"
                
                st.metric("Nivel de Riesgo", riesgo)

            # Información adicional expandible
            with st.expander("🔍 Detalles Técnicos de la Predicción"):
                st.write("**Información del modelo:**")
                st.write(f"- Algoritmo: {type(modelo.named_steps['classifier']).__name__}")
                st.write(f"- Total de características: {len(modelo.feature_names_in_)}")
                st.write(f"- Características numéricas activas: {sum(X.iloc[0] != 0)}")
                
                st.write("**Top 5 características con valor:**")
                features_activas = X.iloc[0][X.iloc[0] != 0].sort_values(ascending=False).head()
                for feature, valor in features_activas.items():
                    st.write(f"- {feature}: {valor}")

        except Exception as e:
            st.error(f"❌ Error durante la predicción: {str(e)}")
            
            with st.expander("🔧 Información de Debug"):
                st.write(f"**Tipo de error:** {type(e).__name__}")
                st.write(f"**Mensaje:** {str(e)}")
                
                try:
                    st.write(f"**Shape de datos:** {X.shape if 'X' in locals() else 'No definido'}")
                    st.write(f"**Features esperadas:** {len(modelo.feature_names_in_)}")
                except:
                    st.write("Error al obtener información de debug")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🎓 Sistema de Predicción de Deserción Estudiantil<br>
    Desarrollado para apoyar la retención estudiantil y el éxito académico
    </small>
</div>
""", unsafe_allow_html=True)
