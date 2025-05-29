import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo (pipeline completo)
@st.cache_resource
def cargar_modelo():
    return joblib.load("pipeline_final_desercion.pkl")

modelo = cargar_modelo()

# Información del modelo en el sidebar
st.sidebar.subheader("🔍 Información del Modelo")
st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))

# Mostrar algunas features importantes
scaler_features = modelo.named_steps['preprocessor'].transformers_[0][2]
st.sidebar.write("Features escaladas:", scaler_features)

st.title("🎓 Predicción de Deserción Estudiantil")
st.markdown("Completa los datos del estudiante para predecir si existe riesgo de deserción.")

# Formulario de entrada
with st.form("formulario"):
    st.subheader("🧑‍🎓 Información general")
    col1, col2 = st.columns(2)

    with col1:
        application_order = st.slider("Application order (0=1ra opción)", 0, 9, 1)
        attendance = st.radio("Horario", ["Diurno", "Vespertino"])
        prev_grade = st.number_input("Nota previa", 0.0, 200.0, 120.0)
        admission_grade = st.number_input("Nota de admisión", 0.0, 200.0, 130.0)
        age = st.slider("Edad al ingresar", 17, 60, 22)
        gender = st.radio("Género", ["Mujer", "Hombre"])

    with col2:
        displaced = st.radio("¿Desplazado?", ["No", "Sí"])
        debtor = st.radio("¿Moroso?", ["No", "Sí"])
        tuition_paid = st.radio("¿Pago al día?", ["No", "Sí"])
        scholarship = st.radio("¿Becado?", ["No", "Sí"])
        unemployment = st.slider("Tasa de desempleo (%)", 0.0, 25.0, 7.5)
        inflation = st.slider("Inflación (%)", 0.0, 15.0, 3.0)
        gdp = st.slider("PIB (GDP)", 0.0, 20.0, 5.0)

    st.subheader("📚 Rendimiento académico")
    col3, col4 = st.columns(2)

    with col3:
        eval1 = st.number_input("Evaluaciones 1er semestre", 0, 20, 5)
        noeval1 = st.number_input("Sin evaluación 1er semestre", 0, 10, 0)
        eval2 = st.number_input("Evaluaciones 2do semestre", 0, 20, 5)
        noeval2 = st.number_input("Sin evaluación 2do semestre", 0, 10, 0)

    with col4:
        credited2 = st.number_input("Créditos 2do semestre", 0, 20, 6)
        enrolled2 = st.number_input("Inscritas 2do semestre", 0, 20, 6)
        approved2 = st.number_input("Aprobadas 2do semestre", 0, 20, 4)
        grade2 = st.number_input("Nota 2do semestre", 0.0, 20.0, 13.0)

    st.subheader("📌 Selección de categoría")
    
    # Opciones para variables categóricas
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

    marital = st.selectbox("Estado civil", marital_options)
    app_mode = st.selectbox("Modalidad de ingreso", app_mode_options)
    course = st.selectbox("Curso", course_options)
    prev_qual = st.selectbox("Tipo de formación previa", prev_qual_options)
    nacionality = st.selectbox("Nacionalidad", nacionality_options)
    mq = st.selectbox("Nivel educativo de la madre", mq_options)
    fq = st.selectbox("Nivel educativo del padre", fq_options)
    mo = st.selectbox("Ocupación de la madre", mo_options)
    fo = st.selectbox("Ocupación del padre", fo_options)

    submit = st.form_submit_button("Predecir")

# Procesamiento y predicción
if submit:
    # 1. Inicializar todas las features esperadas en 0
    datos = {col: 0 for col in modelo.feature_names_in_}
    
    # 2. Asignar valores numéricos directos
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
    
    # 3. Asignar variables categóricas (one-hot)
    # Mapeo de opciones del formulario a nombres de columnas esperados
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
    
    for base_name, (options, selected) in category_mappings.items():
        for option in options:
            col_name = f"{base_name}_{option}"
            if col_name in modelo.feature_names_in_:
                datos[col_name] = 1 if option == selected else 0
    
    # 4. Crear DataFrame con el orden exacto que espera el modelo
    X = pd.DataFrame([datos])[modelo.feature_names_in_]
    
    # Verificación detallada
    with st.expander("🔍 Ver detalles de los datos enviados"):
        st.write("Número de columnas:", len(X.columns))
        
        missing_cols = set(modelo.feature_names_in_) - set(X.columns)
        extra_cols = set(X.columns) - set(modelo.feature_names_in_)
        
        if missing_cols:
            st.warning(f"Columnas faltantes: {missing_cols}")
        if extra_cols:
            st.warning(f"Columnas extra: {extra_cols}")
        
        st.write("Valores de ejemplo:")
        st.write(X.iloc[0, :20])  # Mostrar primeras 20 columnas
        st.write("...")
        st.write(X.iloc[0, 20:40])
        st.write("...")
        st.write(X.iloc[0, 40:60])
        st.write("...")
        st.write(X.iloc[0, 60:])
        
        st.write("Tipos de datos:", X.dtypes)
    
    # Realizar predicción
    try:
        with st.spinner("Realizando predicción..."):
            pred = modelo.predict(X)[0]
            proba = modelo.predict_proba(X)[0][1]
        
        # Mostrar resultado
        st.subheader("📈 Resultado de la predicción:")
        if pred == 1:
            st.error(f"🚨 El estudiante tiene riesgo de **deserción**.\n\nProbabilidad: {proba:.2%}")
        else:
            st.success(f"✅ El estudiante **no tiene riesgo de deserción**.\n\nProbabilidad: {proba:.2%}")
            
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {str(e)}")
        with st.expander("⚠️ Detalles técnicos del error"):
            st.write(f"Tipo de error: {type(e).__name__}")
            st.write("Shape de X:", X.shape)
            st.write("Columnas en X:", X.columns.tolist())
            st.write("Valores no nulos:", X.notnull().sum())
            
            # Comparar con las features esperadas
            st.write("\nComparación con features esperadas:")
            st.write("Total features esperadas:", len(modelo.feature_names_in_))
            st.write("Features en datos:", len(X.columns))
            
            # Mostrar diferencias
            st.write("\nPrimeras 10 features esperadas:", modelo.feature_names_in_[:10])
            st.write("Primeras 10 features en datos:", X.columns.tolist()[:10])
