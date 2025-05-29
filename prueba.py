import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo (pipeline completo)
@st.cache_resource
def cargar_modelo():
    return joblib.load("pipeline_final_desercion.pkl")

modelo = cargar_modelo()

# Verificación del modelo cargado
st.sidebar.subheader("🔍 Información del Modelo")
try:
    st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
    if 'preprocessor' in modelo.named_steps:
        st.sidebar.write("Transformers:", modelo.named_steps['preprocessor'].transformers_)
    st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))
except Exception as e:
    st.sidebar.error(f"Error al inspeccionar modelo: {str(e)}")

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
    marital = st.selectbox("Estado civil", ["Divorced", "FactoUnion", "Separated", "Single"])
    app_mode = st.selectbox("Modalidad de ingreso", [
        "Admisión Especial", "Admisión Regular", "Admisión por Ordenanza",
        "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"
    ])
    course = st.selectbox("Curso", [
        "Agricultural & Environmental Sciences", "Arts & Design", "Business & Management",
        "Communication & Media", "Education", "Engineering & Technology",
        "Health Sciences", "Social Sciences"
    ])
    prev_qual = st.selectbox("Tipo de formación previa", [
        "Higher Education", "Other", "Secondary Education", "Technical Education"
    ])
    nacionality = st.selectbox("Nacionalidad", [
        "Colombian", "Cuban", "Dutch", "English", "German", "Italian", "Lithuanian",
        "Moldovan", "Mozambican", "Portuguese", "Romanian", "Santomean", "Turkish"
    ])
    mq = st.selectbox("Nivel educativo de la madre", [
        "Basic_or_Secondary", "Other_or_Unknown", "Postgraduate", "Technical_Education"
    ])
    fq = st.selectbox("Nivel educativo del padre", [
        "Basic_or_Secondary", "Other_or_Unknown", "Postgraduate"
    ])
    mo = st.selectbox("Ocupación de la madre", [
        "Administrative/Clerical", "Skilled Manual Workers", "Special Cases",
        "Technicians/Associate Professionals", "Unskilled Workers"
    ])
    fo = st.selectbox("Ocupación del padre", [
        "Administrative/Clerical", "Professionals", "Skilled Manual Workers",
        "Special Cases", "Technicians/Associate Professionals"
    ])

    submit = st.form_submit_button("Predecir")

# Procesamiento y predicción
if submit:
    # Crear diccionario con todos los valores esperados en 0 primero
    datos = {col: 0 for col in modelo.feature_names_in_}
    
    # Actualizar con los valores del formulario
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
    
    # Agregar dummies (solo una activa por grupo)
    dummy_cols = [
        ("Marital status", marital),
        ("Application mode", app_mode),
        ("Course", course),
        ("Previous qualification", prev_qual),
        ("Nacionality", nacionality),
        ("Mother's qualification", mq),
        ("Father's qualification", fq),
        ("Mother's occupation", mo),
        ("Father's occupation", fo)
    ]
    
    for grupo, valor in dummy_cols:
        colname = f"{grupo}_{valor}"
        # Verificar si la columna existe en las features esperadas
        if colname in modelo.feature_names_in_:
            datos[colname] = 1
        else:
            st.warning(f"Columna no encontrada en el modelo: {colname}")
    
    # Crear DataFrame asegurando el orden correcto
    X = pd.DataFrame([datos])[modelo.feature_names_in_]
    
    # Verificación detallada
    with st.expander("🔍 Ver detalles de los datos enviados al modelo"):
        st.subheader("Información de Entrada")
        st.write("Número de columnas:", len(X.columns))
        st.write("Columnas faltantes en el modelo:", set(modelo.feature_names_in_) - set(X.columns))
        st.write("Columnas extra en los datos:", set(X.columns) - set(modelo.feature_names_in_))
        st.write("Tipos de datos:", X.dtypes)
        st.write("Primeras filas:", X.head())
    
    try:
        # Predicción con manejo de errores
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
            st.write("Valores en X:", X.values)
            st.write("Tipos de datos:", X.dtypes)
            
            # Información adicional para debugging
            st.write("\nInformación del modelo:")
            try:
                st.write("Features esperadas:", modelo.feature_names_in_)
                st.write("Número de features esperadas:", len(modelo.feature_names_in_))
            except Exception as model_error:
                st.write("Error al obtener info del modelo:", str(model_error))
