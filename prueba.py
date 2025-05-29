import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# Limpiar cache si hay cambios
if st.button("🗑️ Limpiar Cache", help="Presiona si hay problemas con el modelo"):
    st.cache_resource.clear()
    st.experimental_rerun()

# Cargar el modelo (sin asumir pipeline)
@st.cache_resource
def cargar_modelo():
    return joblib.load("pipeline_final_desercion.pkl")

try:
    modelo = cargar_modelo()
    
    # Información del modelo en el sidebar, pero solo si tiene el atributo
    st.sidebar.subheader("🔍 Información del Modelo")
    
    if hasattr(modelo, "named_steps"):
        st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
        if hasattr(modelo, "feature_names_in_"):
            st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))
        else:
            st.sidebar.write("No se encontró 'feature_names_in_' en el modelo.")
        # Mostrar algunas features escaladas si existen
        if "preprocessor" in modelo.named_steps:
            scaler_features = modelo.named_steps['preprocessor'].transformers_[0][2]
            st.sidebar.write("Features escaladas:", scaler_features)
        else:
            st.sidebar.write("No se encontró el preprocessor en el pipeline.")
    else:
        st.sidebar.write(f"Modelo cargado: {type(modelo)}")
        if hasattr(modelo, "feature_names_in_"):
            st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))
        else:
            st.sidebar.write("El modelo no es un pipeline ni tiene atributos de pipeline.")

    # Verificar compatibilidad del modelo
    import sklearn
    st.sidebar.info(f"Scikit-learn actual: {sklearn.__version__}")
    st.sidebar.warning("⚠️ Modelo entrenado con sklearn 1.1.3, ejecutándose con versión actual")

except Exception as e:
    st.error("No se pudo cargar el modelo correctamente")
    st.stop()

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

if submit:
    try:
        # Crear diccionario con ceros para todas las features si las tiene
        if hasattr(modelo, "feature_names_in_"):
            datos = {col: 0 for col in modelo.feature_names_in_}
        else:
            st.error("El modelo no tiene atributo 'feature_names_in_', no se puede preparar la entrada.")
            st.stop()

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

        category_mappings = {
            "Marital status": (marital_options, marital),
            "Application mode": (app_mode_options, app_mode),
            "Course": (course_options, course),
            "Previous qualification": (prev_qual_options, prev_qual),
            "Nacionality": (nacionality_options, nacionality),
            "Mother's qualification": (mq_options, mq),
            "Father's qualification": (fq_options, fq),
            "Mother's occupation": (mo_options, mo),
            "Father's occupation": (fo_options, fo),
        }

        for key, (options, selected) in category_mappings.items():
            for option in options:
                colname = f"{key}_{option}"
                datos[colname] = 1 if selected == option else 0

        X_pred = pd.DataFrame([datos])

        # Predecir
        pred = modelo.predict(X_pred)[0]
        prob = modelo.predict_proba(X_pred)[0][1]

        if pred == 1:
            st.error(f"⚠️ Riesgo de deserción alto con probabilidad {prob:.2%}")
        else:
            st.success(f"✅ Riesgo bajo de deserción con probabilidad {1 - prob:.2%}")

    except Exception as e:
        st.error(f"Error al predecir: {e}")

