import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predictor de Deserción Estudiantil", layout="wide")

# Título de la aplicación
st.title("Sistema de Predicción de Deserción Estudiantil")
st.markdown("""
Esta aplicación predice la probabilidad de que un estudiante abandone sus estudios 
basándose en diversas características académicas y demográficas.
""")

# Cargar el modelo (asegúrate de tener el archivo en el repositorio)
@st.cache_resource
def load_model():
    try:
        import xgboost
        model = joblib.load('mejor_modelo_desercion.pkl')
        
        # Verificar que es un pipeline
        if not hasattr(model, 'steps'):
            st.error("El archivo cargado no es un pipeline de scikit-learn")
            return None
            
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Función para crear inputs del usuario
def get_user_input():
    st.sidebar.header("Información del Estudiante")
    
    # Dividir en secciones con expanders
    with st.sidebar.expander("Datos Básicos"):
        application_order = st.slider("Orden de Aplicación", 0, 10, 1)
        daytime_attendance = st.selectbox("Asistencia", ["Diurna", "Nocturna"], index=0)
        displaced = st.selectbox("Desplazado", ["No", "Sí"], index=0)
        debtor = st.selectbox("Deudor", ["No", "Sí"], index=0)
        tuition_up_to_date = st.selectbox("Matrícula al día", ["Sí", "No"], index=0)
        gender = st.selectbox("Género", ["Masculino", "Femenino"], index=0)
        scholarship = st.selectbox("Becado", ["No", "Sí"], index=0)
        age = st.slider("Edad al matricularse", 15, 60, 20)
    
    with st.sidebar.expander("Calificaciones Previas"):
        prev_qualification = st.slider("Calificación previa (0-200)", 0, 200, 100)
        admission_grade = st.slider("Calificación de admisión (0-200)", 0, 200, 100)
    
    with st.sidebar.expander("Rendimiento Académico - Primer Semestre"):
        cu_1st_sem_eval = st.slider("Unidades curriculares 1er sem (evaluadas)", 0, 20, 5)
        cu_1st_sem_no_eval = st.slider("Unidades curriculares 1er sem (no evaluadas)", 0, 20, 0)
    
    with st.sidebar.expander("Rendimiento Académico - Segundo Semestre"):
        cu_2nd_sem_credited = st.slider("Unidades curriculares 2do sem (con crédito)", 0, 20, 0)
        cu_2nd_sem_enrolled = st.slider("Unidades curriculares 2do sem (inscritas)", 0, 20, 5)
        cu_2nd_sem_eval = st.slider("Unidades curriculares 2do sem (evaluadas)", 0, 20, 5)
        cu_2nd_sem_approved = st.slider("Unidades curriculares 2do sem (aprobadas)", 0, 20, 5)
        cu_2nd_sem_grade = st.slider("Calificación promedio 2do sem (0-20)", 0.0, 20.0, 10.0)
        cu_2nd_sem_no_eval = st.slider("Unidades curriculares 2do sem (no evaluadas)", 0, 20, 0)
    
    with st.sidebar.expander("Indicadores Económicos"):
        unemployment = st.slider("Tasa de desempleo (%)", 0.0, 30.0, 10.0)
        inflation = st.slider("Tasa de inflación (%)", -5.0, 20.0, 2.0)
        gdp = st.slider("PIB (crecimiento anual %)", -10.0, 10.0, 2.0)
    
    with st.sidebar.expander("Estado Civil"):
        marital_status = st.radio("Estado civil", 
                                ["Soltero", "Casado", "Divorciado", "Unión de hecho", "Separado"])
    
    with st.sidebar.expander("Modo de Aplicación"):
        application_mode = st.radio("Modo de aplicación", 
                                  ["Admisión Regular", "Admisión Especial", 
                                   "Admisión por Ordenanza", "Cambios/Transferencias",
                                   "Estudiantes Internacionales", "Mayores de 23 años"])
    
    with st.sidebar.expander("Programa Académico"):
        course = st.selectbox("Programa de estudio", 
                            ["Ciencias Agrícolas y Ambientales", "Artes y Diseño",
                             "Negocios y Administración", "Comunicación y Medios",
                             "Educación", "Ingeniería y Tecnología",
                             "Ciencias de la Salud", "Ciencias Sociales"])
    
    with st.sidebar.expander("Calificación Previa"):
        prev_qualification_type = st.radio("Tipo de calificación previa", 
                                         ["Educación Secundaria", "Educación Técnica",
                                          "Educación Superior", "Otro"])
    
    with st.sidebar.expander("Nacionalidad"):
        nationality = st.selectbox("Nacionalidad", 
                                 ["Colombiana", "Cubana", "Holandesa", "Inglesa",
                                  "Alemana", "Italiana", "Lituana", "Moldava",
                                  "Mozambiqueña", "Portuguesa", "Rumana",
                                  "Santomeana", "Turca"])
    
    with st.sidebar.expander("Educación de los Padres"):
        mother_qual = st.radio("Educación de la madre", 
                              ["Básica/Secundaria", "Técnica", "Postgrado", "Otro/Desconocido"])
        father_qual = st.radio("Educación del padre", 
                              ["Básica/Secundaria", "Técnica", "Postgrado", "Otro/Desconocido"])
    
    with st.sidebar.expander("Ocupación de los Padres"):
        mother_occupation = st.selectbox("Ocupación de la madre", 
                                        ["Administrativo/Oficinista", "Trabajador Manual Calificado",
                                         "Casos Especiales", "Técnico/Profesional Asociado",
                                         "Trabajador No Calificado"])
        father_occupation = st.selectbox("Ocupación del padre", 
                                        ["Administrativo/Oficinista", "Profesional",
                                         "Trabajador Manual Calificado", "Casos Especiales",
                                         "Técnico/Profesional Asociado"])

    # Convertir inputs a formato del modelo
    data = {
        'Application order': application_order,
        'Daytime/evening attendance': 1 if daytime_attendance == "Diurna" else 0,
        'Previous qualification (grade)': prev_qualification,
        'Admission grade': admission_grade,
        'Displaced': 1 if displaced == "Sí" else 0,
        'Debtor': 1 if debtor == "Sí" else 0,
        'Tuition fees up to date': 1 if tuition_up_to_date == "Sí" else 0,
        'Gender': 1 if gender == "Masculino" else 0,
        'Scholarship holder': 1 if scholarship == "Sí" else 0,
        'Age at enrollment': age,
        'Curricular units 1st sem (evaluations)': cu_1st_sem_eval,
        'Curricular units 1st sem (without evaluations)': cu_1st_sem_no_eval,
        'Curricular units 2nd sem (credited)': cu_2nd_sem_credited,
        'Curricular units 2nd sem (enrolled)': cu_2nd_sem_enrolled,
        'Curricular units 2nd sem (evaluations)': cu_2nd_sem_eval,
        'Curricular units 2nd sem (approved)': cu_2nd_sem_approved,
        'Curricular units 2nd sem (grade)': cu_2nd_sem_grade,
        'Curricular units 2nd sem (without evaluations)': cu_2nd_sem_no_eval,
        'Unemployment rate': unemployment,
        'Inflation rate': inflation,
        'GDP': gdp,
        
        # Variables dummy (one-hot encoded)
        'Marital status_Divorced': 1 if marital_status == "Divorciado" else 0,
        'Marital status_FactoUnion': 1 if marital_status == "Unión de hecho" else 0,
        'Marital status_Separated': 1 if marital_status == "Separado" else 0,
        'Marital status_Single': 1 if marital_status == "Soltero" else 0,
        
        'Application mode_Admisión Especial': 1 if application_mode == "Admisión Especial" else 0,
        'Application mode_Admisión Regular': 1 if application_mode == "Admisión Regular" else 0,
        'Application mode_Admisión por Ordenanza': 1 if application_mode == "Admisión por Ordenanza" else 0,
        'Application mode_Cambios/Transferencias': 1 if application_mode == "Cambios/Transferencias" else 0,
        'Application mode_Estudiantes Internacionales': 1 if application_mode == "Estudiantes Internacionales" else 0,
        'Application mode_Mayores de 23 años': 1 if application_mode == "Mayores de 23 años" else 0,
        
        'Course_Agricultural & Environmental Sciences': 1 if course == "Ciencias Agrícolas y Ambientales" else 0,
        'Course_Arts & Design': 1 if course == "Artes y Diseño" else 0,
        'Course_Business & Management': 1 if course == "Negocios y Administración" else 0,
        'Course_Communication & Media': 1 if course == "Comunicación y Medios" else 0,
        'Course_Education': 1 if course == "Educación" else 0,
        'Course_Engineering & Technology': 1 if course == "Ingeniería y Tecnología" else 0,
        'Course_Health Sciences': 1 if course == "Ciencias de la Salud" else 0,
        'Course_Social Sciences': 1 if course == "Ciencias Sociales" else 0,
        
        'Previous qualification_Higher Education': 1 if prev_qualification_type == "Educación Superior" else 0,
        'Previous qualification_Other': 1 if prev_qualification_type == "Otro" else 0,
        'Previous qualification_Secondary Education': 1 if prev_qualification_type == "Educación Secundaria" else 0,
        'Previous qualification_Technical Education': 1 if prev_qualification_type == "Educación Técnica" else 0,
        
        'Nacionality_Colombian': 1 if nationality == "Colombiana" else 0,
        'Nacionality_Cuban': 1 if nationality == "Cubana" else 0,
        'Nacionality_Dutch': 1 if nationality == "Holandesa" else 0,
        'Nacionality_English': 1 if nationality == "Inglesa" else 0,
        'Nacionality_German': 1 if nationality == "Alemana" else 0,
        'Nacionality_Italian': 1 if nationality == "Italiana" else 0,
        'Nacionality_Lithuanian': 1 if nationality == "Lituana" else 0,
        'Nacionality_Moldovan': 1 if nationality == "Moldava" else 0,
        'Nacionality_Mozambican': 1 if nationality == "Mozambiqueña" else 0,
        'Nacionality_Portuguese': 1 if nationality == "Portuguesa" else 0,
        'Nacionality_Romanian': 1 if nationality == "Rumana" else 0,
        'Nacionality_Santomean': 1 if nationality == "Santomeana" else 0,
        'Nacionality_Turkish': 1 if nationality == "Turca" else 0,
        
        'Mother\'s qualification_Basic_or_Secondary': 1 if mother_qual == "Básica/Secundaria" else 0,
        'Mother\'s qualification_Other_or_Unknown': 1 if mother_qual == "Otro/Desconocido" else 0,
        'Mother\'s qualification_Postgraduate': 1 if mother_qual == "Postgrado" else 0,
        'Mother\'s qualification_Technical_Education': 1 if mother_qual == "Técnica" else 0,
        
        'Father\'s qualification_Basic_or_Secondary': 1 if father_qual == "Básica/Secundaria" else 0,
        'Father\'s qualification_Other_or_Unknown': 1 if father_qual == "Otro/Desconocido" else 0,
        'Father\'s qualification_Postgraduate': 1 if father_qual == "Postgrado" else 0,
        
        'Mother\'s occupation_Administrative/Clerical': 1 if mother_occupation == "Administrativo/Oficinista" else 0,
        'Mother\'s occupation_Skilled Manual Workers': 1 if mother_occupation == "Trabajador Manual Calificado" else 0,
        'Mother\'s occupation_Special Cases': 1 if mother_occupation == "Casos Especiales" else 0,
        'Mother\'s occupation_Technicians/Associate Professionals': 1 if mother_occupation == "Técnico/Profesional Asociado" else 0,
        'Mother\'s occupation_Unskilled Workers': 1 if mother_occupation == "Trabajador No Calificado" else 0,
        
        'Father\'s occupation_Administrative/Clerical': 1 if father_occupation == "Administrativo/Oficinista" else 0,
        'Father\'s occupation_Professionals': 1 if father_occupation == "Profesional" else 0,
        'Father\'s occupation_Skilled Manual Workers': 1 if father_occupation == "Trabajador Manual Calificado" else 0,
        'Father\'s occupation_Special Cases': 1 if father_occupation == "Casos Especiales" else 0,
        'Father\'s occupation_Technicians/Associate Professionals': 1 if father_occupation == "Técnico/Profesional Asociado" else 0
    }
    
    # Crear DataFrame con todas las columnas necesarias
    features = pd.DataFrame(data, index=[0])
    
    # Asegurarse de que todas las columnas estén presentes
    # (añadir cualquier columna faltante con valor 0)
    expected_columns = [
        'Application order', 'Daytime/evening attendance', 'Previous qualification (grade)', 
        'Admission grade', 'Displaced', 'Debtor', 'Tuition fees up to date', 'Gender', 
        'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (evaluations)', 
        'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 
        'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 
        'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 
        'Inflation rate', 'GDP', 'Marital status_Divorced', 'Marital status_FactoUnion', 
        'Marital status_Separated', 'Marital status_Single', 'Application mode_Admisión Especial', 
        'Application mode_Admisión Regular', 'Application mode_Admisión por Ordenanza', 
        'Application mode_Cambios/Transferencias', 'Application mode_Estudiantes Internacionales', 
        'Application mode_Mayores de 23 años', 'Course_Agricultural & Environmental Sciences', 
        'Course_Arts & Design', 'Course_Business & Management', 'Course_Communication & Media', 
        'Course_Education', 'Course_Engineering & Technology', 'Course_Health Sciences', 
        'Course_Social Sciences', 'Previous qualification_Higher Education', 
        'Previous qualification_Other', 'Previous qualification_Secondary Education', 
        'Previous qualification_Technical Education', 'Nacionality_Colombian', 
        'Nacionality_Cuban', 'Nacionality_Dutch', 'Nacionality_English', 'Nacionality_German', 
        'Nacionality_Italian', 'Nacionality_Lithuanian', 'Nacionality_Moldovan', 
        'Nacionality_Mozambican', 'Nacionality_Portuguese', 'Nacionality_Romanian', 
        'Nacionality_Santomean', 'Nacionality_Turkish', 
        'Mother\'s qualification_Basic_or_Secondary', 'Mother\'s qualification_Other_or_Unknown', 
        'Mother\'s qualification_Postgraduate', 'Mother\'s qualification_Technical_Education', 
        'Father\'s qualification_Basic_or_Secondary', 'Father\'s qualification_Other_or_Unknown', 
        'Father\'s qualification_Postgraduate', 
        'Mother\'s occupation_Administrative/Clerical', 
        'Mother\'s occupation_Skilled Manual Workers', 'Mother\'s occupation_Special Cases', 
        'Mother\'s occupation_Technicians/Associate Professionals', 
        'Mother\'s occupation_Unskilled Workers', 
        'Father\'s occupation_Administrative/Clerical', 'Father\'s occupation_Professionals', 
        'Father\'s occupation_Skilled Manual Workers', 'Father\'s occupation_Special Cases', 
        'Father\'s occupation_Technicians/Associate Professionals'
    ]
    
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Reordenar columnas para que coincidan con el orden de entrenamiento
    features = features[expected_columns]
    
    return features

# Obtener input del usuario
user_input = get_user_input()

# Mostrar los datos ingresados (opcional)
if st.checkbox("Mostrar datos ingresados"):
    st.subheader("Datos ingresados")
    st.write(user_input)

# Botón para realizar predicción
if st.button("Predecir Probabilidad de Deserción"):
    try:
        # Realizar predicción
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)
        
        # Mostrar resultados
        st.subheader("Resultados de la Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicción", "Deserción" if prediction[0] == 1 else "No Deserción")
        
        with col2:
            st.metric("Probabilidad de Deserción", f"{prediction_proba[0][1]*100:.2f}%")
        
        # Mostrar gráfico de probabilidad
        st.progress(int(prediction_proba[0][1]*100))
        
        # Interpretación de resultados
        if prediction[0] == 1:
            st.warning("El estudiante tiene alto riesgo de deserción. Se recomienda intervención.")
        else:
            st.success("El estudiante tiene bajo riesgo de deserción.")
        
        # Explicación de factores (simplificada)
        st.subheader("Factores Clave")
        st.markdown("""
        Los principales factores que influyen en esta predicción incluyen:
        - Rendimiento académico en el primer y segundo semestre
        - Situación económica (becas, deudas, pago de matrícula)
        - Indicadores macroeconómicos (desempleo, inflación)
        - Características demográficas y educativas previas
        """)
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
