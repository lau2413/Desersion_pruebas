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

# Cargar el modelo con mejor manejo de errores
@st.cache_resource
def load_model():
    try:
        import xgboost
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        model = joblib.load('mejor_modelo_desercion.pkl')
        
        # Debug: Mostrar información sobre el modelo cargado
        st.sidebar.write(f"Tipo de modelo cargado: {type(model)}")
        
        # Verificar si es un pipeline
        if hasattr(model, 'steps'):
            st.sidebar.write(f"Pipeline con pasos: {[step[0] for step in model.steps]}")
            
            # Inspeccionar cada paso del pipeline
            for i, (name, step) in enumerate(model.steps):
                st.sidebar.write(f"Paso {i}: {name} - Tipo: {type(step)}")
                
                # Si encontramos un string en lugar de un objeto, intentar repararlo
                if isinstance(step, str):
                    st.warning(f"⚠️ Paso '{name}' es un string: {step}")
                    st.warning("Intentando cargar modelo sin preprocessor...")
                    
                    # Intentar usar solo el clasificador si el preprocessor está corrupto
                    try:
                        # Si el preprocessor está corrupto, usar solo el clasificador
                        if name == 'preprocessor':
                            # Crear un preprocessor dummy que no haga nada
                            from sklearn.preprocessing import FunctionTransformer
                            model.steps[i] = (name, FunctionTransformer(lambda x: x))
                            st.info("✅ Preprocessor reemplazado con transformador identidad")
                    except Exception as repair_error:
                        st.error(f"No se pudo reparar el paso {name}: {repair_error}")
                        return None
            
        elif hasattr(model, 'predict'):
            st.sidebar.write("Modelo directo (no pipeline)")
        else:
            st.error("El objeto cargado no tiene método predict")
            return None
            
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo 'mejor_modelo_desercion.pkl'. Asegúrate de que esté en el directorio correcto.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.error(f"Tipo de error: {type(e)}")
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
    
    return data

# Función para preparar los datos para el modelo
def prepare_data_for_model(data_dict):
    """
    Convierte el diccionario de datos a DataFrame y asegura el formato correcto
    """
    # Crear DataFrame
    df = pd.DataFrame([data_dict])
    
    # Lista de todas las columnas esperadas (puedes necesitar ajustar esto según tu modelo)
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
    
    # Agregar columnas faltantes con valor 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reordenar columnas para que coincidan con el orden esperado
    df = df[expected_columns]
    
    # Asegurar tipos de datos correctos
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# Obtener input del usuario
user_data = get_user_input()

# Mostrar los datos ingresados (opcional)
if st.checkbox("Mostrar datos ingresados"):
    st.subheader("Datos ingresados")
    prepared_data = prepare_data_for_model(user_data)
    st.write(prepared_data)
    st.write(f"Forma de los datos: {prepared_data.shape}")

# Función para realizar predicción robusta
def robust_predict(model, data):
    """
    Función que maneja diferentes tipos de errores en la predicción
    """
    try:
        # Intento 1: Predicción normal
        prediction = model.predict(data)
        prediction_proba = None
        
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(data)
            
        return prediction, prediction_proba
        
    except AttributeError as e:
        if "'str' object has no attribute 'transform'" in str(e):
            st.warning("⚠️ Pipeline corrupto detectado. Intentando solución alternativa...")
            
            # Intento 2: Acceder directamente al clasificador
            try:
                if hasattr(model, 'steps') and len(model.steps) >= 2:
                    classifier = model.steps[-1][1]  # Último paso (clasificador)
                    
                    if hasattr(classifier, 'predict'):
                        st.info("✅ Usando clasificador directamente (sin preprocessor)")
                        prediction = classifier.predict(data)
                        prediction_proba = None
                        
                        if hasattr(classifier, 'predict_proba'):
                            prediction_proba = classifier.predict_proba(data)
                            
                        return prediction, prediction_proba
                    
            except Exception as e2:
                st.error(f"Error al acceder al clasificador: {e2}")
                
            # Intento 3: Recrear el pipeline
            try:
                st.info("🔧 Intentando recrear el pipeline...")
                from sklearn.preprocessing import FunctionTransformer
                from sklearn.pipeline import Pipeline
                
                # Obtener el clasificador
                if hasattr(model, 'steps') and len(model.steps) >= 2:
                    classifier = model.steps[-1][1]
                    
                    # Crear un nuevo pipeline con transformador identidad
                    new_pipeline = Pipeline([
                        ('preprocessor', FunctionTransformer(lambda x: x)),
                        ('classifier', classifier)
                    ])
                    
                    prediction = new_pipeline.predict(data)
                    prediction_proba = None
                    
                    if hasattr(new_pipeline, 'predict_proba'):
                        prediction_proba = new_pipeline.predict_proba(data)
                        
                    st.success("✅ Pipeline recreado exitosamente")
                    return prediction, prediction_proba
                    
            except Exception as e3:
                st.error(f"Error al recrear pipeline: {e3}")
                
        raise e  # Re-lanzar el error original si no se pudo solucionar
    try:
        # Preparar datos para el modelo
        model_input = prepare_data_for_model(user_data)
        
        # Debug: Mostrar información sobre los datos preparados
        st.write(f"Forma de los datos de entrada: {model_input.shape}")
        st.write(f"Tipos de datos: {model_input.dtypes.unique()}")
        
        # Verificar si hay valores NaN
        if model_input.isnull().any().any():
            st.warning("Se encontraron valores NaN en los datos. Se rellenarán con 0.")
            model_input = model_input.fillna(0)
        
        # Realizar predicción
        prediction = model.predict(model_input)
        
        # Verificar si el modelo tiene predict_proba
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(model_input)
        else:
            # Si no tiene predict_proba, usar un valor aproximado basado en la predicción
            if prediction[0] == 1:
                prediction_proba = np.array([[0.2, 0.8]])  # Alta probabilidad de deserción
            else:
                prediction_proba = np.array([[0.8, 0.2]])  # Baja probabilidad de deserción
        # Manejar casos donde prediction_proba es None
        if prediction_proba is None:
            # Si no tiene predict_proba, usar un valor aproximado basado en la predicción
            if prediction[0] == 1:
                prediction_proba = np.array([[0.2, 0.8]])  # Alta probabilidad de deserción
            else:
                prediction_proba = np.array([[0.8, 0.2]])  # Baja probabilidad de deserción
        st.subheader("Resultados de la Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            result_text = "Deserción" if prediction[0] == 1 else "No Deserción"
            st.metric("Predicción", result_text)
        
        with col2:
            prob_desercion = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else prediction_proba[0][0]
            st.metric("Probabilidad de Deserción", f"{prob_desercion*100:.2f}%")
        
        # Mostrar gráfico de probabilidad
        st.progress(int(prob_desercion*100))
        
        # Interpretación de resultados
        if prediction[0] == 1:
            st.warning("⚠️ El estudiante tiene alto riesgo de deserción. Se recomienda intervención.")
        else:
            st.success("✅ El estudiante tiene bajo riesgo de deserción.")
        
        # Explicación de factores (simplificada)
        st.subheader("Factores Clave")
        st.markdown("""
        Los principales factores que influyen en esta predicción incluyen:
        - **Rendimiento académico**: Calificaciones y unidades curriculares del primer y segundo semestre
        - **Situación económica**: Becas, deudas, pago de matrícula al día
        - **Indicadores macroeconómicos**: Tasa de desempleo, inflación, PIB
        - **Características personales**: Edad, género, estado civil
        - **Antecedentes educativos**: Calificación previa, tipo de educación
        - **Contexto familiar**: Educación y ocupación de los padres
        """)
        
        # Recomendaciones basadas en el riesgo
        if prediction[0] == 1:
            st.subheader("Recomendaciones de Intervención")
            st.markdown("""
            **Acciones sugeridas para reducir el riesgo de deserción:**
            - 🎯 Proporcionar tutorías académicas personalizadas
            - 💰 Evaluar opciones de apoyo financiero adicional
            - 👥 Asignar un mentor o consejero estudiantil
            - 📚 Ofrecer programas de nivelación académica
            - 🤝 Facilitar grupos de estudio y apoyo entre pares
            - 📞 Establecer seguimiento regular del progreso académico
            """)
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.error(f"Tipo de error: {type(e)}")
        
        # Información adicional para debug
        st.subheader("Información de Debug")
        st.write("Si el error persiste, verifica:")
        st.write("1. Que el archivo del modelo sea el correcto")
        st.write("2. Que las columnas del modelo coincidan con las esperadas")
        st.write("3. Que el modelo fue entrenado con la versión correcta de scikit-learn/xgboost")
        
        # Mostrar estructura de los datos para debug
        model_input = prepare_data_for_model(user_data)
        st.write(f"Columnas de entrada: {model_input.columns.tolist()}")
        st.write(f"Primeras filas de datos:")
        st.write(model_input.head())
