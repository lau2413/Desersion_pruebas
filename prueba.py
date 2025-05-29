import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎓 Predictor de Deserción Estudiantil")
st.markdown("### Modelo XGBoost para predicción de deserción académica")

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Intentar cargar el pipeline completo
        model = joblib.load('pipeline_final_desercion.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo del modelo. Asegúrate de que 'pipeline_final_desercion.pkl' esté en el directorio.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Función para obtener las columnas esperadas en el orden correcto
def get_expected_columns():
    """Retorna las columnas en el orden exacto que espera el modelo"""
    return [
        'Application order',
        'Daytime/evening attendance',
        'Previous qualification (grade)',
        'Admission grade',
        'Displaced',
        'Debtor',
        'Tuition fees up to date',
        'Gender',
        'Scholarship holder',
        'Age at enrollment',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate',
        'Inflation rate',
        'GDP',
        'Marital status_Divorced',
        'Marital status_FactoUnion',
        'Marital status_Separated',
        'Marital status_Single',
        'Application mode_Admisión Especial',
        'Application mode_Admisión Regular',
        'Application mode_Admisión por Ordenanza',
        'Application mode_Cambios/Transferencias',
        'Application mode_Estudiantes Internacionales',
        'Application mode_Mayores de 23 años',
        'Course_Agricultural & Environmental Sciences',
        'Course_Arts & Design',
        'Course_Business & Management',
        'Course_Communication & Media',
        'Course_Education',
        'Course_Engineering & Technology',
        'Course_Health Sciences',
        'Course_Social Sciences',
        'Previous qualification_Higher Education',
        'Previous qualification_Other',
        'Previous qualification_Secondary Education',
        'Previous qualification_Technical Education',
        'Nacionality_Colombian',
        'Nacionality_Cuban',
        'Nacionality_Dutch',
        'Nacionality_English',
        'Nacionality_German',
        'Nacionality_Italian',
        'Nacionality_Lithuanian',
        'Nacionality_Moldovan',
        'Nacionality_Mozambican',
        'Nacionality_Portuguese',
        'Nacionality_Romanian',
        'Nacionality_Santomean',
        'Nacionality_Turkish',
        "Mother's qualification_Basic_or_Secondary",
        "Mother's qualification_Other_or_Unknown",
        "Mother's qualification_Postgraduate",
        "Mother's qualification_Technical_Education",
        "Father's qualification_Basic_or_Secondary",
        "Father's qualification_Other_or_Unknown",
        "Father's qualification_Postgraduate",
        "Mother's occupation_Administrative/Clerical",
        "Mother's occupation_Skilled Manual Workers",
        "Mother's occupation_Special Cases",
        "Mother's occupation_Technicians/Associate Professionals",
        "Mother's occupation_Unskilled Workers",
        "Father's occupation_Administrative/Clerical",
        "Father's occupation_Professionals",
        "Father's occupation_Skilled Manual Workers",
        "Father's occupation_Special Cases",
        "Father's occupation_Technicians/Associate Professionals"
    ]

# Función para crear el DataFrame con los valores por defecto
def create_input_dataframe(user_inputs):
    """Crea un DataFrame con todas las columnas necesarias en el orden correcto"""
    expected_columns = get_expected_columns()
    
    # Crear DataFrame con todas las columnas inicializadas en 0
    df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Asignar valores del usuario
    for col, value in user_inputs.items():
        if col in df.columns:
            df[col] = value
    
    # Asegurar tipos de datos correctos
    # Variables enteras
    int_cols = [
        'Application order', 'Daytime/evening attendance', 'Displaced', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (without evaluations)'
    ]
    
    # Variables flotantes
    float_cols = [
        'Previous qualification (grade)', 'Admission grade', 'Curricular units 2nd sem (grade)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    # Convertir tipos
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Las variables dummy ya son int por defecto (0 o 1)
    
    return df

# Interfaz de usuario
def main():
    # Cargar modelo
    model = load_model()
    if model is None:
        st.stop()
    
    st.sidebar.header("📊 Parámetros del Estudiante")
    
    # Crear columnas para organizar mejor los inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información Académica")
        
        # Inputs académicos
        application_order = st.number_input("Orden de Aplicación", min_value=0, max_value=10, value=1)
        daytime_evening = st.selectbox("Modalidad", ["Diurno (1)", "Nocturno (0)"], index=0)
        daytime_evening_val = 1 if "Diurno" in daytime_evening else 0
        
        prev_qualification_grade = st.slider("Calificación Previa (%)", 0.0, 100.0, 75.0)
        admission_grade = st.slider("Calificación de Admisión (%)", 0.0, 200.0, 120.0)
        
        age_enrollment = st.number_input("Edad al Matricularse", min_value=16, max_value=70, value=20)
        
        # Unidades curriculares 1er semestre
        st.subheader("1er Semestre")
        cu_1sem_eval = st.number_input("Evaluaciones 1er Sem", min_value=0, max_value=30, value=6)
        cu_1sem_without_eval = st.number_input("Sin Evaluaciones 1er Sem", min_value=0, max_value=30, value=0)
        
        # Unidades curriculares 2do semestre
        st.subheader("2do Semestre")
        cu_2sem_credited = st.number_input("Acreditadas 2do Sem", min_value=0, max_value=30, value=0)
        cu_2sem_enrolled = st.number_input("Matriculadas 2do Sem", min_value=0, max_value=30, value=6)
        cu_2sem_eval = st.number_input("Evaluaciones 2do Sem", min_value=0, max_value=30, value=6)
        cu_2sem_approved = st.number_input("Aprobadas 2do Sem", min_value=0, max_value=30, value=5)
        cu_2sem_grade = st.slider("Calificación 2do Sem", 0.0, 20.0, 12.0)
        cu_2sem_without_eval = st.number_input("Sin Evaluaciones 2do Sem", min_value=0, max_value=30, value=0)
    
    with col2:
        st.subheader("Información Personal")
        
        # Información personal
        displaced = st.selectbox("Desplazado", ["No (0)", "Sí (1)"], index=0)
        displaced_val = 1 if "Sí" in displaced else 0
        
        debtor = st.selectbox("Deudor", ["No (0)", "Sí (1)"], index=0)
        debtor_val = 1 if "Sí" in debtor else 0
        
        tuition_fees = st.selectbox("Matrícula al día", ["No (0)", "Sí (1)"], index=1)
        tuition_fees_val = 1 if "Sí" in tuition_fees else 0
        
        gender = st.selectbox("Género", ["Masculino (1)", "Femenino (0)"], index=0)
        gender_val = 1 if "Masculino" in gender else 0
        
        scholarship = st.selectbox("Becario", ["No (0)", "Sí (1)"], index=0)
        scholarship_val = 1 if "Sí" in scholarship else 0
        
        # Estado civil
        st.subheader("Estado Civil")
        marital_status = st.selectbox(
            "Estado Civil",
            ["Single", "Divorced", "FactoUnion", "Separated"]
        )
        
        # Modo de aplicación
        st.subheader("Modo de Aplicación")
        application_mode = st.selectbox(
            "Modo de Aplicación",
            ["Admisión Regular", "Admisión Especial", "Admisión por Ordenanza", 
             "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"]
        )
        
        # Curso
        st.subheader("Programa Académico")
        course = st.selectbox(
            "Programa",
            ["Engineering & Technology", "Business & Management", "Health Sciences",
             "Social Sciences", "Arts & Design", "Education", "Communication & Media",
             "Agricultural & Environmental Sciences"]
        )
        
        # Variables económicas
        st.subheader("Variables Económicas")
        unemployment_rate = st.slider("Tasa de Desempleo (%)", 0.0, 30.0, 10.0)
        inflation_rate = st.slider("Tasa de Inflación (%)", -5.0, 15.0, 2.0)
        gdp = st.slider("PIB", -10.0, 10.0, 2.0)
    
    # Botón de predicción
    if st.button("🔮 Realizar Predicción", type="primary"):
        try:
            # Crear diccionario con los inputs del usuario
            user_inputs = {
                'Application order': application_order,
                'Daytime/evening attendance': daytime_evening_val,
                'Previous qualification (grade)': prev_qualification_grade,
                'Admission grade': admission_grade,
                'Displaced': displaced_val,
                'Debtor': debtor_val,
                'Tuition fees up to date': tuition_fees_val,
                'Gender': gender_val,
                'Scholarship holder': scholarship_val,
                'Age at enrollment': age_enrollment,
                'Curricular units 1st sem (evaluations)': cu_1sem_eval,
                'Curricular units 1st sem (without evaluations)': cu_1sem_without_eval,
                'Curricular units 2nd sem (credited)': cu_2sem_credited,
                'Curricular units 2nd sem (enrolled)': cu_2sem_enrolled,
                'Curricular units 2nd sem (evaluations)': cu_2sem_eval,
                'Curricular units 2nd sem (approved)': cu_2sem_approved,
                'Curricular units 2nd sem (grade)': cu_2sem_grade,
                'Curricular units 2nd sem (without evaluations)': cu_2sem_without_eval,
                'Unemployment rate': unemployment_rate,
                'Inflation rate': inflation_rate,
                'GDP': gdp,
                f'Marital status_{marital_status}': 1,
                f'Application mode_{application_mode}': 1,
                f'Course_{course}': 1
            }
            
            # Crear DataFrame con el orden correcto
            input_df = create_input_dataframe(user_inputs)
            
            # Realizar predicción
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultados de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ ALTO RIESGO DE DESERCIÓN")
                else:
                    st.success("✅ BAJO RIESGO DE DESERCIÓN")
            
            with col2:
                prob_desercion = prediction_proba[1] * 100
                st.metric(
                    label="Probabilidad de Deserción",
                    value=f"{prob_desercion:.1f}%"
                )
            
            with col3:
                prob_permanencia = prediction_proba[0] * 100
                st.metric(
                    label="Probabilidad de Permanencia",
                    value=f"{prob_permanencia:.1f}%"
                )
            
            # Gráfico de probabilidades
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Permanencia', 'Deserción']
            probabilities = [prediction_proba[0], prediction_proba[1]]
            colors = ['green', 'red']
            
            bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
            ax.set_ylabel('Probabilidad')
            ax.set_title('Probabilidades de Predicción')
            ax.set_ylim(0, 1)
            
            # Añadir valores en las barras
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones")
            if prediction == 1:
                st.warning("""
                **Acciones Recomendadas:**
                - Implementar programa de tutorías académicas
                - Seguimiento personalizado del estudiante
                - Evaluar situación socioeconómica
                - Considerar flexibilización de horarios
                - Apoyo psicopedagógico
                """)
            else:
                st.info("""
                **Mantener Seguimiento:**
                - Continuar con el rendimiento académico actual
                - Monitoreo periódico del progreso
                - Motivar la participación en actividades extracurriculares
                """)
                
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            st.write("Detalles del error para debugging:")
            st.write(f"Tipo de error: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎓 Predictor de Deserción Estudiantil | Modelo XGBoost | Desarrollado con Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()

