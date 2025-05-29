import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# Limpiar cache si hay cambios
if st.button("🗑️ Limpiar Cache", help="Presiona si hay problemas con el modelo"):
    st.cache_resource.clear()
    st.rerun()

# Cargar el modelo (pipeline completo) y columnas esperadas
@st.cache_resource
def cargar_modelo_y_columnas():
    try:
        modelo = joblib.load("pipeline_final_desercion.pkl")
        # Intentar cargar las columnas esperadas
        try:
            columnas_esperadas = joblib.load("columnas_esperadas.pkl")
        except FileNotFoundError:
            # Si no existe el archivo, usar las del modelo
            columnas_esperadas = modelo.feature_names_in_.tolist()
        return modelo, columnas_esperadas
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

try:
    modelo, columnas_esperadas = cargar_modelo_y_columnas()
    
    if modelo is None:
        st.error("No se pudo cargar el modelo correctamente")
        st.stop()
    
    # Información del modelo en el sidebar
    st.sidebar.subheader("🔍 Información del Modelo")
    st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
    st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))
    
    # Verificar compatibilidad del modelo
    import sklearn
    import xgboost as xgb
    st.sidebar.info(f"Scikit-learn: {sklearn.__version__}")
    st.sidebar.info(f"XGBoost: {xgb.__version__}")
    
    # Información del preprocessor
    preprocessor = modelo.named_steps['preprocessor']
    st.sidebar.write("Transformers del preprocessor:")
    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(columns, '__len__'):
            st.sidebar.write(f"- {name}: {len(columns)} columnas")
        else:
            st.sidebar.write(f"- {name}: {columns}")

except Exception as e:
    st.error(f"Error crítico: {e}")
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

    # Validación del modelo antes de procesar
    st.write("**🔍 Validación del Modelo:**")
    try:
        # Verificar integridad del pipeline
        model_ok = True
        validation_messages = []
        
        for step_name, step_obj in modelo.named_steps.items():
            if isinstance(step_obj, str):
                validation_messages.append(f"❌ Error: '{step_name}' es un string en lugar de un objeto")
                model_ok = False
            else:
                validation_messages.append(f"✅ '{step_name}': {type(step_obj).__name__}")
        
        # Validación específica del nuevo preprocessor
        preprocessor = modelo.named_steps['preprocessor']
        validation_messages.append("**Validación del Preprocessor:**")
        
        # Verificar cada transformer
        for i, (name, transformer, columns) in enumerate(preprocessor.transformers_):
            if isinstance(transformer, str):
                if transformer == 'passthrough':
                    validation_messages.append(f"✅ Transformer {i} ({name}): passthrough - OK")
                elif transformer == 'drop':
                    validation_messages.append(f"⚠️ Transformer {i} ({name}): drop - OK pero inusual")
                else:
                    validation_messages.append(f"❌ Transformer {i} ({name}): string inválido '{transformer}'")
                    model_ok = False
            else:
                validation_messages.append(f"✅ Transformer {i} ({name}): {type(transformer).__name__} - OK")
        
        # Verificar remainder
        remainder_policy = getattr(preprocessor, 'remainder', 'drop')
        if remainder_policy == 'drop':
            validation_messages.append("✅ Remainder policy: 'drop' - OK")
        elif remainder_policy == 'passthrough':
            validation_messages.append("⚠️ Remainder policy: 'passthrough' - Puede causar problemas")
        else:
            validation_messages.append(f"❌ Remainder policy: '{remainder_policy}' - Problemático")
            model_ok = False
        
        for msg in validation_messages:
            if "❌" in msg:
                st.error(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            elif "**" in msg:
                st.write(msg)
            else:
                st.success(msg)
        
        if not model_ok:
            st.error("**El modelo tiene problemas. Necesitas regenerarlo.**")
            st.stop()
        else:
            st.success("**Modelo validado correctamente**")
            
    except Exception as validation_error:
        st.error(f"Error en validación del modelo: {validation_error}")
        st.stop()

# Procesamiento y predicción
if submit:
    try:
        # Crear un DataFrame con todas las columnas esperadas inicializadas en 0
        datos = pd.DataFrame(0, index=[0], columns=columnas_esperadas)
        
        # Llenar los valores básicos (variables numéricas y binarias)
        datos_basicos = {
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
        }
        
        # Actualizar datos básicos
        for col, val in datos_basicos.items():
            if col in datos.columns:
                datos[col] = val

        # Mapear variables categóricas (one-hot encoding)
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
                if col_name in datos.columns:
                    datos[col_name] = 1 if option == selected else 0

        # Asegurar que las columnas estén en el orden correcto
        X = datos[modelo.feature_names_in_].copy()
        
        # Verificar y ajustar tipos de datos según las listas del entrenamiento
        variables_int = [
            'Application order',
            'Daytime/evening attendance',
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
            'Curricular units 2nd sem (without evaluations)'
        ]

        variables_float = [
            'Previous qualification (grade)',
            'Admission grade',
            'Curricular units 2nd sem (grade)',
            'Unemployment rate',
            'Inflation rate',
            'GDP'
        ]

        # Convertir tipos de datos
        for col in X.columns:
            if col in variables_int:
                X[col] = X[col].astype('int64')
            elif col in variables_float:
                X[col] = X[col].astype('float64')
            else:
                # Variables booleanas (one-hot encoded) a int
                X[col] = X[col].astype('int64')

        # Mostrar información de verificación
        with st.expander("🔍 Verificación de Datos Procesados"):
            st.write("**Información del DataFrame de entrada:**")
            st.write(f"- Shape: {X.shape}")
            st.write(f"- Columnas: {len(X.columns)}")
            st.write(f"- Tipos de datos únicos: {X.dtypes.value_counts().to_dict()}")
            
            # Verificar valores nulos
            if X.isnull().any().any():
                st.error("⚠️ Hay valores nulos en los datos")
                null_info = []
                for col in X.columns:
                    if X[col].isnull().any():
                        null_count = X[col].isnull().sum()
                        null_info.append(f"{col}: {null_count}")
                st.write("Columnas con nulos:", null_info[:5])
            else:
                st.success("✅ No hay valores nulos")
            
            # Verificar rangos de algunas variables clave
            st.write("**Rangos de variables clave:**")
            key_vars = ['Application order', 'Age at enrollment', 'Previous qualification (grade)']
            for var in key_vars:
                if var in X.columns:
                    val = X[var].iloc[0]
                    st.write(f"- {var}: {val}")

        # Realizar predicción
        with st.spinner("Realizando predicción..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Predicción
                pred = modelo.predict(X)[0]
                proba = modelo.predict_proba(X)[0]
                
                # Obtener probabilidad de la clase positiva (deserción)
                prob_desercion = proba[1] if len(proba) > 1 else proba[0]

        # Mostrar resultados
        st.subheader("📈 Resultado de la predicción:")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if pred == 1:
                st.error(f"🚨 **RIESGO DE DESERCIÓN**")
                st.error(f"Probabilidad: **{prob_desercion:.2%}**")
            else:
                st.success(f"✅ **SIN RIESGO DE DESERCIÓN**")
                st.success(f"Probabilidad de no deserción: **{(1-prob_desercion):.2%}**")
        
        with col_result2:
            # Medidor visual de probabilidad
            st.metric(
                label="Probabilidad de Deserción",
                value=f"{prob_desercion:.2%}",
                delta=f"{'Alto riesgo' if prob_desercion > 0.5 else 'Bajo riesgo'}"
            )

        # Información adicional
        st.subheader("📊 Información adicional:")
        if prob_desercion > 0.7:
            st.warning("🔴 **Riesgo Muy Alto** - Se recomienda intervención inmediata")
        elif prob_desercion > 0.5:
            st.warning("🟡 **Riesgo Moderado** - Monitoreo recomendado")
        else:
            st.info("🟢 **Riesgo Bajo** - Situación favorable")

    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")
        
        with st.expander("🔍 Debugging Detallado", expanded=True):
            st.write("**Información del Error:**")
            st.write(f"- Tipo de error: `{type(e).__name__}`")
            st.write(f"- Mensaje completo: `{str(e)}`")
            
            # Información del modelo
            st.write("**Información del Modelo:**")
            try:
                st.write(f"- Tipo de modelo: `{type(modelo)}`")
                st.write(f"- Pasos del pipeline: `{list(modelo.named_steps.keys())}`")
                
                # Verificar cada paso del pipeline
                for step_name, step_obj in modelo.named_steps.items():
                    st.write(f"- {step_name}: `{type(step_obj)}`")
                    
                    # Si es el preprocessor, mostrar más detalles
                    if step_name == 'preprocessor':
                        st.write("  **Transformers en preprocessor:**")
                        for i, (name, transformer, columns) in enumerate(step_obj.transformers_):
                            col_count = len(columns) if hasattr(columns, '__len__') else 'N/A'
                            st.write(f"    - {i}: {name} -> `{type(transformer) if not isinstance(transformer, str) else transformer}` -> {col_count} columnas")
                            
                            # Verificar si el transformer es una string (problema común)
                            if isinstance(transformer, str) and transformer not in ['passthrough', 'drop']:
                                st.error(f"    ⚠️ PROBLEMA: Transformer {name} es un string inválido: '{transformer}'")
                
            except Exception as model_error:
                st.error(f"Error inspeccionando modelo: {model_error}")
            
            # Información de los datos
            st.write("**Información de los Datos:**")
            if 'X' in locals():
                st.write("**DataFrame X:**")
                st.write(f"- Shape: `{X.shape}`")
                st.write(f"- Columnas: `{len(X.columns)}`")
                st.write(f"- Features esperadas por modelo: `{len(modelo.feature_names_in_)}`")
                
                # Verificar si las columnas coinciden
                missing_features = set(modelo.feature_names_in_) - set(X.columns)
                extra_features = set(X.columns) - set(modelo.feature_names_in_)
                
                if missing_features:
                    st.error(f"- ⚠️ Features faltantes: `{list(missing_features)[:5]}...` ({len(missing_features)} total)")
                if extra_features:
                    st.warning(f"- ⚠️ Features extra: `{list(extra_features)[:5]}...` ({len(extra_features)} total)")
                
                # Mostrar tipos de datos
                dtype_counts = X.dtypes.value_counts()
                st.write("- Tipos de datos:")
                for dtype, count in dtype_counts.items():
                    st.write(f"    - `{dtype}`: {count} columnas")
                
                # Verificar valores únicos en algunas columnas categóricas
                problematic_cols = []
                for col in X.columns:
                    if col.startswith(("Marital status_", "Application mode_", "Course_")):
                        unique_vals = X[col].unique()
                        if len(unique_vals) > 2 or any(val not in [0, 1] for val in unique_vals):
                            problematic_cols.append((col, unique_vals))
                
                if problematic_cols:
                    st.error("- ⚠️ Problemas en variables categóricas:")
                    for col, vals in problematic_cols[:3]:
                        st.write(f"    - `{col}`: valores únicos `{vals}`")
            else:
                st.write("- Error: DataFrame X no fue creado")
            
            # Intentar identificar dónde falla específicamente
            st.write("**Diagnóstico Paso a Paso:**")
            try:
                st.write("Probando cada paso del pipeline...")
                
                current_data = X if 'X' in locals() else None
                if current_data is None:
                    st.error("- ❌ No se pudo crear el DataFrame de entrada")
                else:
                    for step_name, step_obj in modelo.named_steps.items():
                        try:
                            st.write(f"- Probando paso '{step_name}'...")
                            
                            # Verificar si es un string (error común)
                            if isinstance(step_obj, str):
                                st.error(f"  ❌ PROBLEMA: '{step_name}' es un string: '{step_obj}'")
                                st.write("  **Solución:** El modelo está corrupto. Regenera el modelo.")
                                break
                            
                            if hasattr(step_obj, 'transform'):
                                current_data = step_obj.transform(current_data)
                                st.success(f"  ✅ '{step_name}' funcionó correctamente")
                                st.write(f"     Output shape: `{current_data.shape if hasattr(current_data, 'shape') else 'N/A'}`")
                            elif hasattr(step_obj, 'predict'):
                                # Es el clasificador final
                                prediction = step_obj.predict(current_data)
                                probabilities = step_obj.predict_proba(current_data)
                                st.success(f"  ✅ '{step_name}' (clasificador) funcionó correctamente")
                                st.write(f"     Predicción: `{prediction}`")
                                st.write(f"     Probabilidades: `{probabilities}`")
                            else:
                                st.warning(f"  ⚠️ '{step_name}' no tiene método transform ni predict")
                                
                        except Exception as step_error:
                            st.error(f"  ❌ Error en paso '{step_name}': {step_error}")
                            st.write(f"     Tipo de objeto: `{type(step_obj)}`")
                            if hasattr(step_obj, '__dict__'):
                                relevant_attrs = {k: v for k, v in step_obj.__dict__.items() 
                                                if not k.startswith('_') and not callable(v)}
                                st.write(f"     Atributos relevantes: `{list(relevant_attrs.keys())[:5]}`")
                            break
                            
            except Exception as debug_error:
                st.error(f"Error en diagnóstico: {debug_error}")
            
            # Recomendaciones de solución
            st.write("**Posibles Soluciones:**")
            st.write("1. **Verificar compatibilidad:** Asegúrate de que el modelo fue entrenado con la nueva estructura")
            st.write("2. **Re-entrenar modelo:** Ejecuta el script de entrenamiento actualizado")
            st.write("3. **Verificar archivos:** Confirma que `pipeline_final_desercion.pkl` sea la versión nueva")
            st.write("4. **Limpiar cache:** Usa el botón 'Limpiar Cache' y recarga la página")
            st.write("5. **Verificar columnas:** Confirma que las columnas esperadas coincidan con el modelo")
