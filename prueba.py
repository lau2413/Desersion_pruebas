import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Limpiar cache si hay cambios
if st.button("🗑️ Limpiar Cache", help="Presiona si hay problemas con el modelo"):
    st.cache_resource.clear()
    st.rerun()

# Cargar el modelo (pipeline completo)
@st.cache_resource
def cargar_modelo():
    return joblib.load("pipeline_final_desercion.pkl")

try:
    modelo = cargar_modelo()
    
    # Información del modelo en el sidebar
    st.sidebar.subheader("🔍 Información del Modelo")
    st.sidebar.write("Pasos del pipeline:", list(modelo.named_steps.keys()))
    st.sidebar.write("Número de features esperadas:", len(modelo.feature_names_in_))
    
    # Verificar compatibilidad del modelo
    import sklearn
    st.sidebar.info(f"Scikit-learn actual: {sklearn.__version__}")
    st.sidebar.warning("⚠️ Modelo entrenado con sklearn 1.1.3, ejecutándose con versión actual")
    
    # Mostrar algunas features importantes
    scaler_features = modelo.named_steps['preprocessor'].transformers_[0][2]
    st.sidebar.write("Features escaladas:", scaler_features)

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
        
        for msg in validation_messages:
            if "❌" in msg:
                st.error(msg)
            else:
                st.success(msg)
        
        if not model_ok:
            st.error("**El modelo está corrupto. Necesitas regenerarlo o usar una versión compatible.**")
            st.stop()
        else:
            st.success("**Modelo validado correctamente**")
            
    except Exception as validation_error:
        st.error(f"Error en validación del modelo: {validation_error}")
        st.stop()

    # Procesamiento y predicción
if submit:
    datos = {col: 0 for col in modelo.feature_names_in_}

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
        "Father's occupation": (fo_options, fo)
    }

    for base_name, (options, selected) in category_mappings.items():
        for option in options:
            col_name = f"{base_name}_{option}"
            if col_name in modelo.feature_names_in_:
                datos[col_name] = 1 if option == selected else 0

    X = pd.DataFrame([datos])[modelo.feature_names_in_]

    # Conversión de tipos de datos corregida - usar tipos nativos de Python/numpy
    for col in X.columns:
        if col.startswith(("Marital status_", "Application mode_", "Course_",
                           "Previous qualification_", "Nacionality_",
                           "Mother's qualification_", "Father's qualification_",
                           "Mother's occupation_", "Father's occupation_")):
            # Convertir a int nativo de Python primero, luego a numpy
            X[col] = X[col].astype(int).astype('int64')
        elif col in scaler_features:
            # Conversión segura a float64 (más compatible)
            try:
                X[col] = pd.to_numeric(X[col], downcast=None).astype('float64')
            except (ValueError, TypeError):
                X[col] = X[col].astype('float64')
        else:
            # Conversión a tipos básicos
            try:
                X[col] = pd.to_numeric(X[col], downcast=None)
            except (ValueError, TypeError):
                pass  # Mantener el tipo original si no se puede convertir
    
    # Forzar conversión final a tipos compatibles con Arrow
    X = X.astype({col: 'float64' if X[col].dtype in ['int64', 'float32', 'float64'] 
                  else str for col in X.columns})

    with st.expander("🔍 Verificación de Tipos de Datos"):
        st.write("Tipos de datos finales:")
        # Mostrar tipos sin usar st.write con el DataFrame completo
        dtype_info = {}
        for col in X.columns:
            dtype_info[col] = str(X[col].dtype)
        
        # Contar tipos
        dtype_counts = {}
        for dtype in dtype_info.values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        st.write("Conteo de tipos:")
        for dtype, count in dtype_counts.items():
            st.write(f"- {dtype}: {count} columnas")
        
        object_cols = [col for col, dtype in dtype_info.items() if 'object' in dtype]
        if object_cols:
            st.write("Columnas problemáticas:", object_cols)

    try:
        if X.isnull().any().any():
            st.error("Error: Existen valores nulos en los datos")
            # Mostrar información de nulos sin serializar todo el DataFrame
            null_cols = []
            for col in X.columns:
                if X[col].isnull().any():
                    null_count = X[col].isnull().sum()
                    null_cols.append(f"{col}: {null_count} nulos")
            
            if null_cols:
                st.write("Columnas con valores nulos:")
                for info in null_cols:
                    st.write(f"- {info}")
            st.stop()

        # Conversión final a array con manejo de errores
        try:
            X_array = X.values.astype('float64')  # Usar float64 más compatible
        except (ValueError, TypeError) as e:
            st.error(f"Error en la conversión de tipos: {str(e)}")
            st.write("Tipos de datos actuales:")
            # Evitar mostrar el DataFrame completo para prevenir el error de Arrow
            for col in X.columns:
                st.write(f"{col}: {X[col].dtype}")
            st.stop()

        with st.spinner("Realizando predicción..."):
            # Hacer predicción con manejo de warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = modelo.predict(X)[0]
                proba = modelo.predict_proba(X)[0][1]

        st.subheader("📈 Resultado de la predicción:")
        if pred == 1:
            st.error(f"🚨 Riesgo de deserción (Probabilidad: {proba:.2%})")
        else:
            st.success(f"✅ Sin riesgo de deserción (Probabilidad: {proba:.2%})")

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
                            st.write(f"    - {i}: {name} -> `{type(transformer)}` -> {len(columns) if hasattr(columns, '__len__') else 'N/A'} columnas")
                            # Verificar si el transformer es una string (problema común)
                            if isinstance(transformer, str):
                                st.error(f"    ⚠️ PROBLEMA: Transformer {name} es un string: '{transformer}'")
                
            except Exception as model_error:
                st.error(f"Error inspeccionando modelo: {model_error}")
            
            # Información de los datos
            st.write("**Información de los Datos:**")
            if 'X_array' in locals():
                st.write(f"- Shape de los datos: `{X_array.shape}`")
                st.write(f"- Tipos de datos del array: `{X_array.dtype}`")
                st.write(f"- Valores mínimos: `{X_array.min()}`")
                st.write(f"- Valores máximos: `{X_array.max()}`")
                st.write(f"- Contiene NaN: `{np.isnan(X_array).any()}`")
                st.write(f"- Contiene Inf: `{np.isinf(X_array).any()}`")
            else:
                st.write("- Error antes de crear X_array")
            
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
                
                # Mostrar tipos de datos problemáticos
                problematic_types = {}
                for col in X.columns:
                    dtype_str = str(X[col].dtype)
                    if dtype_str not in ['float64', 'int64']:
                        problematic_types[col] = dtype_str
                
                if problematic_types:
                    st.error("- ⚠️ Tipos de datos problemáticos:")
                    for col, dtype in list(problematic_types.items())[:10]:  # Mostrar solo los primeros 10
                        st.write(f"    - `{col}`: `{dtype}`")
                    if len(problematic_types) > 10:
                        st.write(f"    - ... y {len(problematic_types) - 10} más")
                
                # Verificar valores únicos en columnas categóricas
                categorical_issues = []
                for col in X.columns:
                    if col.startswith(("Marital status_", "Application mode_", "Course_")):
                        unique_vals = X[col].unique()
                        if len(unique_vals) > 2 or any(val not in [0, 1] for val in unique_vals):
                            categorical_issues.append((col, unique_vals))
                
                if categorical_issues:
                    st.error("- ⚠️ Problemas en variables categóricas:")
                    for col, vals in categorical_issues[:5]:
                        st.write(f"    - `{col}`: valores únicos `{vals}`")
            
            # Intentar identificar dónde falla específicamente
            st.write("**Diagnóstico del Error:**")
            try:
                # Intentar cada paso del pipeline por separado
                st.write("Probando cada paso del pipeline...")
                
                current_data = X
                for step_name, step_obj in modelo.named_steps.items():
                    try:
                        st.write(f"- Probando paso '{step_name}'...")
                        
                        # Verificar si es un string (error común)
                        if isinstance(step_obj, str):
                            st.error(f"  ❌ PROBLEMA ENCONTRADO: '{step_name}' es un string: '{step_obj}'")
                            st.write("  **Solución:** El modelo parece estar corrupto. Necesitas regenerarlo.")
                            break
                        
                        if hasattr(step_obj, 'transform'):
                            current_data = step_obj.transform(current_data)
                            st.success(f"  ✅ '{step_name}' funcionó correctamente")
                        elif hasattr(step_obj, 'predict'):
                            # Es el clasificador final
                            prediction = step_obj.predict(current_data)
                            st.success(f"  ✅ '{step_name}' (clasificador) funcionó correctamente")
                        else:
                            st.warning(f"  ⚠️ '{step_name}' no tiene método transform ni predict")
                            
                    except Exception as step_error:
                        st.error(f"  ❌ Error en paso '{step_name}': {step_error}")
                        st.write(f"     Tipo de objeto: `{type(step_obj)}`")
                        st.write(f"     Métodos disponibles: `{[m for m in dir(step_obj) if not m.startswith('_')][:10]}`")
                        break
                        
            except Exception as debug_error:
                st.error(f"Error en diagnóstico: {debug_error}")
            
            # Recomendaciones de solución
            st.write("**Posibles Soluciones:**")
            st.write("1. **Regenerar el modelo:** El pipeline puede estar corrupto")
            st.write("2. **Verificar versiones:** Incompatibilidad entre versiones de sklearn")
            st.write("3. **Limpiar cache:** Usar el botón 'Limpiar Cache' arriba")
            st.write("4. **Re-entrenar:** Crear un nuevo modelo desde cero")
