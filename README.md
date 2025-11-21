## Integrante del proyecto
- Breyneer Nieto Cardeño
# Iris Species Classification — Proyecto de Machine Learning

# Iris Species Classification — Proyecto

Este repositorio contiene la aplicación Streamlit para clasificar especies de Iris, así como los archivos del proyecto, modelo entrenado y guion del video.

## Contenido
- `proyect.py` — Código completo de la app Streamlit.
- `requirements.txt` — Dependencias del proyecto.
- `iris_rf_model.joblib` — Modelo Random Forest entrenado listo para usar.
- `video_presentation_link.txt` — (https://youtu.be/YBdfUI9EfWc)
- `IRIS SPECIES CLASSIFICATION PROYECT.pdf` — Documento base del proyecto.

## Propósito del Proyecto

El objetivo de este proyecto es desarrollar un modelo de clasificación capaz de predecir la especie de una flor Iris utilizando cuatro características morfológicas:  
- Longitud del sépalo  
- Ancho del sépalo  
- Longitud del pétalo  
- Ancho del pétalo  

El proyecto incluye:  
✔ Entrenamiento de un modelo de Machine Learning  
✔ Construcción de un dashboard interactivo con Streamlit  
✔ Visualización de datos y predicciones  
✔ Archivo con todos los requisitos del curso

---

## 🔬 Metodología y Flujo de Trabajo (Workflow)

El proyecto se desarrolló siguiendo este flujo:

1. **Carga del dataset**  
   Se utilizó el dataset Iris incluido en `sklearn.datasets`, el cual ya viene limpio y estructurado.

2. **Exploración inicial**  
   Se analizaron las distribuciones mediante histogramas y una matriz de dispersión para comprender la relación entre las características.

3. **División del dataset**  
   Se separaron los datos en entrenamiento y prueba (75% – 25%) utilizando estratificación para mantener el equilibrio entre clases.

4. **Selección del modelo**  
   El algoritmo elegido fue **Random Forest**, debido a:  
   - Su alta precisión en problemas de clasificación  
   - Su robustez ante sobreajuste  
   - Su buen rendimiento con datasets pequeños  
   - Su facilidad de interpretación

5. **Entrenamiento del modelo**  
   Se entrenó ajustando los hiperparámetros `n_estimators` y `random_state`.

6. **Evaluación**  
   Se calcularon métricas como:  
   - **Accuracy**  
   - **Precision (weighted)**  
   - **Recall (weighted)**  
   - **F1-score (weighted)**  

7. **Implementación del dashboard**  
   Se desarrolló una aplicación con Streamlit para:  
   - Visualizar métricas  
   - Realizar predicciones manuales  
   - Explorar los datos  
   - Visualizar puntos en gráficas 3D  
   - Guardar el modelo entrenado

---

## 🖥️ Descripción del Dashboard

La aplicación construida en Streamlit incluye:

### **1️⃣ Panel lateral**
- Configuración del modelo (`n_estimators` y `random_state`)  
- Sliders para ingresar una flor nueva  
- Botón para generar una predicción

### **2️⃣ Métricas del modelo**
- Accuracy  
- Precision  
- Recall  
- F1-score  

### **3️⃣ Predicción interactiva**
Al ingresar una muestra nueva:
- Se muestra la especie predicha  
- Se despliega la probabilidad para cada clase  
- Se grafica un punto “X” en una gráfica 3D que representa la flor

### **4️⃣ Visualización de datos**
Incluye:
- Histogramas por característica  
- Matriz de dispersión  
- Boxplots por especie  

### **5️⃣ Guardado del modelo**
Botón para exportar el modelo entrenado (`iris_rf_model.joblib`).


## Ejecución local

```bash
pip install -r requirements.txt
streamlit run proyect.py
```

## Vista previa simulada de la aplicación

### Página Principal
(Imagen simulada)
```
<img width="1366" height="695" alt="Página Principal" src="https://github.com/user-attachments/assets/31da91e8-5ed4-4258-971f-8c1e49b9ae4d" />

```

### Predicción de muestra nueva
(Imagen simulada)
```
<img width="1366" height="691" alt="Predicción de muestra nueva" src="https://github.com/user-attachments/assets/f4646ec7-a103-4868-9a8d-1e968b715e91" />

```

### Exploración de Datos
(Imagen simulada)
```
<img width="1366" height="695" alt="Exploración de Datos" src="https://github.com/user-attachments/assets/6546c1ad-7eaf-404c-ab48-aa302e21e72a" />

```

## Notas
- Dataset cargado desde scikit-learn.
- El modelo incluido (`iris_rf_model.joblib`) está entrenado con RandomForestClassifier.
