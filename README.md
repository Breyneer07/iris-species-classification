## Integrante del proyecto
- Breyneer Nieto Cardeño

# Iris Species Classification — Proyecto

Este repositorio contiene la aplicación Streamlit para clasificar especies de Iris, así como los archivos del proyecto, modelo entrenado y guion del video.

## Contenido
- `proyect.py` — Código completo de la app Streamlit.
- `requirements.txt` — Dependencias del proyecto.
- `iris_rf_model.joblib` — Modelo Random Forest entrenado listo para usar.
- `video_presentation_link.txt` — (https://youtu.be/YBdfUI9EfWc)
- `IRIS SPECIES CLASSIFICATION PROYECT.pdf` — Documento base del proyecto.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run proyect.py
```

## Vista previa simulada de la aplicación

### Página Principal
(Imagen simulada)
```
--------------------------------------------------
| Iris Species Classification — Proyecto         |
| Accuracy: 0.97  Precision: 0.96 ...            |
--------------------------------------------------
```

### Predicción de muestra nueva
(Imagen simulada)
```
--------------------------------------------------
| Predicción: Iris-virginica                     |
| Probabilidades: ...                             |
--------------------------------------------------
```

### Exploración de Datos
(Imagen simulada)
```
--------------------------------------------------
| Histogramas, boxplots, matriz de dispersión... |
--------------------------------------------------
```

## Notas
- Dataset cargado desde scikit-learn.
- El modelo incluido (`iris_rf_model.joblib`) está entrenado con RandomForestClassifier.
