# proyect.py
# App Streamlit para clasificación de especies Iris


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title='Iris Species Classifier', layout='wide')

@st.cache_data
def load_data():
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df['target'] = data.target
    df['species'] = df['target'].map({i: s for i, s in enumerate(data.target_names)})
    return df, data

@st.cache_data
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

# --- Load data
df, iris = load_data()
FEATURES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# --- Sidebar: training options & input sample
st.sidebar.header('Configuración del modelo')
n_estimators = st.sidebar.slider('Árboles (n_estimators)', 10, 300, 100)
random_state = st.sidebar.number_input('Random state', value=42, step=1)
retrain = st.sidebar.button('Reentrenar modelo')

st.sidebar.markdown('---')
st.sidebar.header('Entrada manual (nueva muestra)')
input_vals = {
    'sepal length (cm)': st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean())),
    'sepal width (cm)': st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean())),
    'petal length (cm)': st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean())),
    'petal width (cm)': st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean())),
}

# --- Main UI
st.title('Iris Species Classification — Proyecto')
st.markdown('**Descripción:** modelo de clasificación para predecir la especie de Iris a partir de medidas de sépalo y pétalo.')

# Split data and (re)train model
X = df[FEATURES]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)

if 'model' not in st.session_state or retrain:
    with st.spinner('Entrenando modelo...'):
        model = train_model(X_train, y_train, n_estimators=n_estimators, random_state=random_state)
        st.session_state['model'] = model
else:
    model = st.session_state['model']

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Accuracy', f'{acc:.3f}')
col2.metric('Precision (weighted)', f'{prec:.3f}')
col3.metric('Recall (weighted)', f'{rec:.3f}')
col4.metric('F1 (weighted)', f'{f1:.3f}')

st.markdown('---')

# Classification report
with st.expander('Reporte de clasificación (test set)'):
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- Prediction for user input
st.subheader('Probar una nueva muestra')
if st.button('Predecir especie para la muestra ingresada'):
    sample = np.array([input_vals[ft] for ft in FEATURES]).reshape(1, -1)
    pred = model.predict(sample)[0]
    pred_proba = model.predict_proba(sample)[0]
    st.success(f'Predicción: **{iris.target_names[pred]}**')
    st.write('Probabilidades:')
    proba_df = pd.DataFrame({'species': iris.target_names, 'probability': pred_proba})
    st.table(proba_df)

    # 3D scatter with new point
    fig = px.scatter_3d(df, x='petal length (cm)', y='petal width (cm)', z='sepal length (cm)', color='species', symbol='species', size_max=6)
    fig.add_trace(go.Scatter3d(x=[sample[0,2]], y=[sample[0,3]], z=[sample[0,0]], mode='markers', marker=dict(size=8, symbol='x', line=dict(width=2), opacity=0.9), name='Nueva muestra'))
    st.plotly_chart(fig, use_container_width=True)

# --- Data exploration
st.markdown('---')
st.subheader('Exploración de datos')

st.markdown('**Histograma de características**')
feature = st.selectbox('Selecciona característica', FEATURES)
fig_hist = px.histogram(df, x=feature, color='species', marginal='box')
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('**Matriz de dispersión**')
fig_pair = px.scatter_matrix(df, dimensions=FEATURES, color='species', symbol='species')
fig_pair.update_traces(diagonal_visible=False)
st.plotly_chart(fig_pair, use_container_width=True)

# --- Save model button
if st.button('Guardar modelo (joblib)'):
    joblib.dump(model, 'iris_rf_model.joblib')
    st.success('Modelo guardado como iris_rf_model.joblib')

st.markdown("#### Integrante del proyecto: **Breyneer Nieto Cardeño**")

st.markdown('---')
st.caption('Dataset: UCI Iris. App generada para el proyecto de Data Mining — Universidad de la Costa')