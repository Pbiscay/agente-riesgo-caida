
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Título
st.title("Agente de Riesgo de Caída en Adultos Mayores")

# Ingreso de datos por el usuario
edad = st.number_input("Edad", min_value=60, max_value=100, value=75)
diagnostico = st.selectbox("Diagnóstico de base", ["Vestibular", "Traumatológico", "Neurológico"])
TUG = st.number_input("Tiempo TUG (segundos)", min_value=0.0, max_value=60.0, value=18.0)
ABC = st.slider("Puntaje ABC (0-100)", 0, 100, 45)
DHI = st.slider("Puntaje DHI (0-100)", 0, 100, 65)

# Mapeo de diagnóstico a número
diagnostico_map = {"Vestibular": 0, "Traumatológico": 1, "Neurológico": 2}
dx = diagnostico_map[diagnostico]

# Generar datos simulados para entrenar el modelo
import random

def generar_paciente():
    edad = random.randint(60, 85)
    dx = random.choice([0, 1, 2])
    tug = random.uniform(8, 25)
    abc = random.randint(30, 95)
    dhi = random.randint(0, 90)

    if tug > 18 or abc < 50 or dhi > 60:
        riesgo = "Alto"
    elif 15 < tug <= 18 or 50 <= abc <= 70 or 40 < dhi <= 60:
        riesgo = "Medio"
    else:
        riesgo = "Bajo"

    return [edad, dx, tug, abc, dhi, riesgo]

data = [generar_paciente() for _ in range(50)]
columns = ["edad", "diagnostico", "TUG", "ABC", "DHI", "riesgo"]
df = pd.DataFrame(data, columns=columns)

X = df[["edad", "diagnostico", "TUG", "ABC", "DHI"]]
y = df["riesgo"]

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X, y)

# Cuando se aprieta el botón
if st.button("Calcular Riesgo de Caída"):
    paciente = pd.DataFrame([[edad, dx, TUG, ABC, DHI]], columns=X.columns)
    prediccion = modelo.predict(paciente)[0]

    st.subheader("🔮 Riesgo estimado de caída:")
    st.success(prediccion)
