#Para cargar el modelo: 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)
# Verifica el tipo del modelo
print(f"Modelo cargado: {type(modelo)}")

app = app = FastAPI(title="API de Predicciones",
                    description="Modelo Random Forest para predecir la calidad del vino blanco",
                    version="1.0")

class DatosVino(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
@app.get("/")
def leer_root():
    return {"mensaje": "🚀 API de predicciones funcionando"}

# Ruta para hacer predicciones
@app.post("/calidad_vino")
def predecir(datos: DatosVino):
     # Convertir los datos recibidos en un array 2D
    x_input = np.array([
            [datos.fixed_acidity, datos.volatile_acidity, datos.citric_acid,
             datos.residual_sugar, datos.chlorides, datos.free_sulfur_dioxide,
             datos.total_sulfur_dioxide, datos.density, datos.pH, datos.sulphates,
             datos.alcohol]
        ])

    pred = modelo.predict(x_input)[0]  # Realiza la predicción y toma el primer valor
    return {"calidad_predicha": pred}