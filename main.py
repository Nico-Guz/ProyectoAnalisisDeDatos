from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
import base64
from fastapi.responses import HTMLResponse

import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Configuración para cargar las plantillas HTML desde el directorio "templates"
templates = Jinja2Templates(directory="templates")

# Cargar el modelo desde el archivo Joblib
modelo = joblib.load("modelo_random_forest.joblib")

# Mock para la variable de la imagen
file_contents = b'Tu_contenido_de_imagen_en_bytes'

@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit_form/")
async def submit_form(request: Request, title: str = Form(...), genre: str = Form(...), release_date: str = Form(...), 
                      vote_average: int = Form(...), vote_count: int = Form(...),file_contents: UploadFile = File(...)):

    # Utilizar el modelo para hacer una predicción
    data = np.array([[vote_average, vote_count]])
    prediction = modelo.predict(data)

    # Interpretar la predicción
    prediction_message = "Buena película" if prediction[0] == 1 else "Mala película"

    
    # Obtener los contenidos de la imagen como bytes
    contents = await file_contents.read()

    # Codificar en base64 los contenidos de la imagen
    encoded_image = base64.b64encode(contents).decode('utf-8')
    
    
    return templates.TemplateResponse(
        "response.html",
        {"request": request,"title": title, "genre": genre, "release_date": release_date, "vote_average": vote_average, 
         "vote_count": vote_count, "prediction_message": prediction_message, "file_contents": encoded_image}
    )

@app.post("/submit_csv/")
async def submit_csv(request: HTMLResponse, csv_file: UploadFile = File(...)):
    # Leer el archivo CSV y procesar los datos
    df = pd.read_csv(csv_file.file)
    
    # Hacer predicciones utilizando el modelo
    data_for_predictions = df[['vote_average', 'vote_count']]
    predictions = modelo.predict(data_for_predictions)
    
    # Mapear las predicciones a "Buena película" o "Mala película"
    df['Predicción'] = ["Buena película" if p == 1 else "Mala película" for p in predictions]

    # Convertir el DataFrame a HTML y pasarlo a la plantilla
    html_table = df.to_html(index=False)
    
    return HTMLResponse(content=templates.get_template("response_with_csv.html").render(html_table=html_table), status_code=200)