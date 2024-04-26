import cv2
from fastapi import FastAPI, File, UploadFile
import pickle
import numpy as np

app = FastAPI()

print("Inicia Carga del Modelo")
with open("modelo2.pkl", "rb") as file:
    modelo = pickle.load(file)
print("Modelo Cargado")

def extraer_histograma(image_bytes):
  nparr = np.frombuffer(image_bytes,np.uint8)
  image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
  image = cv2.resize(image,(500,500))
  hist  = cv2.calcHist([image],[1,2],None,[256,256],[0,256,0,256])
  return hist

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    train_labels = ['maduros', 'verdes']
    print("Recibiendo Imagen")
    imagen = await file.read()
    hist = extraer_histograma(imagen)
    predict = modelo.predict(hist.reshape(1,-1))
    label_final = train_labels[predict[0]]
    print("Imagen Recibida")
    return label_final