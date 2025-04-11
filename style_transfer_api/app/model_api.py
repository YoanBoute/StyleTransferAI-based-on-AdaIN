import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from ...style_transfer_model.adain_model import AdaINModel 
import fastapi
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import FileResponse
import base64
import json

MLFLOW_TRACK_URI = Path("D:/Python/ML Flow/mlruns")
TMP_DIR = Path("./__tmp")

TMP_DIR.mkdir(exist_ok=True)

model = AdaINModel.load_registered_model("StyleTransfer", "champion", MLFLOW_TRACK_URI)

app = FastAPI()

@app.get("/")
def test() :
    return {"msg" : "Hello world"}

@app.post("/generate")
def generate(img) :
    return {"Euuhhhh" : "Alors oui"}

@app.post('/retest') 
def retest(file : UploadFile) :
    return FileResponse(Path('D:/StyleTransferAI/StyleTransferAI/test_images/lion.jpg'))

@app.websocket('/ws')
async def endpoint(ws : WebSocket) :
    await ws.accept()
    while True:
        data = await ws.receive_text()
        message = json.loads(data)
        content = base64.b64decode(message["content"])
        content_path = TMP_DIR / 'content.jpg'
        with open(content_path, 'wb') as f :
            f.write(content)
        style = base64.b64decode(message["style"][0])
        style_path = TMP_DIR / 'style.jpg'
        with open(style_path, 'wb') as f :
            f.write(style)
        try :
            stylized_img = model.generate(content_path, style_path)
        except Exception as e :
            print(e)
        gen_path = TMP_DIR / "gen.jpg"
        keras.utils.save_img(gen_path, stylized_img)
        with open(gen_path, 'rb') as f :
            gen_data = f.read()
        await ws.send_bytes(gen_data)
        content_path.unlink()
        style_path.unlink()
        gen_path.unlink()

