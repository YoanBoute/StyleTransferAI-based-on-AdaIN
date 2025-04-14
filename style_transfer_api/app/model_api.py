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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
import asyncio
import psutil
import time
import numpy as np
import uuid

MLFLOW_TRACK_URI = Path("D:/Python/ML Flow/mlruns")
TMP_DIR = Path("./__tmp")

TMP_DIR.mkdir(exist_ok=True)

# TODO : 
# - Store generated images for each client for a certain amount of days
# - Handle multiple style files in a single request
# - Encapsulate file processing instructions in a function
# - Add a function to process and secure the received parameters (remove non existing parameters)

def switch_model_to_device(device, model) :
    keras.src.backend.common.global_state.set_global_attribute("torch_device", device)
    model = model.to(device)
    model.device = device
    return model

def generate_img(content, style, result_queue, **params) :
    model = AdaINModel.load_registered_model("StyleTransfer", "champion", MLFLOW_TRACK_URI)
    model = switch_model_to_device("cpu", model)
    stylized = model.generate(content, style, resize_size = 1500, **params).numpy()
    result_queue.put(stylized)

class RequestManager :
    """Class to manage the generation tasks, and cancel a request if a new request is emitted from the same client"""
    def __init__(self):
        self.tasks = {}
        self.processes = {}
    
    async def cancel_task(self, client_id) :
        if client_id in self.tasks :
            if client_id in self.processes :
                process = self.processes[client_id]
                try:
                    process.terminate() 
                    process.join()  # Wait for the process to clean up
                except psutil.NoSuchProcess:
                    pass
                del self.processes[client_id]
            self.tasks[client_id].cancel() 
            try :
                await self.tasks[client_id]
            except asyncio.CancelledError :
                pass            
            del self.tasks[client_id]

    async def start_task(self, client_id, content, style, params) :
        await self.cancel_task(client_id)
        res_queue = Queue()   
        process = Process(target=generate_img, args=(content, style, res_queue), kwargs={**params}) 
        process.start()
        self.processes[client_id] = process
        task = asyncio.create_task(self._monitor_task(process, res_queue, client_id))
        self.tasks[client_id] = task
        return task
    
    async def _monitor_task(self, process, result_queue, client_id) :
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, result_queue.get)
        process.join() # Make sure the process is cleaned up after it gave its result
        del self.processes[client_id]
        del self.tasks[client_id]
        return result
    

app = FastAPI()
manager = RequestManager()

@app.get("/")
def test() :
    return {"msg" : "Hello world"}

@app.websocket('/generate') 
async def endpoint(ws : WebSocket) :
    async def generation(client_id, request_id, content, style, params) :
        gen_path = TMP_DIR / ('gen_' + client_id + request_id + '.jpg')
        gen_task = await manager.start_task(client_id, content, style, params)
        try :
            gen_img = await gen_task
            keras.utils.save_img(gen_path, gen_img)
            with open(gen_path, 'rb') as f :
                gen_data = base64.b64encode(f.read()).decode('utf-8')
            response = {
                "client_id" : client_id,
                "status" : "success",
                "generated_image" : gen_data
            }
            content.unlink(missing_ok=True)
            style.unlink(missing_ok=True)
            gen_path.unlink(missing_ok=True)
            await ws.send_text(json.dumps(response))
        except asyncio.exceptions.CancelledError :
            response = {
                "client_id" : client_id,
                "status" : "cancelled"
            }
            await ws.send_text(json.dumps(response))
        except Exception as e :
            response = {
                "client_id" : client_id,
                "status" : "error",
                "message" : str(e)
            }
            await ws.send_text(json.dumps(response))
        finally :    
            content.unlink(missing_ok=True)
            style.unlink(missing_ok=True)
            gen_path.unlink(missing_ok=True)
        
    await ws.accept()
    while True:
        client_id = "unknown"
        try :
            data = await ws.receive_text()
            request_id = uuid.uuid4().hex # The request_id is used to differentiate the requests made by a same client in a short time (especially for file handling)
            message = json.loads(data)
            client_id = message["client_id"]
            content = base64.b64decode(message["content"])
            content_path = TMP_DIR / ('content_' + client_id + request_id + '.jpg')
            # TODO : Retrieve file true extension to use it for saving the file
            with open(content_path, 'wb') as f :
                f.write(content)
            style = base64.b64decode(message["style"][0])
            style_path = TMP_DIR / ('style_' + client_id + request_id + '.jpg')
            with open(style_path, 'wb') as f :
                f.write(style)
            gen_params = message["params"]
            asyncio.create_task(generation(client_id, request_id, content=content_path, style=style_path, params=gen_params))            
        except Exception as e :
            response = {
                "client_id" : client_id,
                "status" : "error",
                "message" : str(e)
            }
            await ws.send_text(json.dumps(response))
