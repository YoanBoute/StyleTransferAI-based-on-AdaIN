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
from fastapi import FastAPI, WebSocket
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
from ..utils.file import File

MLFLOW_TRACK_URI = Path("D:/Python/ML Flow/mlruns")
TMP_DIR = Path("./__tmp")

TMP_DIR.mkdir(exist_ok=True)

# TODO : 
# - Store generated images for each client for a certain amount of days
# - Add a function to process and secure the received parameters (remove non existing parameters)
# - Cancel task if connection with client is lost / closed

def switch_model_to_device(device, model) :
    keras.src.backend.common.global_state.set_global_attribute("torch_device", device)
    model = model.to(device)
    model.device = device
    return model

def generate_img(content, styles, result_queue, **params) :
    model = AdaINModel.load_registered_model("StyleTransfer", "champion", MLFLOW_TRACK_URI)
    model = switch_model_to_device("cpu", model)
    stylized = model.generate(content, styles, **params).numpy()
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

    async def start_task(self, client_id, content, styles, params) :
        await self.cancel_task(client_id)
        res_queue = Queue()   
        process = Process(target=generate_img, args=(content, styles, res_queue), kwargs={**params}) 
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
    async def generation(client_id, request_id, content, styles, params) :
        gen_path = TMP_DIR / ('gen_' + client_id + request_id + '.jpg')
        gen_task = await manager.start_task(client_id, content, styles, params)
        try :
            gen_img = await gen_task
            keras.utils.save_img(gen_path, gen_img)
            gen_file = File.from_path(gen_path)
            response = {
                "client_id" : client_id,
                "status" : "success",
                "generated_image" : gen_file.model_dump()
            }
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
            print(e)
            await ws.send_text(json.dumps(response))
        finally :        
            content.unlink(missing_ok=True)
            for style in styles :
                style.unlink(missing_ok=True)
            gen_path.unlink(missing_ok=True)
        
    await ws.accept()
    while True:
        client_id = "unknown"
        try :
            data = await ws.receive_text()
            message = json.loads(data)
            client_id = message["client_id"]
            request_id = uuid.uuid4().hex # The request_id is used to differentiate the requests made by a same client in a short time (especially for file handling)
            content = File.from_dict(message["content"])
            content_path = TMP_DIR / ('content_' + client_id + request_id)
            content_path = content.save_to(content_path)
            # style = File.from_dict(message["style"][0])
            # style_path = TMP_DIR / ('style_' + client_id + request_id)
            # style_path = style.save_to(style_path)
            styles = [File.from_dict(style_file) for style_file in message["style"]]
            style_paths = [TMP_DIR / ('style_' + client_id + request_id + f'_{i}') for i in range(len(styles))]
            style_paths = [style.save_to(style_paths[i]) for i, style in enumerate(styles)]
            gen_params = message["params"]
            asyncio.create_task(generation(client_id, request_id, content=content_path, styles=style_paths, params=gen_params))            
        except Exception as e :
            response = {
                "client_id" : client_id,
                "status" : "error",
                "message" : str(e)
            }
            print(e)
            await ws.send_text(json.dumps(response))
            if 'content_path' in locals() :       
                content_path.unlink(missing_ok=True)
            if 'style_paths' in locals() : 
                for style in style_paths :
                    style.unlink(missing_ok=True)