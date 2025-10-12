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
import uuid
from ..utils.file import File
from ..utils.requests import Request, RequestType, Response, GenerationStatus

MLFLOW_TRACK_URI = Path("D:/Python/ML Flow/mlruns")
TMP_DIR = Path("./__tmp")

TMP_DIR.mkdir(exist_ok=True)

# TODO : 
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
        self.files = {}
    
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
            if self.files.get(client_id) is not None :
                for file in self.files[client_id] :
                    file.unlink(missing_ok=True) 
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
        self.files[client_id] = [content, *styles]
        task = asyncio.create_task(self._monitor_task(process, res_queue, client_id))
        self.tasks[client_id] = task
        return task
    
    async def _monitor_task(self, process, result_queue, client_id) :
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, result_queue.get)
        process.join() # Make sure the process is cleaned up after it gave its result
        del self.processes[client_id]
        del self.tasks[client_id]
        del self.files[client_id]
        return result
    

app = FastAPI()
manager = RequestManager()

@app.websocket('/generate') 
async def endpoint(ws : WebSocket) :
    async def generation(client_id, request_id, content, styles, params) :
        gen_path = TMP_DIR / ('gen_' + client_id + request_id + '.jpg')
        gen_task = await manager.start_task(client_id, content, styles, params)
        try :
            gen_img = await gen_task
            keras.utils.save_img(gen_path, gen_img)
            gen_file = File.from_path(gen_path)
            response = Response(request_id=request_id, status=GenerationStatus.success, message="Image generated successfully", generated_image=gen_file) 
        except asyncio.exceptions.CancelledError :
            response = Response(request_id=request_id, status=GenerationStatus.cancel, message="Generation was cancelled")
        except Exception as e :
            response = Response(request_id=request_id, status=GenerationStatus.error, message=str(e))
            print(e)
        finally :        
            content.unlink(missing_ok=True)
            for style in styles :
                style.unlink(missing_ok=True)
            gen_path.unlink(missing_ok=True)
            await ws.send_text(json.dumps(response.model_dump()))
    
    async def cancel_request(client_id) :
        await manager.cancel_task(client_id)

    await ws.accept()
    client_id = uuid.uuid4().hex
    while True:
        # client_id = "unknown"
        try :
            req = await ws.receive_text()
            req_dict = json.loads(req)
            request = Request.from_dict(req_dict)
            # client_id = request.client_id
            request_id = request.request_id
            if request.type == RequestType.gen :
                content = request.content_img
                content_path = TMP_DIR / ('content_' + client_id + request_id)
                content_path = content.save_to(content_path)
                styles = [style_file for style_file in request.style_imgs]
                style_paths = [TMP_DIR / ('style_' + client_id + request_id + f'_{i}') for i in range(len(styles))]
                style_paths = [style.save_to(style_paths[i]) for i, style in enumerate(styles)]
                gen_params = request.params
                asyncio.create_task(generation(client_id, request_id, content=content_path, styles=style_paths, params=gen_params))  
            elif request.type == RequestType.cancel :
                asyncio.create_task(cancel_request(client_id))        
        except fastapi.WebSocketDisconnect :
            print(ws.client)
            break
        except Exception as e :
            response = Response(request_id=request_id, status=GenerationStatus.error, message=str(e))
            print(e)
            await ws.send_text(json.dumps(response.model_dump()))
            if 'content_path' in locals() :       
                content_path.unlink(missing_ok=True)
            if 'style_paths' in locals() : 
                for style in style_paths :
                    style.unlink(missing_ok=True)