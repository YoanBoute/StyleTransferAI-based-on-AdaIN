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

MLFLOW_TRACK_URI = Path("D:/Python/ML Flow/mlruns")
TMP_DIR = Path("./__tmp")

TMP_DIR.mkdir(exist_ok=True)

def switch_model_to_device(device, model) :
    keras.src.backend.common.global_state.set_global_attribute("torch_device", device)
    model = model.to(device)
    model.device = device
    return model

def generate_img(content, style, result_queue, **params) :
    model = AdaINModel.load_registered_model("StyleTransfer", "champion", MLFLOW_TRACK_URI)
    model = switch_model_to_device("cpu", model)
    stylized = model.generate(content, style, resize_size = 500, **params).numpy()
    result_queue.put(stylized)

class RequestManager :
    """Class to manage the generation tasks, and cancel a request if a new request is emitted from the same client"""
    def __init__(self):
        self.tasks = {}
        self.processes = {}
        self.executor = ProcessPoolExecutor(max_workers=1)
    
    async def cancel_task(self, request_id) :
        if request_id in self.tasks :
            if request_id in self.processes :
                process = self.processes[request_id]
                try:
                    process.terminate() 
                    process.join()  # Wait for the process to clean up
                except psutil.NoSuchProcess:
                    pass
                del self.processes[request_id]
            self.tasks[request_id].cancel() 
            try :
                await self.tasks[request_id]
            except asyncio.CancelledError :
                pass            
            del self.tasks[request_id]

    async def start_task(self, request_id, content, style) :
        await self.cancel_task(request_id)
        res_queue = Queue()   
        process = Process(target=generate_img, args=(content, style, res_queue)) 
        process.start()
        self.processes[request_id] = process
        task = asyncio.create_task(self._monitor_task(process, res_queue, request_id))
        self.tasks[request_id] = task
        return task
    
    async def _monitor_task(self, process, result_queue, request_id) :
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, result_queue.get)
        process.join() # Make sure the process is cleaned up after it gave its result
        del self.processes[request_id]
        del self.tasks[request_id]
        return result
    

app = FastAPI()
manager = RequestManager()

@app.get("/")
def test() :
    return {"msg" : "Hello world"}

@app.websocket('/generate') # TODO : Use POST instead of WebSocket to transfer files (Maybe with AJAX)
async def endpoint(ws : WebSocket) :
    async def generation(request_id, content, style) :
        gen_path = TMP_DIR / ('gen_' + str(np.random.randint(1e6)) + '.jpg')
        gen_task = await manager.start_task(request_id, content, style)
        try :
            gen_img = await gen_task
            keras.utils.save_img(gen_path, gen_img)
            with open(gen_path, 'rb') as f :
                gen_data = base64.b64encode(f.read()).decode('utf-8')
            response = {
                "request_id" : request_id,
                "status" : "success",
                "generated_image" : gen_data
            }
            content.unlink(missing_ok=True)
            style.unlink(missing_ok=True)
            gen_path.unlink(missing_ok=True)
            await ws.send_text(json.dumps(response))
        except asyncio.exceptions.CancelledError :
            response = {
                "request_id" : request_id,
                "status" : "cancelled"
            }
            await ws.send_text(json.dumps(response))
        except Exception as e :
            response = {
                "request_id" : request_id,
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
        request_id = "unknown"
        try :
            data = await ws.receive_text()
            message = json.loads(data)
            request_id = message["request_id"]
            content = base64.b64decode(message["content"])
            content_path = TMP_DIR / ('content_' + str(np.random.randint(1e6)) + '.jpg') # TODO : use UUID as basis for file names
            # TODO : Retrieve file true extension to use it for saving the file
            with open(content_path, 'wb') as f :
                f.write(content)
            style = base64.b64decode(message["style"][0])
            style_path = TMP_DIR / ('style_' + str(np.random.randint(1e6)) + '.jpg')
            with open(style_path, 'wb') as f :
                f.write(style)
            asyncio.create_task(generation(request_id, content=content_path, style=style_path))            
        except Exception as e :
            response = {
                "request_id" : request_id,
                "status" : "error",
                "message" : str(e)
            }
            await ws.send_text(json.dumps(response))
