import requests
import websockets
from websockets.asyncio.client import connect
import asyncio
from pathlib import Path
import base64
import json
import uuid
# from style_transfer_api.utils.file import File
from ..utils.file import File
from ..utils.requests import Request, Response

img1 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/wolf_forest.jpg')
img2 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/impressionisme.jpg')
img3 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/oil_paint.jpg')

async def request_generation(connection, client_id, content_img, style_imgs, params) :
    request = Request(client_id=client_id, content_img=content_img, style_imgs=style_imgs, params=params)
    await connection.send(request.model_dump())
    response = await Response(**connection.recv())
    return Response

async def generate() :
    client_id = uuid.uuid4().hex
    async with connect("ws://localhost:8000/generate", max_size=5*1024*1024) as websocket : # TODO : Change port number
        data1 = File.from_path(img1)
        data2 = File.from_path(img2)
        data3 = File.from_path(img3)
        message1 = { 
            "client_id" : client_id,
            "content" : data1.model_dump(),
            "style" : [
                data2.model_dump()
            ],
            "params" : {
                'alpha' : 0.5,
                'preserve_colors' : True
                }
        }
        message2 = { 
            "client_id" : client_id,
            "content" : data1.model_dump(),
            "style" : [
                data2.model_dump(),
                data3.model_dump()
            ],
            "params" : {
                'alpha' : 2,
                'preserve_colors' : False
                }
        }
        print("Send first request")
        await websocket.send(json.dumps(message1))
        await asyncio.sleep(5)
        print("Send second request")
        await websocket.send(json.dumps(message2))
        while True:
            try:
                response = await websocket.recv()
                msg = json.loads(response)
                print("Received:", msg["status"])
                if msg.get("status") in ("success"):
                    gen_file = File.from_dict(msg["generated_image"])
                    gen_file.save_to(Path('C:/users/yoanb/Desktop/resp'))
            except websockets.ConnectionClosed:
                print("WebSocket closed.")
                break

if __name__ == "__main__":
    asyncio.run(generate())