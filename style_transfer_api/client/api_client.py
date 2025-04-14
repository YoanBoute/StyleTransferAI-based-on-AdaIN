import requests
import websockets
from websockets.asyncio.client import connect
import asyncio
from pathlib import Path
import base64
import json
import uuid

img1 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/wolf_forest.jpg')
img2 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/impressionisme.jpg')
img3 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/oil_paint.jpg')

# TODO : 
# - Add an authentification method (token / connection / ...)
# - Encapsulate file data in a class / dict (With extension, size, ...)
# - Reconnect to the Websockets every once in a while

async def generate() :
    client_id = uuid.uuid4().hex
    async with connect("ws://localhost:8000/generate", max_size=5*1024*1024) as websocket : # TODO : Change port number
        with open(img1, 'rb') as f :
            data1 = base64.b64encode(f.read()).decode("utf-8")
        with open(img2, 'rb') as f :
            data2 = base64.b64encode(f.read()).decode("utf-8")
        with open(img3, 'rb') as f :
            data3 = base64.b64encode(f.read()).decode("utf-8")
        message1 = { # TODO : Add client ID
            "client_id" : client_id,
            "content" : data1,
            "style" : [
                data2
            ],
            "params" : {
                'alpha' : 0.5,
                'preserve_colors' : True
                }
        }
        message2 = { 
            "client_id" : client_id,
            "content" : data1,
            "style" : [
                data3
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
        # response = await websocket.recv()
        while True:
            try:
                response = await websocket.recv()
                msg = json.loads(response)
                print("Received:", msg["status"])
                if msg.get("status") in ("success"):
                    with open(Path('C:/users/yoanb/Desktop/resp.jpg'), 'wb') as f :
                        f.write(base64.b64decode(msg["generated_image"]))
            except websockets.ConnectionClosed:
                print("WebSocket closed.")
                break

async def loop() :
    while True :
        i = input()
        if i == 'g' :
            await generate()
        elif i == 'exit' :
            quit()
        else :
            continue

if __name__ == "__main__":
    asyncio.run(generate())