import requests
import websockets
from websockets.asyncio.client import connect
import asyncio
from pathlib import Path
import base64
import json

img1 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/sunset2.jpg')
img2 = Path('D:/StyleTransferAI/StyleTransferAI/test_images/impressionisme.jpg')


async def hello():
    async with connect("ws://localhost:8000/ws", max_size=5*1024*1024) as websocket:
        with open(img1, 'rb') as f :
            data1 = base64.b64encode(f.read()).decode("utf-8")
        with open(img2, 'rb') as f :
            data2 = base64.b64encode(f.read()).decode("utf-8")
        message = {
            "content" : data1,
            "style" : [
                data2
            ]
        }
        await websocket.send(json.dumps(message))
        new_data = await websocket.recv()
        with open(Path('C:/users/yoanb/Desktop/resp.jpg'), 'wb') as f :
            f.write(new_data)
        print("Data received !")

async def loop() :
    while True :
        i = input()
        if i == 'g' :
            await hello()
        elif i == 'exit' :
            quit()
        else :
            continue

if __name__ == "__main__":
    asyncio.run(loop())