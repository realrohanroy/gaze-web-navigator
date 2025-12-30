import asyncio
import websockets
import json
import random

clients = set()

async def handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.remove(websocket)

async def fake_sender():
    while True:
        if clients:
            x = random.randint(100, 800)
            y = random.randint(100, 500)
            msg = json.dumps({"x": x, "y": y})
            await asyncio.gather(*[c.send(msg) for c in clients])
        await asyncio.sleep(0.1)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await fake_sender()

asyncio.run(main())
