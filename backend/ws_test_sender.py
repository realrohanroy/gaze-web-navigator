import asyncio
import random
from websocket_server import send_gaze

async def test():
    while True:
        x = random.randint(100, 800)
        y = random.randint(100, 500)
        await send_gaze(x, y)
        await asyncio.sleep(0.1)
        print(x, y)

asyncio.run(test())
