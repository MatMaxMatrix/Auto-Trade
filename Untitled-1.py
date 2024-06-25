#%%
import asyncio

async def main():
        print('Hello ...')
        await asyncio.sleep(1)
        print('... World!')

# Python 3.7+
asyncio.run(main())
#%%

import asyncio
import websockets

async def hello():
    uri = "wss://example.com/websocket"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello world!")
        response = await websocket.recv()
        print(response)

# Get the existing event loop
loop = asyncio.get_event_loop()

# Run the hello coroutine
loop.run_until_complete(hello())
# %%
