import asyncio
import websockets
import json
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092') # Connect to Kafka on localhost:9092

async def binance_websocket(): # Define an asynchronous function
    uri = "wss://stream.binance.com:443/ws/btcusdt@kline_5m" # Define the URI for the websocket
    async with websockets.connect(uri) as websocket: # Connect to the websocket
        while True:   # Loop forever
            message = await websocket.recv() # Wait for a message from the websocket
            data = json.loads(message) # Parse the message as JSON
            producer.send('binance_klines', value=message.encode('utf-8')) # Send the message to the Kafka topic

asyncio.get_event_loop().run_until_complete(binance_websocket()) # Run the asynchronous function

# %%
