import json
import websocket
from kafka import KafkaProducer
from dotenv import load_dotenv
from os import getenv

def on_message(ws, message):
    #parse the message
    data = json.loads(message)
    #producde the message to kafka
    producer.send('binance_data', value=data)
    print(f"send to kafka: {data}")


def on_error(ws, error):
    print(f"ERROR:{error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection opened")
       # Subscribe to the Binance WebSocket stream
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": [
            "btcusdt@kline_1s"
        ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

if __name__ == "__main__":
    load_dotenv()
    producer = KafkaProducer(bootstrap_servers=getenv("KAFKA_BOOSTRAP_SERVERS"), value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    ws = websocket.WebSocketApp(getenv("BINANCE_WEBSOCKET_URL"),
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open = on_open
                                )
    ws.run_forever()
