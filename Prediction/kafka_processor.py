#%%
import json
from confluent_kafka import Consumer, Producer, KafkaError
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from collections import deque
import subprocess
from dotenv import load_dotenv
from os import getenv

load_dotenv()

data_queue = deque(maxlen=20)

# InfluxDB setup
influx_client = InfluxDBClient(url=getenv("INFLUXDB_URL"),
    username=getenv("INFLUXDB_USERNAME"),
    password=getenv("INFLUXDB_PASSWORD"),
    ssl=True, verify_ssl=True,
    org=getenv("INFLUXDB_ORG"))
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Kafka Consumer setup
consumer_conf = {
    'bootstrap.servers': getenv("KAFKA_BOOSTRAP_SERVERS"),
    'group.id': 'binance_processor',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_conf)
consumer.subscribe(['binance_data'])

# Kafka Producer setup (for processed data)
producer_conf = {
    'bootstrap.servers': getenv("KAFKA_BOOSTRAP_SERVERS")
}
producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def process_message(msg):
    
    try:
        data = json.loads(msg.value())
        
        # Extract relevant information
        event_type = data['e']
        event_time = data['E']
        symbol = data['s']
        kline = data['k']
        kline_start_time = kline['t']
        kline_close_time = kline['T']
        interval = kline['i']
        first_trade_id = kline['f']
        last_trade_id = kline['L']
        open_price = float(kline['o'])
        close_price = float(kline['c'])
        high_price = float(kline['h'])
        low_price = float(kline['l'])
        base_asset_volume = float(kline['v'])
        number_of_trades = kline['n']
        is_kline_closed = kline['x']
        quote_asset_volume = float(kline['q'])
        taker_buy_base_asset_volume = float(kline['V'])
        taker_buy_quote_asset_volume = float(kline['Q'])
        ignore = kline['B']

        # Store in InfluxDB
        point = Point("binance_data") \
            .tag("symbol", symbol) \
            .field("event_type", event_type) \
            .field("event_time", event_time) \
            .field("kline_start_time", kline_start_time) \
            .field("kline_close_time", kline_close_time) \
            .field("interval", interval) \
            .field("first_trade_id", first_trade_id) \
            .field("last_trade_id", last_trade_id) \
            .field("open_price", open_price) \
            .field("close_price", close_price) \
            .field("high_price", high_price) \
            .field("low_price", low_price) \
            .field("base_asset_volume", base_asset_volume) \
            .field("number_of_trades", number_of_trades) \
            .field("is_kline_closed", is_kline_closed) \
            .field("quote_asset_volume", quote_asset_volume) \
            .field("taker_buy_base_asset_volume", taker_buy_base_asset_volume) \
            .field("taker_buy_quote_asset_volume", taker_buy_quote_asset_volume) \
            .field("ignore", ignore) \
            .time(event_time, "ms")
        try:
            write_api.write(bucket="mybucket", record=point)
            write_api.flush()
        except Exception as e:
            print(f"Error writing to InfluxDB: {e}")

        # Process and produce to another Kafka topic for the LLM model
        """
        processed_data = {
            'event_type': event_type,
            'event_time': event_time,
            'symbol': symbol,
            'kline_start_time': kline_start_time,
            'kline_close_time': kline_close_time,
            'interval': interval,
            'first_trade_id': first_trade_id,
            'last_trade_id': last_trade_id,
            'open_price': open_price,
            'close_price': close_price,
            'high_price': high_price,
            'low_price': low_price,
            'base_asset_volume': base_asset_volume,
            'number_of_trades': number_of_trades,
            'is_kline_closed': is_kline_closed,
            'quote_asset_volume': quote_asset_volume,
            'taker_buy_base_asset_volume': taker_buy_base_asset_volume,
            'taker_buy_quote_asset_volume': taker_buy_quote_asset_volume,
            'ignore': ignore
        }
        producer.produce('processed_data', json.dumps(processed_data), callback=delivery_report)
        data_queue.append(processed_data)
        if len(data_queue) == 20:
            batch_data = list(data_queue)
            print(f"Processed and stored data: {batch_data}")
            with open("batch_data.json", "w") as f:
                json.dump(batch_data, f)
            subprocess.run(['python3', 'git_push.py'])
            data_queue.clear()
        """
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Error accessing data key: {e}")
    except Exception as e:
        print(f"Unexpected error processing message: {e}")

# Main loop
def main():
    try:
        while True:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Error: {msg.error()}")
                    break
            process_message(msg)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        producer.flush()  # Make sure all messages are sent
        consumer.close()
        influx_client.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
# %%
