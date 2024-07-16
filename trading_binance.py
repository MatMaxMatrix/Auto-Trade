from binance.client import Client
from dotenv import load_dotenv
from os import getenv

def buy(symbol, quantity):
    client = get_binance_client()
    print("Account balance:")
    print(client.get_asset_balance(asset='BTC'))
    print(client.get_asset_balance(asset='USDT'))
    print(f"Buying {quantity} of {symbol}:")
    market_order = client.order_market_buy(symbol=symbol, quantity=quantity)
    print("Order result:")
    print(market_order)
    print("New account balance:")
    print(client.get_asset_balance(asset='BTC'))
    print(client.get_asset_balance(asset='USDT'))

def get_binance_client():
    load_dotenv()
    api_key = getenv("BINANCE_API_KEY")
    api_secret = getenv("BINANCE_SECRET_KEY")

    client = Client(api_key, api_secret)
    client.API_URL = getenv("BINANCE_API_URL")

    return client

buy("BTCUSDT", 0.0001)