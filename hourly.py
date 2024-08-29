import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

def get_bitcoin_hourly_data(days=3650):
    """
    Fetch hourly Bitcoin data for the specified number of days.
    
    :param days: Number of days of historical data to fetch (default: 3650)
    :return: DataFrame with hourly price data
    """
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (days * 24 * 60 * 60)
    
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    
    all_data = []
    current_time = end_time
    
    while current_time > start_time:
        params = {
            "fsym": "BTC",
            "tsym": "USD",
            "limit": 2000,  # Max limit per request
            "toTs": current_time
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['Response'] == "Error":
            logging.error(f"API Error: {data['Message']}")
            return None
        
        hourly_data = data['Data']['Data']
        all_data = hourly_data + all_data
        
        if len(hourly_data) < 2000:
            break
        
        current_time = hourly_data[0]['time'] - 3600  # Move to the previous hour
    
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def plot_bitcoin_price(df):
    """
    Plot the Bitcoin price data.
    
    :param df: DataFrame with Bitcoin price data
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f'Bitcoin Hourly Closing Price\n({df.index[0].date()} to {df.index[-1].date()})')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Fetch the data
    bitcoin_data = get_bitcoin_hourly_data()
    
    if bitcoin_data is not None and not bitcoin_data.empty:
        logging.info("Bitcoin Hourly Data:")
        logging.info(bitcoin_data.tail())
        plot_bitcoin_price(bitcoin_data)
        logging.info(f"Number of data points: {len(bitcoin_data)}")

        # Save to CSV
        bitcoin_data.to_csv('bitcoin_hourly_10years.csv')
        logging.info("Data saved to bitcoin_hourly_10years.csv")
    else:
        logging.error("Unable to retrieve or process Bitcoin data.")