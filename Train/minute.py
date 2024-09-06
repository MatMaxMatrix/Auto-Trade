#%%
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from download_binance_data import get_binance_bitcoin_data

#%%
def get_bitcoin_minute_data(days=70, output_csv = None):
    today_datetime = datetime.today()
    start_datetime = datetime.today() - timedelta(days=days)
    df_monthly = get_binance_bitcoin_data("1m", "monthly", start_datetime, today_datetime)

    # Add days from the current month if needed
    highest_timestamp_seconds = df_monthly['time'].max() / 1000

    start_day = datetime.fromtimestamp(highest_timestamp_seconds)
    df_daily = get_binance_bitcoin_data("1m", "daily", start_day, today_datetime)

    df = pd.concat([df_monthly, df_daily])

    # Index and sort by time
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df[['close']]
    if output_csv:
        df.to_csv(output_csv)

    return df

#%%
def plot_bitcoin_price(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.title(f'Bitcoin Minute Closing Price (Last {len(df) / 60 / 24} Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test the function
output_file_path = "bitcoin_minute_data.csv"

bitcoin_data = get_bitcoin_minute_data(days=365, output_csv=output_file_path)
if bitcoin_data is not None and not bitcoin_data.empty:
    print("Latest Bitcoin Minute Data:")
    print(bitcoin_data.head())
    print(bitcoin_data.tail())
    plot_bitcoin_price(bitcoin_data)
else:
    print("Unable to retrieve or process Bitcoin data.")

# %%
#experiment
import pandas as pd
csv_file_path = "bitcoin_minute_data.csv"
bitcoin_close_data = pd.read_csv(csv_file_path, usecols=['close'])
print(bitcoin_close_data.head())

# %%
