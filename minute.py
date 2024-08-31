#%%
import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from zipfile import ZipFile
from io import BytesIO
#%%
def get_bitcoin_minute_data(days=70, output_csv = None):
    list_url = 'https://www.binance.com/bapi/bigdata/v1/public/bigdata/finance/exchange/listDownloadData2'
    today_datetime = datetime.today()
    start_datetime = datetime.today() - timedelta(days=days)
    json_params_monthly = {
        "bizType": "SPOT",
        "productName":"klines",
        "symbolRequestItems": [
            {
                "endDay":today_datetime.strftime('%Y-%m-%d'),
                "granularityList":["1m"],
                "interval":"monthly",
                "startDay":start_datetime.strftime('%Y-%m-%d'),
                "symbol":"BTCUSDT"
            }
        ]
    }
    response = requests.post(url=list_url, json=json_params_monthly)
    data = response.json()
    if 'Error Message' in data:
        print(f"API Error: {data['Error Message']}")
        return None
    
    download_list_months = data.get('data', {}).get('downloadItemList', {})
    if not download_list_months or len(download_list_months) < 1:
        print("No monthly download links found.")
        return None

    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
    df = download_data(download_list_months, column_names)

    # Add days from the current month if needed
    highest_timestamp_seconds = df['time'].max() / 1000

    start_day = datetime.fromtimestamp(highest_timestamp_seconds)
    json_params_daily = {
        "bizType": "SPOT",
        "productName":"klines",
        "symbolRequestItems": [
            {
                "endDay":today_datetime.strftime('%Y-%m-%d'),
                "granularityList":["1m"],
                "interval":"daily",
                "startDay":start_day.strftime('%Y-%m-%d'),
                "symbol":"BTCUSDT"
            }
        ]
    }
    response = requests.post(url=list_url, json=json_params_daily)
    data = response.json()

    if 'Error Message' in data:
        print(f"API Error: {data['Error Message']}")
        return None
    
    download_list_days = data.get('data', {}).get('downloadItemList', {})

    if not download_list_days or len(download_list_days) < 1:
        print("No daily download links found.")
        return None

    df = pd.concat([df, download_data(download_list_days, column_names)])

    # Index and sort by time
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df[['close']]
    if output_csv:
        df.to_csv(output_csv)

    return df
#%%
def download_data(download_list, column_names):
    df = pd.DataFrame()
    for download_info in download_list:

        zipped_content = requests.get(download_info['url']).content
        monthly_data_files = ZipFile(BytesIO(zipped_content))
        filename = monthly_data_files.namelist()[0]
        # Create DataFrame from the data
        partial_df = pd.read_csv(monthly_data_files.open(filename), header=None, names=column_names)

        # Convert all columns to numeric
        for col in partial_df.columns:
            partial_df[col] = pd.to_numeric(partial_df[col], errors='coerce')

        # Select and rename columns
        partial_df = partial_df.loc[:, ['open_time', 'open', 'high', 'low', 'close']]
        partial_df.rename(columns={'open_time': 'time'}, inplace=True)

        df = pd.concat(objs=[df, partial_df], ignore_index=True)
    return df

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

bitcoin_data = get_bitcoin_minute_data(days=70, output_csv=output_file_path)
if bitcoin_data is not None and not bitcoin_data.empty:
    print("Latest Bitcoin Minute Data:")
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
