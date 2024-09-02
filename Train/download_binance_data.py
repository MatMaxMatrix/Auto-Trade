
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

def create_binance_json_params(granularity, interval, start_datetime, end_datetime):
    return {
        "bizType": "SPOT",
        "productName":"klines",
        "symbolRequestItems": [
            {
                "endDay":end_datetime.strftime('%Y-%m-%d'),
                "granularityList":[granularity],
                "interval": interval,
                "startDay":start_datetime.strftime('%Y-%m-%d'),
                "symbol":"BTCUSDT"
            }
        ]
    }

def get_download_links(params):
    list_url = 'https://www.binance.com/bapi/bigdata/v1/public/bigdata/finance/exchange/listDownloadData2'
    response = requests.post(url=list_url, json=params)

    data = response.json()
    if 'Error Message' in data:
        print(f"API Error: {data['Error Message']}")
        return None
    
    download_list_months = data.get('data', {}).get('downloadItemList', {})
    if not download_list_months or len(download_list_months) < 1:
        print("No download links found.")
        return None
    
    return download_list_months


def get_binance_bitcoin_data(granularity, interval, start_datetime, end_datetime):
    json_params = create_binance_json_params(granularity, interval, start_datetime, end_datetime)
    download_list = get_download_links(json_params)
    return download_data(download_list)

def download_data(download_list):
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
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
