# Cryptocurrency Trading Data Pipeline

## Overview

This project is a comprehensive data pipeline for real-time cryptocurrency trading data processing and prediction, focusing on Bitcoin (BTC) trading data from Binance. The system collects, processes, and prepares data for potential machine learning predictions.

## Project Structure

```
Prediction/
├── data_preparation.py      # Data processing and preparation for modeling
├── kafka_processor.py       # Kafka message processing and InfluxDB storage
├── trading_binance.py       # Binance trading functions
└── websocker_producer.py    # WebSocket data streaming to Kafka
```

## Key Components

### 1. WebSocket Producer (`websocker_producer.py`)
- Connects to Binance WebSocket stream
- Streams real-time kline (candlestick) data for BTCUSDT
- Produces raw trading data to Kafka topic `binance_data`

### 2. Kafka Processor (`kafka_processor.py`)
- Consumes messages from Kafka topic `binance_data`
- Processes and stores trading data in InfluxDB
- Supports potential further data processing and redistribution

### 3. Data Preparation (`data_preparation.py`)
- Fetches historical data from InfluxDB
- Prepares data for machine learning models
- Supports configurable context and prediction lengths

### 4. Trading Functions (`trading_binance.py`)
- Provides utility functions for executing trades on Binance
- Currently supports market buy orders

## Prerequisites

- Python 3.8+
- Kafka
- InfluxDB
- Binance API access

## Required Environment Variables

Create a `.env` file with the following variables:
```
INFLUXDB_URL=
INFLUXDB_USERNAME=
INFLUXDB_PASSWORD=
INFLUXDB_ORG=
KAFKA_BOOSTRAP_SERVERS=
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
BINANCE_API_URL=
BINANCE_WEBSOCKET_URL=
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with the required credentials
4. Ensure Kafka and InfluxDB are running

## Usage

### Start WebSocket Data Collection
```bash
python websocker_producer.py
```

### Process Kafka Messages
```bash
python kafka_processor.py
```

### Prepare Data for Modeling
```python
from data_preparation import DataPreparation

dp = DataPreparation()
prepared_data = dp.run_data_preparation(
    bucket='mybucket', 
    measurement='binance_data',
    start_time='-1h',
    end_time='now()',
    context_length=60,
    prediction_length=30
)
```

## Data Flow

1. WebSocket captures real-time trading data
2. Data is produced to Kafka
3. Kafka processor stores data in InfluxDB
4. Data preparation module extracts and transforms data for modeling


## Disclaimer
This is a research and learning project. Always use caution and never risk more than you can afford to lose when dealing with cryptocurrency trading.

