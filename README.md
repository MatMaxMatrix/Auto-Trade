# Cryptocurrency Trading and Prediction System

## Overview
This project is a comprehensive cryptocurrency trading and prediction system that combines real-time data collection, processing, and machine learning-based price prediction for Bitcoin (BTC). The system is split into two main components: Prediction Pipeline and Training Pipeline.

## Project Structure
```
./
├── Prediction/ # Real-time data collection and processing
│   ├── data_preparation.py
│   ├── kafka_processor.py
│   ├── trading_binance.py
│   └── websocker_producer.py
├── Train/ # Model training and prediction
│   ├── download_binance_data.py
│   ├── hourly_train.py
│   ├── hourly.py
│   ├── minute_train.py
│   ├── minute-colab.py
│   ├── minute.py
│   ├── prediction-test.py
│   └── second.py
├── .env
├── docker-compose.yml
└── requirements.txt
```

## Features
- Real-time cryptocurrency data collection via Binance WebSocket
- Data processing pipeline using Kafka and InfluxDB
- Multiple time granularity support (seconds, minutes, hours)
- Advanced LSTM-based prediction models
- Docker-based deployment
- Comprehensive data visualization
- TensorBoard integration for training monitoring

## Prerequisites
- Python 3.8+
- Docker and Docker Compose
- CUDA (optional, for GPU support)
- Binance API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables by copying the example .env file:
```bash
cp .env.example .env
```

4. Update the .env file with your credentials:
```
INFLUXDB_URL=http://localhost:8086
INFLUXDB_USERNAME=myuser
INFLUXDB_PASSWORD=mypassword
INFLUXDB_ORG=myorg
KAFKA_BOOSTRAP_SERVERS=localhost:9092
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

5. Start the infrastructure services:
```bash
docker-compose up -d
```

## Usage

### Real-time Data Collection
Start the WebSocket producer:
```bash
python Prediction/websocker_producer.py
```

Start the Kafka processor:
```bash
python Prediction/kafka_processor.py
```

### Model Training
Download historical data:
```bash
python Train/download_binance_data.py
```

Train models at different time granularities:
```bash
python Train/hourly_train.py    # For hourly predictions
python Train/minute_train.py    # For minute predictions
```

Test predictions:
```bash
python Train/prediction-test.py
```

## Model Architecture
The project implements sophisticated neural network architectures:
- xLSTM Block Stack with attention mechanisms
- Configurable embedding dimensions
- Dropout layers for regularization
- Input projection layer
- Multiple LSTM layers

## Data Processing Features
- Automatic data downloading from Binance API
- Time feature engineering
- Data normalization
- Sequence preparation for time series prediction
- Real-time data processing pipeline

## Monitoring and Visualization
- TensorBoard integration for training metrics
- Price trend plotting
- Prediction visualization
- Comprehensive logging

## Infrastructure Components

### Kafka
- Message broker for real-time data streaming
- Configured with Zookeeper for cluster management
- Exposed on port 9092

### InfluxDB
- Time-series database for storing trading data
- Web interface available at port 8086
- Automatic bucket creation and data retention policies

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Disclaimer
This is a research and learning project. Use caution and never risk more than you can afford to lose when dealing with cryptocurrency trading.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For queries or suggestions, please contact: azimipanah.mobin@gmail.com
