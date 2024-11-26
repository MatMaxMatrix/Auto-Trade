# Bitcoin Price Prediction System

## Overview
This project implements a sophisticated Bitcoin price prediction system using deep learning models, specifically focusing on different time granularities (seconds, minutes, and hours). The system uses historical Bitcoin price data from Binance and implements various LSTM-based models for prediction.

## Features
- Multiple time granularity support (seconds, minutes, hours)
- Data downloading and preprocessing from Binance
- Advanced LSTM-based models including xLSTM implementations
- Real-time prediction capabilities
- Comprehensive data visualization
- Model training with early stopping and checkpointing
- TensorBoard integration for training monitoring

## Project Structure
```
Train/
├── download_binance_data.py    # Binance data fetching utilities
├── hourly_train.py             # Hourly model training implementation
├── hourly.py                   # Hourly data processing
├── minute_train.py             # Minute-level model training
├── minute-colab.py             # Google Colab compatible minute training
├── minute.py                   # Minute data processing
├── prediction-test.py          # Model prediction testing
└── second.py                   # Second-level data processing
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU support)

### Dependencies
```bash
pip install torch pandas numpy matplotlib requests sklearn tensorboard
```

## Usage

1. Data Collection
```python
# Download Bitcoin price data
python Train/hourly.py    # For hourly data
python Train/minute.py    # For minute data
python Train/second.py    # For second data
```

2. Model Training
```python
# Train the model using different time granularities
python Train/hourly_train.py     # For hourly predictions
python Train/minute_train.py     # For minute predictions
```

3. Making Predictions
```python
python Train/prediction-test.py
```

## Model Architecture
The project implements several sophisticated neural network architectures:
1. **xLSTM Block Stack**
   * Multiple LSTM layers with attention mechanisms
   * Configurable embedding dimensions
   * Dropout layers for regularization
2. **Stock Prediction Model**
   * Input projection layer
   * Configurable LSTM stack
   * Dropout and linear layers for final prediction

## Data Processing
* Automatic data downloading from Binance API
* Time feature engineering (minute, hour, day features)
* Data normalization using MinMaxScaler
* Sequence preparation for time series prediction

## Configuration
Key configurations can be adjusted in the respective training files:
* Context length (lookback period)
* Prediction length (forecast horizon)
* Model architecture parameters
* Training hyperparameters

## Model Training Features
* Early stopping mechanism
* Learning rate scheduling
* Gradient clipping
* TensorBoard integration
* Model checkpointing
* Comprehensive logging

## Visualization
The project includes various visualization tools:
* Price trend plotting
* Prediction visualization
* Training metrics visualization via TensorBoard

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details


## Contact
For any queries or suggestions, please open an issue in the repository or contact: azimipanah.mobin@gmail.com
