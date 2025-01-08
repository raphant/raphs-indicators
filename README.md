# Raph's Technical Analysis Indicators

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A Python library providing robust, well-tested technical analysis indicators for financial market analysis. Built with a focus on reliability, performance, and avoiding lookahead bias.

## ✨ Features

- **Strict Data Validation**: Enforces OHLCV (Open, High, Low, Close, Volume) data requirements
- **Lookahead Bias Prevention**: All signals are properly shifted to avoid using future data
- **Rich Logging**: Detailed debug logging with pretty formatting using `rich`
- **Vectorized Operations**: Optimized for performance using NumPy and TA-Lib
- **Comprehensive Testing**: Full test coverage with pytest
- **Multi-Timeframe Support**: Built-in support for analyzing multiple timeframes simultaneously
- **Type Hints**: Full Python type hints for better IDE support
- **Modular Design**: Easy to extend with new indicators
- **Cryptocurrency Support**: Built-in CCXT integration for crypto market data

## 🚀 Installation

Requires Python 3.10 or higher.

### Using UV (Recommended)

```bash
uv add "raphs-indicators @ git+https://github.com/raphant/raphs-indicators.git"
```

UV is a fast, reliable, and deterministic Python package installer. Learn more at [UV's documentation](https://github.com/astral-sh/uv).

### Using pip

```bash
pip install git+https://github.com/raphant/raphs-indicators.git
```

### Dependencies

Core dependencies:
- `numpy>=1.26.0`: Numerical computations
- `pandas>=2.2.3`: Data manipulation
- `ta-lib>=0.6.0`: Technical analysis functions
- `rich>=13.7.0`: Pretty logging and formatting
- `ccxt>=4.4.44`: Cryptocurrency exchange API
- `ccxt-easy-dl`: Easy cryptocurrency data downloading

Development dependencies:
- `backtesting>=0.3.3`: Strategy backtesting
- `ipykernel>=6.29.5`: Jupyter notebook support
- `matplotlib>=3.10.0`: Data visualization
- `seaborn>=0.13.2`: Statistical data visualization

## 📊 Available Indicators

Here are the key indicators available. For complete documentation, see the API reference.

### Ladder Breakout Pattern

Identifies potential buy signals based on a specific pattern of higher highs and higher lows.

```python
from raphs_indicators import ladder_breakout

result = ladder_breakout(df)
signals = result['ladder_breakout_signal']  # 1 = buy signal, 0 = no signal
```

### Dual Moving Average

Flexible moving average crossover system with support for multiple MA types.

```python
from raphs_indicators import dual_ma

result = dual_ma(
    df,
    fast_period=10,
    slow_period=20,
    fast_ma_type='EMA',  # 'MA', 'EMA', 'WMA', 'TEMA'
    slow_ma_type='EMA',
    crossover_only=False
)
signals = result['dual_ma_signal']  # 1 = bullish, 0 = bearish
```

### Supertrend

Trend-following indicator that uses ATR to determine support/resistance levels.

```python
from raphs_indicators import supertrend

result = supertrend(df, multiplier=3.0, period=10)
trend = result['supertrend_value']     # Support/resistance levels
signal = result['supertrend_signal']    # 1 = bullish, 0 = bearish
```

### On-Balance Volume (OBV)

Volume-based momentum indicator for predicting price changes.

```python
from raphs_indicators import on_balance_volume

result = on_balance_volume(df)
obv = result['obv_value']  # Cumulative volume flow
```

### Volatility Threshold

Dynamic volatility-based threshold calculation.

```python
from raphs_indicators import volatility_threshold

result = volatility_threshold(df, volatility_multiplier=0.7)
threshold = result['volatility_threshold']  # Threshold as percentage of price
```

## 📋 Data Requirements

- Input DataFrames must contain OHLCV columns
- All column names must be lowercase
- Required columns: `open`, `high`, `low`, `close`, `volume`

## 🎯 Signal Conventions

- All indicator signals use 0/1 values (not True/False)
- 0 = No signal
- 1 = Signal active
- Signals are shifted by +1 to avoid lookahead bias
- Signal column names end with '_signal' (e.g. 'breakout_signal')

## 🔄 Multi-Timeframe Analysis

All indicators support automatic multi-timeframe analysis through the `with_higher_timeframes` decorator:

```python
from raphs_indicators import dual_ma

# Configure timeframes and parameters
config = {
    'symbol': 'BTC/USDT',
    'original_timeframe': '1h',  # Base timeframe
    '4h': {  # 4-hour timeframe with custom parameters
        'fast_period': 5,
        'slow_period': 10,
        'fast_ma_type': 'WMA'
    },
    '1d': {  # Daily timeframe
        'fast_period': 3,
        'slow_period': 7
    }
}

# Calculate indicators across all timeframes
result = dual_ma(df, config)

# Access signals
base_signal = result['dual_ma_signal']           # 1h timeframe
h4_signal = result['dual_ma_signal_4h']         # 4h timeframe
daily_signal = result['dual_ma_signal_1d']      # Daily timeframe
```

## 🛠️ Development

### Setting Up Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/raphant/raphs-indicators.git
   cd raphs-indicators
   ```

2. Install UV (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies:

   ```bash
   uv sync                  # install all dependencies
   uv sync --group dev      # install development dependencies
   ```

### Running Tests

The project uses pytest for testing with the following configuration:
- Test files must be named `test_*.py`
- Tests are located in the `tests` directory
- Verbose output with short tracebacks

```bash
uv sync  # ensure dependencies are up to date
pytest   # runs with -v --tb=short by default
```

### Managing Dependencies

Add a new package:

```bash
uv add package_name                 # regular dependency
uv add --dev package_name          # development dependency
uv add --group prod package_name   # production dependency
```

Update dependencies and regenerate lockfile:

```bash
uv pip compile pyproject.toml -o uv.lock
```

### Development Tools

The project includes several development tools:
- **Backtesting**: Use `backtesting` package for strategy testing
- **Jupyter Support**: Full notebook support with `ipykernel`
- **Visualization**: `matplotlib` and `seaborn` for data plotting
- **Type Checking**: Full type hints throughout the codebase

## 📝 Best Practices

- Always use `df.copy()` when manipulating DataFrames
- Return computed Series rather than modified DataFrames
- Return a dictionary mapping column names to their respective Series
- Use descriptive variable names that match the technical analysis domain
- Add comprehensive docstrings with examples
- Include proper type hints
- Write tests for new indicators

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
