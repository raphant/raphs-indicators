# Raph's Technical Analysis Indicators

A Python library providing my own robust, well-tested technical analysis indicators for financial market analysis. Built with a focus on reliability, performance, and avoiding lookahead bias.

## Features

- **Strict Data Validation**: Enforces OHLCV (Open, High, Low, Close, Volume) data requirements
- **Lookahead Bias Prevention**: All signals are properly shifted to avoid using future data
- **Rich Logging**: Detailed debug logging with pretty formatting using `rich`
- **Vectorized Operations**: Optimized for performance using NumPy and TA-Lib
- **Comprehensive Testing**: Full test coverage with pytest

## Installation

Requires Python 3.9 or higher.

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

- numpy
- pandas
- ta-lib
- rich

## Sample Indicators & Usage

Here are some examples of the available indicators. For a complete list and detailed documentation, please refer to the API documentation.

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

### Volatility Threshold

Dynamic volatility-based threshold calculation.

```python
from raphs_indicators import volatility_threshold

result = volatility_threshold(df, volatility_multiplier=0.7)
threshold = result['volatility_threshold']  # Threshold as percentage of price
```

## Data Requirements

- Input DataFrames must contain OHLCV columns
- All column names must be lowercase
- Required columns: 'open', 'high', 'low', 'close', 'volume'

## Signal Conventions

- All indicator signals use 0/1 values (not True/False)
- 0 = No signal
- 1 = Signal active
- Signals are shifted by +1 to avoid lookahead bias
- Signal column names end with '_signal' (e.g. 'breakout_signal')

## Multi-Timeframe Analysis

All indicators support automatic multi-timeframe analysis through the `with_higher_timeframes` decorator. This allows you to calculate indicators across multiple timeframes in a single call.

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

The higher timeframe results are automatically downloaded, calculated, and merged with the base timeframe data. All signals are properly aligned to avoid lookahead bias.

## Development

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
   uv sync
   ```

### Running Tests

```bash
uv sync  # ensure dependencies are up to date
pytest tests/
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

### Best Practices

- Always use `df.copy()` when manipulating DataFrames
- Return computed Series rather than modified DataFrames
- Return a dictionary mapping column names to their respective Series
- Use descriptive variable names that match the technical analysis domain
- Add comprehensive docstrings with examples
- Include proper type hints
- Write tests for new indicators

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
