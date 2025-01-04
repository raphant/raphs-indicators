"""
Raphs Technical Analysis Indicators Library

This module contains technical analysis indicators for financial market analysis.
All indicator functions follow strict conventions for consistency and reliability.

DataFrame Requirements:
---------------------
- Input DataFrames must contain OHLCV columns (open, high, low, close, volume)
- All column names must be lowercase
- Missing columns will raise descriptive ValueError exceptions

Signal Conventions:
-----------------
- All indicator signals use 0/1 values (not True/False)
- 0 = No signal
- 1 = Signal active
- Signals are shifted by +1 to avoid lookahead bias
- Signal column names end with '_signal' (e.g. 'breakout_signal')

Best Practices:
-------------
- Always use df.copy() when manipulating DataFrames to prevent modifying originals
- Return computed Series rather than modified DataFrames
- Return a dictionary mapping column names to their respective Series
- Use descriptive variable names that match the technical analysis domain

Avoiding Lookahead Bias:
----------------------
- Never use future data points to calculate current signals
- Always use .iloc[] instead of .loc[] to ensure strict positional indexing
- When calculating indicators, only use data available up to the current position
- For rolling windows, ensure window size N only uses past N-1 periods
- Shift any signals that depend on the current candle's close price by +1
- Test indicators with walk-forward analysis to verify no future data leakage
- Document any assumptions about data availability in docstrings

DO's:
-----
✓ Validate input DataFrame has required columns before computation
✓ Return meaningful error messages when requirements aren't met
✓ Document all parameters and return values clearly
✓ Include references to technical analysis documentation
✓ Use vectorized operations instead of loops for performance
✓ Include examples in docstrings showing typical usage
✓ Names must be unique and descriptive

DON'Ts:
-------
✗ Modify input DataFrames directly - always return new Series
✗ Use uppercase column names
✗ Assume columns exist without validation
✗ Return different data structures across different indicator functions
✗ Use non-standard column names without documentation
✗ Perform heavy computations without progress indicators for long operations
✗ Use future data points to calculate current signals
✗ Mix .loc[] and .iloc[] indexing methods

Example:
-------
def my_indicator(df: pd.DataFrame, param: int) -> Dict[str, pd.Series]:
    '''
    Compute custom indicator.
    
    Args:
        df: DataFrame with OHLCV data
        param: Indicator parameter
        
    Returns:
        dict: Dictionary mapping column names to indicator Series
              e.g. {'my_ind_upper': upper_series, 'my_ind_lower': lower_series}
    '''
    validate_ohlcv(df)  # Always validate first
    df_calc = df.copy()  # Use copy for calculations
    
    upper = compute_upper(df_calc, param)
    lower = compute_lower(df_calc, param)
    
    return {
        f'my_ind_upper': upper,
        f'my_ind_lower': lower
    }
"""

from ccxt_easy_dl import download_ohlcv
import pandas as pd
import talib

from raphs_indicators.utils import merge_informative_pair
from .helpers import calculate_ma, detect_crossover, validate_ohlcv
import numpy as np
import logging
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Create module logger
logger = logging.getLogger("raphs_indicators")
logger.setLevel(logging.INFO)

def with_timeframes(func):
    """
    Decorator that automatically calculates indicators across multiple timeframes.
    
    The config dictionary must include the original_timeframe and symbol:
    {
        'symbol': 'BTC/USD',
        'original_timeframe': '1h',  # Required - base timeframe
        '4h': {'param1': value1},
        '1d': {'param1': value1}
    }
    """
    def wrapper(df: pd.DataFrame, config: dict = None, *args, **kwargs):
        # Get base results first using original arguments
        logger.debug("🔄 Calculating base timeframe indicators")
        base_results = func(df, *args, **kwargs)
        
        if not config:
            return base_results
            
        # Validate required config keys
        required_keys = {'symbol', 'original_timeframe'}
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            logger.error(f"❌ Missing required config keys: {missing_keys}")
            raise ValueError(f"Config must include {required_keys}")
            
        final_results = base_results.copy()
        symbol = config['symbol']
        original_tf = config['original_timeframe']
        
        # Get timeframe configs (excluding special keys)
        timeframe_configs = {k: v for k, v in config.items() 
                           if k not in {'symbol', 'original_timeframe'}}
        
        # Extract date range from input DataFrame
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Calculate for each additional timeframe
        for tf, tf_kwargs in timeframe_configs.items():
            logger.debug(f"🔄 Processing {tf} timeframe for {symbol}")
            try:
                tf_df = download_ohlcv(
                    symbol,
                    timeframes=[tf],
                    start_date=start_date,
                    end_date=end_date
                )[tf]
                
                # Calculate indicators for this timeframe
                tf_results = func(tf_df, **tf_kwargs)
                
                # Convert results to DataFrame for merging
                tf_results_df = pd.DataFrame(tf_results)
                
                # Create empty DataFrame with correct index
                empty_df = pd.DataFrame(index=df.index)
                
                # Merge with higher timeframe data
                merged = merge_informative_pair(
                    dataframe=empty_df,
                    informative=tf_results_df,
                    timeframe=original_tf,
                    timeframe_inf=tf,
                    ffill=True,
                    append_timeframe=True,
                    suffix=None
                )
                
                # Add merged columns to final results
                for col in merged.columns:
                    # Check if this is a signal column (with timeframe suffix)
                    is_signal = any(col.startswith(f"{base_col}_") and base_col.endswith('_signal') 
                                  for base_col in tf_results_df.columns)
                    
                    if is_signal:
                        try:
                            # Convert signal columns to int, handling NaN values
                            final_results[col] = merged[col].fillna(0).astype('int64')
                        except Exception as e:
                            logger.error(f"❌ Failed to convert {col} to int: {str(e)}")
                            raise
                    else:
                        final_results[col] = merged[col]
                    
            except Exception as e:
                logger.exception(f"❌ Failed to process {tf} timeframe: {str(e)}")
                continue
                
        return final_results
        
    return wrapper

@with_timeframes
def ladder_breakout(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute the Ladder Breakout indicator which identifies potential buy signals
    based on a specific pattern of higher highs and higher lows.
    
    The indicator checks 7 conditions across 4 consecutive price bars to identify
    a bullish ladder breakout pattern:
    
    Pattern Requirements:
    Bar N (current):  High > Bar N-1 High
    Bar N-1:         High > Bar N Low
    Bar N:           Low > Bar N-2 High
    Bar N-2:         High > Bar N-1 Low
    Bar N-1:         Low > Bar N-3 High
    Bar N-3:         High > Bar N-2 Low
    Bar N-2:         Low > Bar N-3 Low
    
    Visual Pattern:
          ┌─── Current Bar (N)
          │    High > Previous High
          │
    ┌─────┘    Previous Bar (N-1)
    │          
    │     ┌──── Bar N-2
    └─────┘
          │
          └──── Bar N-3
    """
    df_calc = validate_ohlcv(df)
    
    # Vectorized conditions using shift operations
    c1 = df_calc['high'] > df_calc['high'].shift(1)  # Current high > Previous high
    c2 = df_calc['high'].shift(1) > df_calc['low']   # Previous high > Current low
    c3 = df_calc['low'] > df_calc['high'].shift(2)   # Current low > Previous-2 high
    c4 = df_calc['high'].shift(2) > df_calc['low'].shift(1)  # Previous-2 high > Previous low
    c5 = df_calc['low'].shift(1) > df_calc['high'].shift(3)  # Previous low > Previous-3 high
    c6 = df_calc['high'].shift(3) > df_calc['low'].shift(2)  # Previous-3 high > Previous-2 low
    c7 = df_calc['low'].shift(2) > df_calc['low'].shift(3)   # Previous-2 low > Previous-3 low
    
    # Combine all conditions and handle NaN values before converting to int
    buy_signal = (c1 & c2 & c3 & c4 & c5 & c6 & c7).fillna(False).astype(int)
    
    # Count signals before shift
    signal_count = buy_signal.sum()
    logger.debug(f"📊 Found {signal_count} ladder breakout patterns")
    
    # Shift by 1 to avoid lookahead bias and ensure int type
    buy_signal = buy_signal.shift(1).fillna(0).astype(int)
    
    return {
        'ladder_breakout_signal': buy_signal
    }

@with_timeframes
def volatility_threshold(
    df: pd.DataFrame, 
    volatility_multiplier: float = 0.7
) -> dict[str, pd.Series]:
    """
    Calculate a volatility-based threshold as a percentage of price using ATR(1).
    
    The threshold is calculated as: (ATR(1) / low) * volatility_multiplier
    This provides a dynamic threshold that adapts to both price and volatility levels.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        volatility_multiplier (float, optional): Multiplier to scale the threshold. 
            Higher values create wider thresholds. Defaults to 0.7.
        
    Returns:
        dict: Dictionary with key 'volatility_threshold' mapping to a Series
             representing the threshold as a percentage of price
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug(f"📈 Calculating volatility threshold (multiplier={volatility_multiplier})")
    
    # Convert to float64 for TA-Lib
    high = df_calc['high'].values.astype(np.float64)
    low = df_calc['low'].values.astype(np.float64)
    close = df_calc['close'].values.astype(np.float64)
    
    # Calculate TR using TA-Lib
    tr = pd.Series(talib.TRANGE(high, low, close), index=df_calc.index)
    
    # Calculate threshold as percentage of price
    # First value will be NaN due to TR calculation
    threshold = pd.Series(np.nan, index=df_calc.index)
    
    # Calculate only where we have valid TR values and non-zero low prices
    valid_idx = (df_calc['low'] > 0) & tr.notna()
    
    if valid_idx.any():
        # Calculate threshold as percentage of price
        threshold[valid_idx] = (tr[valid_idx] / df_calc['low'][valid_idx]) * volatility_multiplier
        logger.debug(f"✨ Calculated thresholds for {valid_idx.sum()} bars")
    else:
        logger.warning("⚠️ No valid data points found for threshold calculation")
    
    return {
        'volatility_threshold': threshold
    }

@with_timeframes
def dual_ma(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 20,
    fast_ma_type: str = 'EMA',
    slow_ma_type: str = 'EMA',
    crossover_only: bool = False
) -> dict[str, pd.Series]:
    """
    Calculate dual moving average indicator with flexible MA types and crossover detection.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Period for fast moving average (must be < slow_period)
        slow_period: Period for slow moving average
        fast_ma_type: Type of fast MA ('MA', 'EMA', 'WMA', 'TEMA')
        slow_ma_type: Type of slow MA ('MA', 'EMA', 'WMA', 'TEMA')
        crossover_only: If True, signal only triggers on initial crossover
        
    Returns:
        dict: Dictionary containing:
            - dual_ma_{fast_ma_type}_fast: Fast moving average series
            - dual_ma_{slow_ma_type}_slow: Slow moving average series
            - dual_ma_signal: Binary signal series (1=bullish, 0=bearish)
            
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = dual_ma(df, fast_period=2, slow_period=3)
        >>> result['dual_ma_signal']
        0    0
        1    0
        2    1
        3    1
        4    1
        dtype: int64
    """
    df_calc = validate_ohlcv(df)
    
    if fast_period >= slow_period:
        logger.error(f"❌ Invalid periods: fast_period ({fast_period}) >= slow_period ({slow_period})")
        raise ValueError(f"fast_period ({fast_period}) must be less than slow_period ({slow_period})")
        
    # Calculate fast and slow MAs
    fast_ma = calculate_ma(df_calc, fast_period, fast_ma_type)
    slow_ma = calculate_ma(df_calc, slow_period, slow_ma_type)
    
    # Generate signal
    signal = detect_crossover(fast_ma, slow_ma, crossover_only)
    
    # Count signals before shift
    signal_count = signal.sum()
    logger.debug(f"📊 Found {signal_count} {'crossover' if crossover_only else 'trend'} signals")
    
    # Shift signal by 1 to avoid lookahead bias
    signal = signal.shift(1).fillna(0).astype(int)
    
    # Create dynamic column names based on MA types
    fast_col = f"dual_ma_{fast_ma_type.lower()}_fast"
    slow_col = f"dual_ma_{slow_ma_type.lower()}_slow"
    signal_col = "dual_ma_signal"
    
    return {
        fast_col: fast_ma,
        slow_col: slow_ma,
        signal_col: signal
    }

