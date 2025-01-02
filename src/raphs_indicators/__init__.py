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

import pandas as pd
import talib
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
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        dict: Dictionary with key 'ladder_breakout_signal' mapping to an integer Series
              where 1 indicates a buy signal and 0 indicates no signal. Note that signals 
              are shifted by +1 to avoid lookahead bias, meaning the signal appears on 
              the next bar after the pattern completes.
              
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = ladder_breakout(df)
        >>> result['ladder_breakout_signal']
        0    0
        1    0
        2    0
        3    0
        4    1
        dtype: int64
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug(f"🔍 Analyzing ladder breakout pattern for {len(df)} bars")
    
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
    
    # Shift by 1 to avoid lookahead bias
    buy_signal = buy_signal.shift(1).fillna(0).astype(int)
    
    return {
        'ladder_breakout_signal': buy_signal
    }

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
             
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = volatility_threshold(df)
        >>> result['volatility_threshold']
        0         NaN
        1    0.300000
        2    0.272727
        3    0.250000
        4    0.230769
        dtype: float64
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug(f"📈 Calculating volatility threshold with volatility_multiplier={volatility_multiplier}")
    logger.debug(f"📊 Input data shape: {df_calc.shape}")
    logger.debug(f"📊 High values: {df_calc['high'].values}")
    logger.debug(f"📊 Low values: {df_calc['low'].values}")
    logger.debug(f"📊 Close values: {df_calc['close'].values}")
    
    # Convert to float64 for TA-Lib
    high = df_calc['high'].values.astype(np.float64)
    low = df_calc['low'].values.astype(np.float64)
    close = df_calc['close'].values.astype(np.float64)
    
    # Calculate TR using TA-Lib
    tr = pd.Series(talib.TRANGE(high, low, close), index=df_calc.index)
    logger.debug(f"📊 TR values: {tr.values}")
    
    # Calculate threshold as percentage of price
    # First value will be NaN due to TR calculation
    threshold = pd.Series(np.nan, index=df_calc.index)
    
    # Calculate only where we have valid TR values and non-zero low prices
    # Skip the first value which should remain NaN
    valid_idx = (df_calc['low'] > 0) & tr.notna() & (pd.Series(range(len(df_calc))) > 0)
    logger.debug(f"✨ Valid indices: {valid_idx}")
    logger.debug(f"📊 Low prices > 0: {df_calc['low'] > 0}")
    logger.debug(f"📊 TR not NaN: {tr.notna()}")
    logger.debug(f"📊 Index > first: {pd.Series(range(len(df_calc))) > 0}")
    
    if valid_idx.any():
        # Calculate threshold as percentage of price
        threshold[valid_idx] = (tr[valid_idx] / df_calc['low'][valid_idx]) * volatility_multiplier
        logger.debug(f"✨ Calculated thresholds for {valid_idx.sum()} valid bars")
        logger.debug(f"📊 TR values used: {tr[valid_idx].values}")
        logger.debug(f"📊 Low values used: {df_calc['low'][valid_idx].values}")
        logger.debug(f"📊 Raw ratios (TR/Low): {(tr[valid_idx] / df_calc['low'][valid_idx]).values}")
        logger.debug(f"📊 Final threshold values: {threshold.values}")
        logger.debug(f"📊 Threshold stats - Mean: {threshold.mean():.4f}, Max: {threshold.max():.4f}, Min: {threshold.min():.4f}")
    else:
        logger.warning("⚠️ No valid data points found for threshold calculation")
    
    return {
        'volatility_threshold': threshold
    }

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
    
    The indicator calculates two moving averages (fast and slow) using specified types
    and periods, then generates signals based on their relative positions.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Period for fast moving average (must be < slow_period)
        slow_period: Period for slow moving average
        fast_ma_type: Type of fast MA ('MA', 'EMA', 'WMA', 'TEMA')
        slow_ma_type: Type of slow MA ('MA', 'EMA', 'WMA', 'TEMA')
        crossover_only: If True, signal only triggers on initial crossover
        
    Returns:
        dict: Dictionary containing:
            - '{fast_ma_type}{fast_period}': Fast moving average series
            - '{slow_ma_type}{slow_period}': Slow moving average series
            - 'dual_ma_binary_signal': Binary signal series (1=bullish, 0=bearish)
            
    Raises:
        ValueError: If fast_period >= slow_period
            
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
        
    logger.debug(f"📊 Calculating dual MA crossover - Fast: {fast_ma_type}({fast_period}), Slow: {slow_ma_type}({slow_period})")
    
    # Calculate fast and slow MAs
    fast_ma = calculate_ma(df_calc, fast_period, fast_ma_type)
    slow_ma = calculate_ma(df_calc, slow_period, slow_ma_type)
    
    # Generate signal
    signal = detect_crossover(fast_ma, slow_ma, crossover_only)
    
    # Count signals before shift
    signal_count = signal.sum()
    logger.debug(f"✨ Found {signal_count} {'crossover' if crossover_only else 'trend'} signals")
    
    # Shift signal by 1 to avoid lookahead bias
    signal = signal.shift(1).fillna(0).astype(int)
    
    # Create dynamic column names based on MA types and periods
    fast_col = f"{fast_ma_type.lower()}{fast_period}"
    slow_col = f"{slow_ma_type.lower()}{slow_period}"
    signal_col = f"dual_ma_signal"
    
    return {
        fast_col: fast_ma,
        slow_col: slow_ma,
        signal_col: signal
    }

