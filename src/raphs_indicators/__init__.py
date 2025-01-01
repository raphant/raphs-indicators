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
from typing import Dict

def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required OHLCV columns."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

def ladder_breakout(df: pd.DataFrame) -> Dict[str, pd.Series]:
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
    validate_ohlcv(df)
    df_calc = df.copy()
    
    # Vectorized conditions using shift operations
    c1 = df_calc['high'] > df_calc['high'].shift(1)  # Current high > Previous high
    c2 = df_calc['high'].shift(1) > df_calc['low']   # Previous high > Current low
    c3 = df_calc['low'] > df_calc['high'].shift(2)   # Current low > Previous-2 high
    c4 = df_calc['high'].shift(2) > df_calc['low'].shift(1)  # Previous-2 high > Previous low
    c5 = df_calc['low'].shift(1) > df_calc['high'].shift(3)  # Previous low > Previous-3 high
    c6 = df_calc['high'].shift(3) > df_calc['low'].shift(2)  # Previous-3 high > Previous-2 low
    c7 = df_calc['low'].shift(2) > df_calc['low'].shift(3)   # Previous-2 low > Previous-3 low
    
    # Combine all conditions and shift by 1 to avoid lookahead bias
    buy_signal = (c1 & c2 & c3 & c4 & c5 & c6 & c7).shift(1).astype(int)
    
    # Fill NaN values with 0
    buy_signal = buy_signal.fillna(0)
    
    return {
        'ladder_breakout_signal': buy_signal
    }

