"""
Helper functions for technical analysis indicators.
These functions are internal utilities used by the main indicator functions.
"""

import pandas as pd
import talib
import numpy as np
import logging

# Create module logger
logger = logging.getLogger("raphs_indicators")

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that DataFrame has required OHLCV columns and normalize column names to lowercase.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Copy of input DataFrame with lowercase column names
        
    Raises:
        ValueError: If required OHLCV columns are missing
    """
    df_copy = df.copy()
    
    # Convert all column names to lowercase
    df_copy.columns = df_copy.columns.str.lower()
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df_copy.columns]
    
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
    logger.debug("✓ DataFrame validated - all OHLCV columns present")
    return df_copy

def _compare_series(series1: pd.Series, series2: pd.Series, comparison_func) -> pd.Series:
    """
    Internal helper for series comparisons with NaN handling.
    
    Args:
        series1: First series
        series2: Second series
        comparison_func: Function that takes two arrays and returns boolean array
        
    Returns:
        pd.Series: Binary signal series with NaN handling
    """
    # Pre-allocate result array with zeros
    result = np.zeros(len(series1))
    
    # Get valid indices once
    valid_idx = series1.notna() & series2.notna()
    
    # Apply comparison only on valid values
    if valid_idx.any():
        result[valid_idx] = comparison_func(
            series1.values[valid_idx],
            series2.values[valid_idx]
        ).astype(int)
        logger.debug(f"✓ Comparison applied to {valid_idx.sum()} valid points")
    else:
        logger.debug("⚠️ No valid points found for comparison")
    
    return pd.Series(result, index=series1.index, dtype=int)

def series_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect when series1 crosses above series2.
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        pd.Series: Binary signal where 1 indicates series1 crossed above series2,
                  0 indicates no crossover
                  
    Note:
        - NaN values in either series will result in 0 signal for that position
        - Initial values are filled with 0
    """
    logger.debug(f"🔍 Checking crossover between series of length {len(series1)}")
    
    # Get current state using vectorized numpy operations
    curr_state = _compare_series(series1, series2, np.greater)
    prev_state = curr_state.shift(1, fill_value=0)
    
    # Crossover = previous below (0) and current above (1)
    result = ((prev_state == 0) & (curr_state == 1))
    crossover_count = result.sum()
    logger.debug(f"📊 Found {crossover_count} crossover points")
    
    return pd.Series(result.values.astype(int), index=series1.index, dtype=int)

def series_crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect when series1 crosses below series2.
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        pd.Series: Binary signal where 1 indicates series1 crossed below series2,
                  0 indicates no crossunder
                  
    Note:
        - NaN values in either series will result in 0 signal for that position
        - Initial values are filled with 0
    """
    logger.debug(f"🔍 Checking crossunder between series of length {len(series1)}")
    logger.debug(f"📊 Series1 values: {series1.values}")
    logger.debug(f"📊 Series2 values: {series2.values}")
    
    # Fill NaN values with the previous value to maintain continuity
    s1_filled = series1.ffill()
    s2_filled = series2.ffill()
    
    logger.debug(f"📊 Series1 filled: {s1_filled.values}")
    logger.debug(f"📊 Series2 filled: {s2_filled.values}")
    
    # Get current state using vectorized numpy operations
    curr_state = _compare_series(s1_filled, s2_filled, np.less)
    prev_state = _compare_series(s1_filled.shift(1), s2_filled.shift(1), np.greater_equal)
    
    logger.debug(f"📉 Current below state: {curr_state.values}")
    logger.debug(f"📈 Previous above state: {prev_state.values}")
    
    # Crossunder = previous above/equal (1) and current below (1)
    result = (prev_state == 1) & (curr_state == 1)
    
    # First value should always be 0 since we can't determine crossunder
    result = pd.Series(result.values.astype(int), index=series1.index, dtype=int)
    result.iloc[0] = 0
    
    crossunder_count = result.sum()
    logger.debug(f"📊 Found {crossunder_count} crossunder points at indices: {result[result == 1].index.tolist()}")
    
    return result

def series_above(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Detect when series1 is above series2.
    
    Args:
        series1: First series
        series2: Second series
        
    Returns:
        pd.Series: Binary signal where 1 indicates series1 is above series2,
                  0 indicates series1 is below or equal to series2
                  
    Note:
        - NaN values in either series will result in 0 signal for that position
    """
    logger.debug(f"🔍 Checking above condition between series of length {len(series1)}")
    result = _compare_series(series1, series2, np.greater)
    above_count = result.sum()
    logger.debug(f"📊 Found {above_count} points where series1 > series2")
    return result.astype(int)

def detect_crossover(fast_ma: pd.Series, slow_ma: pd.Series, crossover_only: bool = False) -> pd.Series:
    """
    Detect bullish crossovers between fast and slow moving averages.
    
    Args:
        fast_ma: Fast moving average series
        slow_ma: Slow moving average series
        crossover_only: If True, only signals first crossover
        
    Returns:
        pd.Series: Signal series where 1 indicates bullish crossover/trend
        
    Note:
        - NaN values in either MA will result in 0 signal for that position
        - Initial values are filled with 0
    """
    logger.debug(f"🔍 Detecting {'crossover only' if crossover_only else 'trend'} between MAs")
    result = series_crossover(fast_ma, slow_ma) if crossover_only else series_above(fast_ma, slow_ma)
    signal_count = result.sum()
    logger.debug(f"📊 Found {signal_count} {'crossover' if crossover_only else 'trend'} signals")
    return result

def calculate_ma(df: pd.DataFrame, period: int, ma_type: str) -> pd.Series:
    """
    Calculate moving average based on specified type.
    
    Args:
        df: DataFrame with OHLCV data
        period: MA period (must be positive integer)
        ma_type: One of 'MA', 'EMA', 'WMA', 'TEMA'
        
    Returns:
        pd.Series: Calculated moving average
        
    Raises:
        ValueError: If period <= 0 or ma_type is invalid
    """
    if not isinstance(period, int) or period <= 0:
        logger.error(f"❌ Invalid period: {period}")
        raise ValueError(f"Period must be a positive integer, got {period}")
        
    ma_type = ma_type.upper()
    valid_types = {'MA', 'EMA', 'WMA', 'TEMA'}
    if ma_type not in valid_types:
        logger.error(f"❌ Invalid MA type: {ma_type}")
        raise ValueError(f"Unsupported MA type: {ma_type}. Use one of {valid_types}")
    
    logger.debug(f"📈 Calculating {ma_type} with period={period}")
    
    # Convert to float64 for TA-Lib
    close_values = df['close'].values.astype(np.float64)
    
    if ma_type == 'MA':
        result = pd.Series(talib.SMA(close_values, timeperiod=period), index=df.index)
    elif ma_type == 'EMA':
        result = pd.Series(talib.EMA(close_values, timeperiod=period), index=df.index)
    elif ma_type == 'WMA':
        result = pd.Series(talib.WMA(close_values, timeperiod=period), index=df.index)
    else:  # TEMA
        result = pd.Series(talib.TEMA(close_values, timeperiod=period), index=df.index)
    
    logger.debug(f"✨ MA calculation complete - {result.notna().sum()} valid points")
    return result 
    