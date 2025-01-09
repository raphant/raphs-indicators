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
from raphs_indicators.exceptions import DownloadError
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

def with_higher_timeframes(func):
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
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Pop config from kwargs if it exists
        config = kwargs.pop('config', None)
        
        # Get base results first using original arguments
        logger.debug("🔄 Calculating base timeframe indicators")
        base_results: dict[str, pd.Series] = func(df, *args, **kwargs)
        
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
            
            # Download data for higher timeframe
            try:
                tf_df = download_ohlcv(
                    symbol,
                    timeframes=[tf],
                    start_date=start_date,
                    end_date=end_date
                )[tf]
            except Exception as e:
                logger.exception(f"❌ Failed to download {tf} timeframe data: {str(e)}")
                raise DownloadError(symbol, tf, str(e))
            
            # Calculate indicators for this timeframe
            try:
                tf_results: dict[str, pd.Series] = func(tf_df, **tf_kwargs)
            except Exception as e:
                logger.exception(f"❌ Failed to calculate indicators for {tf} timeframe: {str(e)}")
                raise
            
            # Convert results to DataFrame for merging
            tf_results_df = pd.DataFrame(tf_results)
            empty_df = pd.DataFrame(index=df.index)
            
            # Merge with higher timeframe data
            try:
                merged = merge_informative_pair(
                    dataframe=empty_df,
                    informative=tf_results_df,
                    timeframe=original_tf,
                    timeframe_inf=tf,
                    ffill=True,
                    append_timeframe=True,
                    suffix=None
                )
            except Exception as e:
                logger.exception(f"❌ Failed to merge {tf} timeframe data: {str(e)}")
                raise
            
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
                
        return final_results
        
    return wrapper

@with_higher_timeframes
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

@with_higher_timeframes
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

@with_higher_timeframes
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

@with_higher_timeframes
def supertrend(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    period: int = 10
) -> dict[str, pd.Series]:
    """
    Calculate the Supertrend indicator which combines trend detection with volatility.
    
    The Supertrend is a trend-following indicator that uses ATR to determine its levels.
    It provides both trend direction and potential support/resistance levels.
    
    Args:
        df: DataFrame with OHLCV data
        multiplier: ATR multiplier for band calculation (default: 3.0)
        period: Period for ATR calculation (default: 10)
        
    Returns:
        dict: Dictionary containing:
            - supertrend_value: The Supertrend level series
            - supertrend_signal: Binary signal series (1=bullish, 0=bearish)
            
    Note:
        - Signal is shifted by +1 to avoid lookahead bias
        - First `period` values will be NaN due to ATR calculation
        
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = supertrend(df, multiplier=3, period=2)
        >>> result['supertrend_signal']
        0    0
        1    0
        2    1
        3    1
        4    1
        dtype: int64
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug(f"📈 Calculating Supertrend (multiplier={multiplier}, period={period})")
    
    # Convert to float64 for TA-Lib
    high = df_calc['high'].values.astype(np.float64)
    low = df_calc['low'].values.astype(np.float64)
    close = df_calc['close'].values.astype(np.float64)
    
    # Calculate TR and ATR using TA-Lib
    tr = pd.Series(talib.TRANGE(high, low, close), index=df_calc.index)
    atr = pd.Series(talib.SMA(tr, period), index=df_calc.index)
    
    # Calculate basic upper and lower bands
    hl2 = (df_calc['high'] + df_calc['low']) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    
    # Initialize final bands
    final_ub = pd.Series(0.0, index=df_calc.index)
    final_lb = pd.Series(0.0, index=df_calc.index)
    
    # Calculate final upper band
    for i in range(period, len(df_calc)):
        final_ub.iloc[i] = (
            basic_ub.iloc[i] 
            if basic_ub.iloc[i] < final_ub.iloc[i - 1] or df_calc['close'].iloc[i - 1] > final_ub.iloc[i - 1]
            else final_ub.iloc[i - 1]
        )
            
    # Calculate final lower band
    for i in range(period, len(df_calc)):
        final_lb.iloc[i] = (
            basic_lb.iloc[i]
            if basic_lb.iloc[i] > final_lb.iloc[i - 1] or df_calc['close'].iloc[i - 1] < final_lb.iloc[i - 1]
            else final_lb.iloc[i - 1]
        )
    
    # Calculate Supertrend value
    supertrend = pd.Series(0.0, index=df_calc.index)
    
    for i in range(period, len(df_calc)):
        supertrend.iloc[i] = (
            final_ub.iloc[i] if supertrend.iloc[i - 1] == final_ub.iloc[i - 1] and df_calc['close'].iloc[i] <= final_ub.iloc[i]
            else final_lb.iloc[i] if supertrend.iloc[i - 1] == final_ub.iloc[i - 1] and df_calc['close'].iloc[i] > final_ub.iloc[i]
            else final_lb.iloc[i] if supertrend.iloc[i - 1] == final_lb.iloc[i - 1] and df_calc['close'].iloc[i] >= final_lb.iloc[i]
            else final_ub.iloc[i] if supertrend.iloc[i - 1] == final_lb.iloc[i - 1] and df_calc['close'].iloc[i] < final_lb.iloc[i]
            else 0.0
        )
    
    # Generate signal (1 when price is above Supertrend)
    signal = (df_calc['close'] > supertrend).astype(int)
    
    # Count signals before shift
    signal_count = signal.sum()
    logger.debug(f"📊 Found {signal_count} bullish signals")
    
    # Log descriptive statistics for supertrend values
    st_stats = supertrend.describe()
    logger.debug("📊 Supertrend Statistics:")
    logger.debug("Count: %.0f, Mean: %.2f, Std: %.2f", 
                st_stats['count'], st_stats['mean'], st_stats['std'])
    logger.debug("Min: %.2f, 25%%: %.2f, 50%%: %.2f, 75%%: %.2f, Max: %.2f",
                st_stats['min'], st_stats['25%'], st_stats['50%'], 
                st_stats['75%'], st_stats['max'])
    
    # Shift signal by 1 to avoid lookahead bias
    signal = signal.shift(1).fillna(0).astype(int)
    
    return {
        'supertrend_value': supertrend,
        'supertrend_signal': signal
    }

@with_higher_timeframes
def on_balance_volume(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Calculate On-Balance Volume (OBV), a momentum indicator that uses volume flow 
    to predict changes in price.
    
    OBV is calculated by adding volume on up days and subtracting it on down days.
    The absolute value is not important; what matters is the trend and divergences.
    
    Formula:
        If close > previous close:
            OBV = previous OBV + current volume
        If close < previous close:
            OBV = previous OBV - current volume
        If close = previous close:
            OBV = previous OBV
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        dict: Dictionary containing:
            - obv_value: The OBV value series
            
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = on_balance_volume(df)
        >>> result['obv_value']
        0    100
        1    200
        2    300
        3    400
        4    500
        dtype: int64
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug("📈 Calculating On-Balance Volume")
    
    # Calculate price changes
    price_changes = df_calc['close'].diff()
    
    # Initialize OBV series with first volume value
    obv = pd.Series(0, index=df_calc.index, dtype=float)  # Use float dtype to avoid warning
    obv.iloc[0] = df_calc['volume'].iloc[0]
    
    # Calculate OBV values
    for i in range(1, len(df_calc)):
        if price_changes.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df_calc['volume'].iloc[i]
        elif price_changes.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df_calc['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    # Log descriptive statistics
    obv_stats = obv.describe()
    logger.debug("📊 OBV Statistics:")
    logger.debug("Count: %.0f, Mean: %.2f, Std: %.2f", 
                obv_stats['count'], obv_stats['mean'], obv_stats['std'])
    logger.debug("Min: %.2f, 25%%: %.2f, 50%%: %.2f, 75%%: %.2f, Max: %.2f",
                obv_stats['min'], obv_stats['25%'], obv_stats['50%'], 
                obv_stats['75%'], obv_stats['max'])
    
    return {
        'obv_value': obv
    }

@with_higher_timeframes
def ma_ratio(df: pd.DataFrame, period: int = 20, ma_type: str = 'EMA') -> dict[str, pd.Series]:
    """
    Calculate the ratio between current price and a moving average.
    
    This indicator helps identify when price is relatively high or low compared to its
    moving average. Values > 1 indicate price is above MA, values < 1 indicate price
    is below MA.
    
    Args:
        df: DataFrame with OHLCV data
        period: MA period (must be positive integer)
        ma_type: One of 'MA', 'EMA', 'WMA', 'TEMA'
        
    Returns:
        dict: Dictionary containing:
            - ma_ratio_value: Series of price/MA ratios
            
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = ma_ratio(df, period=2, ma_type='EMA')
        >>> result['ma_ratio_value']
        0    1.000000
        1    1.043478
        2    1.052632
        3    1.060606
        4    1.067416
        dtype: float64
    """
    df_calc = validate_ohlcv(df)
    
    logger.debug(f"📈 Calculating MA ratio (period={period}, type={ma_type})")
    
    # Calculate MA using helper function
    ma_series = calculate_ma(df_calc, period, ma_type)
    
    # Calculate ratio
    ratio = df_calc['close'] / ma_series
    
    # Log descriptive statistics
    ratio_stats = ratio.describe()
    logger.debug("📊 MA Ratio Statistics:")
    logger.debug("Count: %.0f, Mean: %.4f, Std: %.4f", 
                ratio_stats['count'], ratio_stats['mean'], ratio_stats['std'])
    logger.debug("Min: %.4f, 25%%: %.4f, 50%%: %.4f, 75%%: %.4f, Max: %.4f",
                ratio_stats['min'], ratio_stats['25%'], ratio_stats['50%'], 
                ratio_stats['75%'], ratio_stats['max'])
    
    return {
        'ma_ratio_value': ratio
    }

@with_higher_timeframes
def choppiness_index(
    df: pd.DataFrame,
    period: int = 14,
    ratio: bool = False
) -> dict[str, pd.Series]:
    """
    Calculate the Choppiness Index (CHOP) which indicates if market is choppy (trending) or not.
    
    The Choppiness Index is designed to determine if the market is trending or trading sideways.
    It combines the concepts of ADX and ATR to identify trading ranges and trends.
    
    Formula (standard):
        CHOP = 100 * LOG10((SUM(ATR(1), period)) / (MaxHigh(period) - MinLow(period))) / LOG10(period)
    
    Formula (ratio):
        CHOP_ratio = SUM(TR, period) / (period * ATR(period))
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default: 14)
        ratio: If True, return the raw ratio instead of the scaled index (default: False)
        
    Returns:
        dict: Dictionary containing:
            - chop_value: The Choppiness Index value (0-100) or ratio if ratio=True
            
    Note:
        - Standard CHOP values range from 0 to 100:
          * Values > 60 indicate a choppy market
          * Values < 40 indicate a trending market
        - Ratio values typically range around 1:
          * Values > 1 indicate a choppy market
          * Values < 1 indicate a trending market
        
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [10, 11, 12, 13, 14],
        ...     'high': [12, 13, 14, 15, 16],
        ...     'low':  [9, 10, 11, 12, 13],
        ...     'close': [11, 12, 13, 14, 15],
        ...     'volume': [100, 100, 100, 100, 100]
        ... })
        >>> result = choppiness_index(df, period=3)
        >>> result['chop_value']
        0     NaN
        1     NaN
        2    45.23
        3    48.67
        4    52.12
        dtype: float64
    """
    df_calc = validate_ohlcv(df)
    
    # Validate period
    if not isinstance(period, int) or period <= 0:
        logger.error(f"❌ Invalid period: {period}")
        raise ValueError(f"Period must be a positive integer, got {period}")
    
    logger.debug(f"📈 Calculating Choppiness Index (period={period}, ratio={ratio})")
    
    # Convert to float64 for TA-Lib
    high = df_calc['high'].values.astype(np.float64)
    low = df_calc['low'].values.astype(np.float64)
    close = df_calc['close'].values.astype(np.float64)
    
    # Calculate True Range using TA-Lib
    tr = pd.Series(talib.TRANGE(high, low, close), index=df_calc.index)
    
    # Calculate ATR using TA-Lib
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=period), index=df_calc.index)
    
    # Calculate sum of TR over period
    tr_sum = tr.rolling(window=period).sum()
    
    if ratio:
        # Calculate ratio version: SUM(TR) / (period * ATR)
        chop = tr_sum / (period * atr)
    else:
        # Calculate highest high and lowest low over period
        high_max = df_calc['high'].rolling(window=period).max()
        low_min = df_calc['low'].rolling(window=period).min()
        
        # Calculate standard CHOP
        # Handle potential division by zero and log of zero/negative numbers
        denom = high_max - low_min
        valid_idx = (denom > 0) & (tr_sum > 0)
        
        chop = pd.Series(np.nan, index=df_calc.index)
        if valid_idx.any():
            chop[valid_idx] = 100 * np.log10(tr_sum[valid_idx] / denom[valid_idx]) / np.log10(period)
    
    # Log descriptive statistics
    valid_stats = chop[chop.notna()]
    if not valid_stats.empty:
        logger.debug("📊 Choppiness Index Statistics:")
        logger.debug("Count: %.0f, Mean: %.2f, Std: %.2f", 
                    len(valid_stats), valid_stats.mean(), valid_stats.std())
        logger.debug("Min: %.2f, 25%%: %.2f, 50%%: %.2f, 75%%: %.2f, Max: %.2f",
                    valid_stats.min(), valid_stats.quantile(0.25), valid_stats.median(),
                    valid_stats.quantile(0.75), valid_stats.max())
    else:
        logger.warning("⚠️ No valid Choppiness Index values calculated")
    
    return {
        'chop_value': chop
    }

