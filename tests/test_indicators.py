"""
Tests for main indicator functions in raphs_indicators module.
"""

import logging
import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch
from raphs_indicators import (
    ladder_breakout,
    dual_ma,
    volatility_threshold,
    validate_ohlcv,
    supertrend,
    on_balance_volume,
    ma_ratio
)
# Set module logger to DEBUG level for tests
logger = logging.getLogger("raphs_indicators")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    # Create datetime index starting from now
    index = pd.date_range(start='2024-01-01', periods=6, freq='1h')
    
    return pd.DataFrame({
        'open':   [10, 11, 12, 13, 14, 15],
        'high':   [12, 13, 14, 15, 16, 17],
        'low':    [9, 10, 11, 12, 13, 14],
        'close':  [11, 12, 13, 14, 15, 16],
        'volume': [100, 100, 100, 100, 100, 100]
    }, index=index)

def test_validate_ohlcv():
    """Test OHLCV validation function."""
    # Valid DataFrame
    valid_df = pd.DataFrame({
        'open': [1], 'high': [2], 'low': [0.5],
        'close': [1.5], 'volume': [1000]
    })
    validate_ohlcv(valid_df)  # Should not raise
    
    # Missing columns
    invalid_df = pd.DataFrame({
        'open': [1], 'high': [2], 'close': [1.5]
    })
    with pytest.raises(ValueError, match="DataFrame missing required columns"):
        validate_ohlcv(invalid_df)

def test_ladder_breakout(sample_ohlcv):
    """Test ladder breakout indicator."""
    result = ladder_breakout(sample_ohlcv)
    signal = result['ladder_breakout_signal']
    
    # Basic validation
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(sample_ohlcv)
    assert signal.dtype == int
    assert set(signal.unique()).issubset({0, 1})
    
    # First few values should be 0 due to lookback period
    assert (signal.iloc[:3] == 0).all()

def test_dual_ma(sample_ohlcv):
    """Test dual moving average indicator."""
    # Test with default parameters
    result = dual_ma(sample_ohlcv)
    
    # Check all expected keys are present
    assert 'dual_ma_ema_fast' in result
    assert 'dual_ma_ema_slow' in result
    assert 'dual_ma_signal' in result
    
    # Validate signal series
    signal = result['dual_ma_signal']
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(sample_ohlcv)
    assert signal.dtype == int
    assert set(signal.unique()).issubset({0, 1})
    
    # Test with custom parameters
    custom_result = dual_ma(
        sample_ohlcv,
        fast_period=2,
        slow_period=4,
        fast_ma_type='MA',
        slow_ma_type='EMA'
    )
    
    assert 'dual_ma_ma_fast' in custom_result
    assert 'dual_ma_ema_slow' in custom_result
    assert 'dual_ma_signal' in custom_result
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        dual_ma(sample_ohlcv, fast_period=5, slow_period=5)

def test_dual_ma_crossover_only(sample_ohlcv):
    """Test dual MA with crossover_only parameter."""
    result = dual_ma(
        sample_ohlcv,
        fast_period=2,
        slow_period=4,
        crossover_only=True
    )
    
    signal = result['dual_ma_signal']
    # Signal should only be 1 at crossover points
    assert len(signal[signal == 1]) <= len(signal[signal == 0])

def test_volatility_threshold(sample_ohlcv):
    """Test volatility threshold indicator."""
    result = volatility_threshold(sample_ohlcv)
    threshold = result['volatility_threshold']
    
    # Basic validation
    assert isinstance(threshold, pd.Series)
    assert len(threshold) == len(sample_ohlcv)
    
    # First value should be NaN due to TR calculation
    assert pd.isna(threshold.iloc[0])
    
    # All other values should be positive
    assert (threshold.iloc[1:] > 0).all()
    
    # Test with custom volatility multiplier
    custom_result = volatility_threshold(
        sample_ohlcv,
        volatility_multiplier=1.5
    )
    custom_threshold = custom_result['volatility_threshold']
    assert len(custom_threshold) == len(sample_ohlcv)
    
    # Verify that custom threshold is proportionally larger
    ratio = custom_threshold.iloc[1:] / threshold.iloc[1:]
    expected_ratio = 1.5 / 0.7
    assert np.allclose(ratio, expected_ratio, rtol=1e-10, atol=0), f"Expected ratio {expected_ratio}, got {ratio}"

def test_supertrend(sample_ohlcv):
    """Test Supertrend indicator."""
    # Test with default parameters
    result = supertrend(sample_ohlcv)
    
    # Check all expected keys are present
    assert 'supertrend_value' in result
    assert 'supertrend_signal' in result
    
    # Validate supertrend series
    st = result['supertrend_value']
    assert isinstance(st, pd.Series)
    assert len(st) == len(sample_ohlcv)
    
    # First few values should be 0 due to ATR calculation period
    assert (st.iloc[:10] == 0).all()
    
    # Validate signal series
    signal = result['supertrend_signal']
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(sample_ohlcv)
    assert signal.dtype == int
    assert set(signal.unique()).issubset({0, 1})
    
    # Test with custom parameters
    custom_result = supertrend(
        sample_ohlcv,
        multiplier=2.0,
        period=5
    )
    
    # Verify custom parameters produce different results
    assert not custom_result['supertrend_value'].equals(result['supertrend_value'])

def test_on_balance_volume(sample_ohlcv):
    """Test On-Balance Volume (OBV) indicator."""
    # Test with default parameters
    result = on_balance_volume(sample_ohlcv)
    
    # Check expected key is present
    assert 'obv_value' in result
    
    # Validate OBV series
    obv = result['obv_value']
    assert isinstance(obv, pd.Series)
    assert len(obv) == len(sample_ohlcv)
    
    # First value should be equal to first volume
    assert obv.iloc[0] == sample_ohlcv['volume'].iloc[0]
    
    # Verify OBV calculation logic
    for i in range(1, len(sample_ohlcv)):
        price_change = sample_ohlcv['close'].iloc[i] - sample_ohlcv['close'].iloc[i-1]
        if price_change > 0:
            assert obv.iloc[i] == obv.iloc[i-1] + sample_ohlcv['volume'].iloc[i]
        elif price_change < 0:
            assert obv.iloc[i] == obv.iloc[i-1] - sample_ohlcv['volume'].iloc[i]
        else:
            assert obv.iloc[i] == obv.iloc[i-1]
            
    # Test with zero volume
    zero_vol_df = sample_ohlcv.copy()
    zero_vol_df['volume'] = 0
    zero_result = on_balance_volume(zero_vol_df)
    assert (zero_result['obv_value'] == 0).all()
    
    # Test with all same prices (no change)
    same_price_df = sample_ohlcv.copy()
    same_price_df['close'] = 10
    same_price_result = on_balance_volume(same_price_df)
    assert (same_price_result['obv_value'] == same_price_df['volume'].iloc[0]).all() 

def test_ma_ratio(sample_ohlcv):
    """Test Moving Average Ratio indicator."""
    # Test with default parameters
    result = ma_ratio(sample_ohlcv)
    
    # Check expected key is present
    assert 'ma_ratio_value' in result
    
    # Validate ratio series
    ratio = result['ma_ratio_value']
    assert isinstance(ratio, pd.Series)
    assert len(ratio) == len(sample_ohlcv)
    
    # First few values should be NaN due to MA calculation
    assert pd.isna(ratio.iloc[0])
    
    # All other values should be positive (since prices and MAs are positive)
    assert (ratio.dropna() > 0).all()
    
    # Test with different MA types
    ma_types = ['MA', 'EMA', 'WMA', 'TEMA']
    for ma_type in ma_types:
        custom_result = ma_ratio(sample_ohlcv, period=2, ma_type=ma_type)
        custom_ratio = custom_result['ma_ratio_value']
        assert len(custom_ratio) == len(sample_ohlcv)
        assert (custom_ratio.dropna() > 0).all()
    
    # Test with invalid MA type
    with pytest.raises(ValueError, match="Unsupported MA type"):
        ma_ratio(sample_ohlcv, ma_type='INVALID')
    
    # Test with invalid period
    with pytest.raises(ValueError, match="Period must be a positive integer"):
        ma_ratio(sample_ohlcv, period=0) 