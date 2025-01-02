"""
Tests for main indicator functions in raphs_indicators module.
"""

import logging
import pandas as pd
import pytest
import numpy as np
from raphs_indicators import (
    ladder_breakout,
    dual_ma,
    volatility_threshold,
    validate_ohlcv
)
# Set module logger to DEBUG level for tests
logger = logging.getLogger("raphs_indicators")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        'open':   [10, 11, 12, 13, 14, 15],
        'high':   [12, 13, 14, 15, 16, 17],
        'low':    [9, 10, 11, 12, 13, 14],
        'close':  [11, 12, 13, 14, 15, 16],
        'volume': [100, 100, 100, 100, 100, 100]
    })

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
    assert 'ema10' in result
    assert 'ema20' in result
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
    
    assert 'ma2' in custom_result
    assert 'ema4' in custom_result
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