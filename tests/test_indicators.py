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
    validate_ohlcv
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

def test_ladder_breakout_with_timeframes(sample_ohlcv):
    """Test ladder breakout with multiple timeframes."""
    # Create a valid config
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {},  # No special params needed for ladder_breakout
        '1d': {}
    }
    
    # Mock the download_ohlcv function to return our sample data
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        # Create sample data for higher timeframes
        mock_download.return_value = {
            '4h': sample_ohlcv.copy(),
            '1d': sample_ohlcv.copy()
        }
        
        result = ladder_breakout(sample_ohlcv, config)
        
        # Check base timeframe results
        assert 'ladder_breakout_signal' in result
        assert isinstance(result['ladder_breakout_signal'], pd.Series)
        
        # Check higher timeframe results
        assert 'ladder_breakout_signal_4h' in result
        assert 'ladder_breakout_signal_1d' in result
        
        # Verify all signals are binary
        for key in ['ladder_breakout_signal', 'ladder_breakout_signal_4h', 'ladder_breakout_signal_1d']:
            signal = result[key]
            assert isinstance(signal, pd.Series)
            assert len(signal) == len(sample_ohlcv)
            assert signal.dtype == int
            assert set(signal.unique()).issubset({0, 1})

def test_ladder_breakout_with_timeframes_missing_config():
    """Test ladder breakout with missing config keys."""
    sample_data = pd.DataFrame({
        'open':   [10, 11, 12, 13, 14],
        'high':   [12, 13, 14, 15, 16],
        'low':    [9, 10, 11, 12, 13],
        'close':  [11, 12, 13, 14, 15],
        'volume': [100, 100, 100, 100, 100]
    })
    
    # Test with missing required keys
    invalid_config = {
        'symbol': 'BTC/USDT',
        # missing original_timeframe
        '4h': {}
    }
    
    with pytest.raises(ValueError, match="Config must include"):
        ladder_breakout(sample_data, invalid_config)
        
    # Test with no config (should return just base timeframe results)
    result = ladder_breakout(sample_data)
    assert 'ladder_breakout_signal' in result
    assert 'ladder_breakout_signal_4h' not in result
    assert 'ladder_breakout_signal_1d' not in result

def test_ladder_breakout_with_timeframes_download_error(sample_ohlcv):
    """Test ladder breakout handling of download errors."""
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {},
        '1d': {}
    }
    
    # Mock download_ohlcv to raise an exception
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        mock_download.side_effect = Exception("Download failed")
        
        # Should still return base timeframe results even if higher timeframes fail
        result = ladder_breakout(sample_ohlcv, config)
        
        assert 'ladder_breakout_signal' in result
        assert isinstance(result['ladder_breakout_signal'], pd.Series)
        # Higher timeframe signals should not be present due to download failure
        assert 'ladder_breakout_signal_4h' not in result
        assert 'ladder_breakout_signal_1d' not in result 

def test_dual_ma_with_timeframes(sample_ohlcv):
    """Test dual MA with multiple timeframes."""
    # Create a valid config with custom params for higher timeframes
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {
            'fast_period': 5,
            'slow_period': 10,
            'fast_ma_type': 'WMA',
            'slow_ma_type': 'TEMA'
        },
        '1d': {
            'fast_period': 3,
            'slow_period': 7,
            'crossover_only': True
        }
    }
    
    # Mock the download_ohlcv function to return our sample data
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        # Create sample data for higher timeframes
        mock_download.return_value = {
            '4h': sample_ohlcv.copy(),
            '1d': sample_ohlcv.copy()
        }
        
        result = dual_ma(sample_ohlcv, config)
        
        # Check base timeframe results (default params)
        assert 'dual_ma_ema_fast' in result
        assert 'dual_ma_ema_slow' in result
        assert 'dual_ma_signal' in result
        
        # Check 4h timeframe results
        assert 'dual_ma_wma_fast_4h' in result
        assert 'dual_ma_tema_slow_4h' in result
        assert 'dual_ma_signal_4h' in result
        
        # Check 1d timeframe results
        assert 'dual_ma_ema_fast_1d' in result
        assert 'dual_ma_ema_slow_1d' in result
        assert 'dual_ma_signal_1d' in result
        
        # Verify all signals are binary
        for key in ['dual_ma_signal', 'dual_ma_signal_4h', 'dual_ma_signal_1d']:
            signal = result[key]
            assert isinstance(signal, pd.Series)
            assert len(signal) == len(sample_ohlcv)
            assert signal.dtype == int
            assert set(signal.unique()).issubset({0, 1})

def test_dual_ma_with_timeframes_missing_config():
    """Test dual MA with missing config keys."""
    sample_data = pd.DataFrame({
        'open':   [10, 11, 12, 13, 14],
        'high':   [12, 13, 14, 15, 16],
        'low':    [9, 10, 11, 12, 13],
        'close':  [11, 12, 13, 14, 15],
        'volume': [100, 100, 100, 100, 100]
    })
    
    # Test with missing required keys
    invalid_config = {
        'symbol': 'BTC/USDT',
        # missing original_timeframe
        '4h': {'fast_period': 5, 'slow_period': 10}
    }
    
    with pytest.raises(ValueError, match="Config must include"):
        dual_ma(sample_data, invalid_config)
        
    # Test with no config (should return just base timeframe results)
    result = dual_ma(sample_data)
    assert 'dual_ma_ema_fast' in result
    assert 'dual_ma_ema_slow' in result
    assert 'dual_ma_signal' in result
    assert 'dual_ma_signal_4h' not in result
    assert 'dual_ma_signal_1d' not in result

def test_dual_ma_with_timeframes_download_error(sample_ohlcv):
    """Test dual MA handling of download errors."""
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {'fast_period': 5, 'slow_period': 10},
        '1d': {'fast_period': 3, 'slow_period': 7}
    }
    
    # Mock download_ohlcv to raise an exception
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        mock_download.side_effect = Exception("Download failed")
        
        # Should still return base timeframe results even if higher timeframes fail
        result = dual_ma(sample_ohlcv, config)
        
        assert 'dual_ma_ema_fast' in result
        assert 'dual_ma_ema_slow' in result
        assert 'dual_ma_signal' in result
        # Higher timeframe signals should not be present due to download failure
        assert 'dual_ma_signal_4h' not in result
        assert 'dual_ma_signal_1d' not in result 