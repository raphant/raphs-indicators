"""
Tests for higher timeframe functionality of indicators.
"""

import logging
import pandas as pd
import pytest
from unittest.mock import patch
from raphs_indicators import supertrend
from raphs_indicators.exceptions import DownloadError

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

def test_supertrend_with_timeframes(sample_ohlcv):
    """Test Supertrend with multiple timeframes."""
    # Create a valid config with custom params for higher timeframes
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {'multiplier': 2.0, 'period': 5},
        '1d': {'multiplier': 1.5, 'period': 3}
    }
    
    # Mock the download_ohlcv function to return our sample data
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        mock_download.return_value = {
            '4h': sample_ohlcv.copy(),
            '1d': sample_ohlcv.copy()
        }
        
        result = supertrend(sample_ohlcv, config=config)
        
        # Check base timeframe results
        assert 'supertrend_value' in result
        assert 'supertrend_signal' in result
        
        # Check higher timeframe results
        assert 'supertrend_value_4h' in result
        assert 'supertrend_signal_4h' in result
        assert 'supertrend_value_1d' in result
        assert 'supertrend_signal_1d' in result
        
        # Verify all signals are binary
        for key in ['supertrend_signal', 'supertrend_signal_4h', 'supertrend_signal_1d']:
            signal = result[key]
            assert isinstance(signal, pd.Series)
            assert len(signal) == len(sample_ohlcv)
            assert signal.dtype == int
            assert set(signal.unique()).issubset({0, 1})
            
        # Verify values are different due to different parameters
        assert not result['supertrend_value'].equals(result['supertrend_value_4h'])
        assert not result['supertrend_value'].equals(result['supertrend_value_1d'])
        assert not result['supertrend_value_4h'].equals(result['supertrend_value_1d'])

def test_supertrend_with_timeframes_missing_config():
    """Test Supertrend with missing config keys."""
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
        '4h': {'multiplier': 2.0, 'period': 5}
    }
    
    with pytest.raises(ValueError, match="Config must include"):
        supertrend(sample_data, config=invalid_config)
        
    # Test with no config (should return just base timeframe results)
    result = supertrend(sample_data)
    assert 'supertrend_value' in result
    assert 'supertrend_signal' in result
    assert 'supertrend_value_4h' not in result
    assert 'supertrend_signal_4h' not in result
    assert 'supertrend_value_1d' not in result
    assert 'supertrend_signal_1d' not in result

def test_supertrend_with_timeframes_download_error(sample_ohlcv):
    """Test Supertrend handling of download errors."""
    config = {
        'symbol': 'BTC/USDT',
        'original_timeframe': '1h',
        '4h': {'multiplier': 2.0, 'period': 5},
        '1d': {'multiplier': 1.5, 'period': 3}
    }
    
    # Mock download_ohlcv to raise an exception
    with patch('raphs_indicators.download_ohlcv') as mock_download:
        mock_download.side_effect = Exception("Download failed")
        
        # Should raise DownloadError with appropriate message
        with pytest.raises(DownloadError) as exc_info:
            supertrend(sample_ohlcv, config=config)
            
        # Verify error details
        assert exc_info.value.symbol == 'BTC/USDT'
        assert exc_info.value.timeframe == '4h'  # First timeframe that fails
        assert 'Download failed' in str(exc_info.value) 