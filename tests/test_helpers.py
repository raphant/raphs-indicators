"""
Tests for helper functions in raphs_indicators.helpers module.
"""

import pandas as pd
import numpy as np
import pytest
import logging
from raphs_indicators.helpers import (
    series_crossover,
    series_crossunder,
    series_above,
    calculate_ma
)

# Set module logger to DEBUG level for tests
logger = logging.getLogger("raphs_indicators")
logger.setLevel(logging.DEBUG)

@pytest.fixture
def sample_series():
    """Create sample series for testing crossovers."""
    series1 = pd.Series([1, 2, 3, 2, 1])
    series2 = pd.Series([2, 2, 2, 2, 2])
    return series1, series2

def test_series_crossover(sample_series):
    """Test series_crossover function."""
    series1, series2 = sample_series
    result = series_crossover(series1, series2)
    expected = pd.Series([0, 0, 1, 0, 0])
    pd.testing.assert_series_equal(result, expected)

def test_series_crossunder(sample_series):
    """Test series_crossunder function."""
    series1, series2 = sample_series
    result = series_crossunder(series1, series2)
    expected = pd.Series([0, 0, 0, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_series_above(sample_series):
    """Test series_above function."""
    series1, series2 = sample_series
    result = series_above(series1, series2)
    expected = pd.Series([0, 0, 1, 0, 0])
    pd.testing.assert_series_equal(result, expected)

def test_nan_handling():
    """Test NaN handling in comparison functions."""
    series1 = pd.Series([1, np.nan, 3, 2, 1])
    series2 = pd.Series([2, 2, 2, np.nan, 2])
    
    # Test crossover with NaN
    crossover_result = series_crossover(series1, series2)
    expected_crossover = pd.Series([0, 0, 1, 0, 0])
    pd.testing.assert_series_equal(crossover_result, expected_crossover)
    
    # Test crossunder with NaN
    crossunder_result = series_crossunder(series1, series2)
    expected_crossunder = pd.Series([0, 0, 0, 0, 1])
    pd.testing.assert_series_equal(crossunder_result, expected_crossunder)

def test_calculate_ma_validation():
    """Test input validation for calculate_ma function."""
    df = pd.DataFrame({
        'close': [1, 2, 3, 4, 5]
    })
    
    # Test invalid period
    with pytest.raises(ValueError, match="Period must be a positive integer"):
        calculate_ma(df, 0, 'MA')
    
    # Test invalid MA type
    with pytest.raises(ValueError, match="Unsupported MA type"):
        calculate_ma(df, 2, 'INVALID')

def test_calculate_ma_types():
    """Test different MA type calculations."""
    df = pd.DataFrame({
        'close': [1, 2, 3, 4, 5]
    })
    
    # Test SMA
    sma = calculate_ma(df, 2, 'MA')
    assert len(sma) == len(df)
    assert not sma.isna().all()
    
    # Test EMA
    ema = calculate_ma(df, 2, 'EMA')
    assert len(ema) == len(df)
    assert not ema.isna().all()
    
    # Test WMA
    wma = calculate_ma(df, 2, 'WMA')
    assert len(wma) == len(df)
    assert not wma.isna().all()
    
    # Test TEMA
    tema = calculate_ma(df, 2, 'TEMA')
    assert len(tema) == len(df)
    assert not tema.isna().all() 