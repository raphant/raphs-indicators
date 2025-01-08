"""
Custom exceptions for raphs_indicators package.
"""

class DownloadError(Exception):
    """
    Raised when there is an error downloading data from an exchange.
    
    Attributes:
        symbol: The trading pair symbol that failed to download
        timeframe: The timeframe that failed to download
        message: Explanation of the error
    """
    def __init__(self, symbol: str, timeframe: str, message: str = "Failed to download data"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.message = f"{message} for {symbol} ({timeframe})"
        super().__init__(self.message) 