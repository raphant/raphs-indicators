from __future__ import annotations
import ccxt
import pandas as pd


def merge_informative_pair(
    dataframe: pd.DataFrame,
    informative: pd.DataFrame,
    timeframe: str,
    timeframe_inf: str,
    ffill: bool = True,
    append_timeframe: bool = True,
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    Correctly merge informative samples to the original dataframe, avoiding lookahead bias.

    Since dates are candle open dates, merging a 15m candle that starts at 15:00, and a
    1h candle that starts at 15:00 will result in all candles to know the close at 16:00
    which they should not know.

    Moves the date of the informative pair by 1 time interval forward.
    This way, the 14:00 1h candle is merged to 15:00 15m candle, since the 14:00 1h candle is the
    last candle that's closed at 15:00, 15:15, 15:30 or 15:45.

    Assuming inf_tf = '1d' - then the resulting columns will be:
    date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

    :param dataframe: Original dataframe with datetime index
    :param informative: Informative pair with datetime index
    :param timeframe: Timeframe of the original pair sample.
    :param timeframe_inf: Timeframe of the informative pair sample.
    :param ffill: Forwardfill missing values - optional but usually required
    :param append_timeframe: Rename columns by appending timeframe.
    :param suffix: A string suffix to add at the end of the informative columns. If specified,
                   append_timeframe must be false.
    :return: Merged dataframe
    :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe
    """
    informative = informative.copy()
    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # No need to forwardshift if the timeframes are identical
        informative.index.name = "merge_index"
        merge_index = informative.index
    elif minutes < minutes_inf:
        # Subtract "small" timeframe so merging is not delayed by 1 small candle
        # Detailed explanation in https://github.com/freqtrade/freqtrade/issues/4073
        if not informative.empty:
            if timeframe_inf == "1M":
                merge_index = (
                    informative.index + pd.offsets.MonthBegin(1)
                ) - pd.to_timedelta(minutes, "m")
            else:
                merge_index = (
                    informative.index
                    + pd.to_timedelta(minutes_inf, "m")
                    - pd.to_timedelta(minutes, "m")
                )
        else:
            merge_index = informative.index
        informative.index = merge_index
        informative.index.name = "merge_index"
    else:
        raise ValueError(
            "Tried to merge a faster timeframe to a slower timeframe."
            "This would create new rows, and can throw off your regular indicators."
        )

    # Rename columns to be unique
    if suffix and append_timeframe:
        raise ValueError(
            "You can not specify `append_timeframe` as True and a `suffix`."
        )
    elif append_timeframe:
        informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]
    elif suffix:
        informative.columns = [f"{col}_{suffix}" for col in informative.columns]

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe.index.name = "merge_index"
    if ffill:
        # https://pandas.pydata.org/docs/user_guide/merging.html#timeseries-friendly-merging
        # merge_ordered - ffill method is 2.5x faster than separate ffill()
        dataframe = pd.merge_ordered(
            dataframe.reset_index(),
            informative.reset_index(),
            fill_method="ffill",
            left_on="merge_index",
            right_on="merge_index",
            how="left",
        ).set_index("merge_index")
    else:
        dataframe = pd.merge(
            dataframe.reset_index(),
            informative.reset_index(),
            left_on="merge_index",
            right_on="merge_index",
            how="left",
        ).set_index("merge_index")

    dataframe.index.name = None
    return dataframe


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60
