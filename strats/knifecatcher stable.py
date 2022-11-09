# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class KnifeCatcherStable(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # ROI table:
    minimal_roi = {
        "0": 0.107,
        "7": 0.031,
        "19": 0.015,
        "42": 0
    }

    # Stoploss:
    stoploss = -1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.00025
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True
    

    # Optimal timeframe for the strategy.
    # timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    #buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    #sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    #short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    #exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 40

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        ################################## Maxima / Minima Points of High / Low #####################################
        
        pivot_range = int(20)

        # Minima code
        
        conditions1_minima = np.array([(dataframe["low"].shift(periods = pivot_range) < dataframe["low"].shift(periods = pivot_range + lb)) for lb in range(1, pivot_range + 1)])
        conditions2_minima = np.array([(dataframe["low"].shift(periods = pivot_range) < dataframe["low"].shift(periods = pivot_range - lb)) for lb in range(1, pivot_range + 1)])
        conditions_minima = conditions1_minima & conditions2_minima
        # 1st element is if condition is true compared to first candle before,
        # 2nd element is if condition is true compared to second candle before and so on ...
        conditions_minima_T = np.transpose(conditions_minima)
        # First element checks if 10 forward and 10 backward minimum conditions are true for 1st row, 
        # Second element checks if 10 forward and 10 backward minimum conditions are true for 2nd row and so on ...
        check_minima = np.all(conditions_minima_T, axis = 1)
        # Test whether all array elements along a given axis evaluate to True.
        dataframe["check_minima"] = check_minima
        dataframe["check_minima"][dataframe["check_minima"] == False] = None
        dataframe["minima"] = dataframe["low"].shift(periods = pivot_range)[check_minima == True]
        
        # Maxima code
        
        conditions1_maxima = np.array([(dataframe["high"].shift(periods = pivot_range) > dataframe["high"].shift(periods = pivot_range + lb)) for lb in range(1, pivot_range + 1)])
        conditions2_maxima = np.array([(dataframe["high"].shift(periods = pivot_range) > dataframe["high"].shift(periods = pivot_range - lb)) for lb in range(1, pivot_range + 1)])
        conditions_maxima = conditions1_maxima & conditions2_maxima
        # 1st element is if condition is true compared to first candle before,
        # 2nd element is if condition is true compared to second candle before and so on ...

        conditions_maxima_T = np.transpose(conditions_maxima)
        # First element checks if 10 forward and 10 backward maximum conditions are true for 1st row, 
        # Second element checks if 10 forward and 10 backward maximum conditions are true for 2nd row and so on ...

        check_maxima = np.all(conditions_maxima_T, axis = 1)
        # Test whether all array elements along a given axis evaluate to True.

        dataframe["check_maxima"] = check_maxima
        dataframe["check_maxima"][dataframe["check_maxima"] == False] = None
        dataframe["maxima"] = dataframe["high"].shift(periods = pivot_range)[check_maxima == True]
        
        dataframe["maxima"][0] = dataframe["high"][0] * 1.5  # an arbitrarily large value assigned to first row (to make .fillna() function work.)
        dataframe["minima"][0] = dataframe["low"][0] * 0.5  # an arbitrarily small value assigned to first row (to make .fillna() function work.)

        dataframe["maxima"] = dataframe["maxima"].fillna(method = "ffill")  # Fill NaN with last value.
        dataframe["minima"] = dataframe["minima"].fillna(method = "ffill")  # Fill NaN with last value.
        
        
        # Rolling max and min to support the pivot points:
        
        dataframe["rolling_max"] = dataframe["close"].rolling(pivot_range * 2).max().shift(periods = 1)
        dataframe["rolling_min"] = dataframe["close"].rolling(pivot_range * 2).min().shift(periods = 1)
        
        
        # Absolute move
        
        dataframe["move"] = dataframe["open"] - dataframe["close"]
        dataframe["abs_move"] = dataframe["move"].abs()
        dataframe["abs_move_shifted"] = dataframe["abs_move"].shift(periods = 1)
        
        # Shifted Volume
        
        dataframe["volume_shifted"] = dataframe["volume"].shift(periods = 1)


        ##########################################################################################################################
        
        # Volume rolling mean
        
        dataframe["volume_ma_slow"] = dataframe["volume"].rolling(60).mean()
        dataframe["volume_ma_fast"] = dataframe["volume"].rolling(5).mean()
        #dataframe["volume_oscillator"] = (dataframe["volume"] / dataframe["volume_ma"])

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        dataframe["rsi_maxima"] = dataframe["rsi"].shift(periods = pivot_range)[check_maxima == True]  # RSI of last maxima point.
        dataframe["rsi_minima"] = dataframe["rsi"].shift(periods = pivot_range)[check_minima == True]  # RSI of last minima point.
        
        dataframe["rsi_maxima"][0] = 70  # an arbitrarily large value assigned to first row (to make .fillna() function work.)
        dataframe["rsi_minima"][0] = 30  # an arbitrarily small value assigned to first row (to make .fillna() function work.)
        
        dataframe["rsi_maxima"] = dataframe["rsi_maxima"].fillna(method = "ffill")  # Fill NaN with last value.
        dataframe["rsi_minima"] = dataframe["rsi_minima"].fillna(method = "ffill")  # Fill NaN with last value.
        

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        
        dataframe["mfi_maxima"] = dataframe["mfi"].shift(periods = pivot_range)[check_maxima == True]  # MFI of last maxima point.
        dataframe["mfi_minima"] = dataframe["mfi"].shift(periods = pivot_range)[check_minima == True]  # MFI of last minima point.
        
        dataframe["mfi_maxima"][0] = 70  # an arbitrarily large value assigned to first row (to make .fillna() function work.)
        dataframe["mfi_minima"][0] = 30  # an arbitrarily small value assigned to first row (to make .fillna() function work.)
        
        dataframe["mfi_maxima"] = dataframe["mfi_maxima"].fillna(method = "ffill")  # Fill NaN with last value.
        dataframe["mfi_minima"] = dataframe["mfi_minima"].fillna(method = "ffill")  # Fill NaN with last value.
        
        # Normalized Average True Range
        
        #dataframe["natr"] = ta.NATR(dataframe)

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

        # # SMA - Simple Moving Average
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                # Signal:
                (dataframe["low"].shift(periods = 1) < (0.99 * dataframe["sma_fast"].shift(periods = 1))) &  # Previous low is lower than MA.
                (dataframe["close"].shift(periods = 1) < dataframe["open"].shift(periods = 1)) &  # Previous candle is red.
                (dataframe["close"] > dataframe["open"]) &  # Green candle.
                (dataframe["abs_move"] > (0.5 * dataframe["abs_move_shifted"])) &  # Move is greater than half of previous candle.
                (dataframe["volume"] < dataframe["volume_ma_fast"]) & # Volume is lower than the fast volume moving average.
                (dataframe["volume"] < (0.5 * dataframe["volume_shifted"])) & # Volume is lower than half of previous volume.
                (dataframe["volume_shifted"] > (2 * dataframe["volume_ma_slow"])) & # Previous candle volume is large.
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Signal:
                (dataframe["high"].shift(periods = 1) > (1.01 * dataframe["sma_fast"].shift(periods = 1))) &  # Previous high is greater than MA.
                (dataframe["close"].shift(periods = 1) > dataframe["open"].shift(periods = 1)) &  # Previous candle is green.
                (dataframe["close"] < dataframe["open"]) &  # Red candle.
                (dataframe["abs_move"] > (0.5 * dataframe["abs_move_shifted"])) &  # Move is greater than half of previous candle.
                (dataframe["volume"] < dataframe["volume_ma_fast"]) & # Volume is lower than the fast volume moving average.
                (dataframe["volume"] < (0.5 * dataframe["volume_shifted"])) & # Volume is lower than half of previous volume.
                (dataframe["volume_shifted"] > (2 * dataframe["volume_ma_slow"])) & # Previous candle volume is large.
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # Signal: RSI crosses below 50 OR RSI crosses below rsi of minima point (invalidation).
                ((qtpylib.crossed_above(dataframe["high"], dataframe["maxima"])) | (qtpylib.crossed_below(dataframe["rsi"], 50))) & # Candle high swept maxima.
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses below 50 OR RSI crosses above rsi of maxima point (invalidation).
                ((qtpylib.crossed_below(dataframe["low"], dataframe["minima"])) | (qtpylib.crossed_above(dataframe["rsi"], 50))) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return dataframe
