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
class PandasPivots(IStrategy):
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

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.005,
        "30": 0.01,
        "0": 0.02
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.01

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    # timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
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

        # Maxima / Minima Points of Close
        
        pivot_range = int(20)

        conditions1_minima = np.array([(dataframe["close"].shift(periods = pivot_range) < dataframe["close"].shift(periods = pivot_range + lb)) for lb in range(1, pivot_range + 1)])
        conditions2_minima = np.array([(dataframe["close"].shift(periods = pivot_range) < dataframe["close"].shift(periods = pivot_range - lb)) for lb in range(1, pivot_range + 1)])
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
        dataframe["minima"] = dataframe["close"].shift(periods = pivot_range)[check_minima == True]
        
        conditions1_maxima = np.array([(dataframe["close"].shift(periods = pivot_range) > dataframe["close"].shift(periods = pivot_range + lb)) for lb in range(1, pivot_range + 1)])
        conditions2_maxima = np.array([(dataframe["close"].shift(periods = pivot_range) > dataframe["close"].shift(periods = pivot_range - lb)) for lb in range(1, pivot_range + 1)])
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
        dataframe["maxima"] = dataframe["close"].shift(periods = pivot_range)[check_maxima == True]
        
        dataframe["maxima"][0] = dataframe["close"][0] * 1.5  # an arbitrarily large value assigned to first row (to make .fillna() function work.)
        dataframe["minima"][0] = dataframe["close"][0] * 0.5  # an arbitrarily small value assigned to first row (to make .fillna() function work.)

        dataframe["maxima"] = dataframe["maxima"].fillna(method = "ffill")  # Fill NaN with last value.
        dataframe["minima"] = dataframe["minima"].fillna(method = "ffill")  # Fill NaN with last value.
        
        


        ##########################################################################################################################
        
        # Momentum Indicators
        # ------------------------------------

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # MACD
        #macd = ta.MACD(dataframe)
        #dataframe['macd'] = macd['macd']
        #dataframe['macdsignal'] = macd['macdsignal']
        #dataframe['macdhist'] = macd['macdhist']

        # MFI
        #dataframe['mfi'] = ta.MFI(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=10, stds=1)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
        #     dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
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
                (dataframe["low"] < dataframe["minima"]) &  # Candle low is lower than pivot minima.
                (dataframe["close"] > dataframe["open"]) &  # Green candle.
                # (dataframe["rsi"] < 30) &  # RSI oversold.
                (dataframe['volume'] > 0)  # Make sure) Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # Signal:
                (dataframe["high"] > dataframe["maxima"]) &  # Candle high is higher than pivot maxima.
                (dataframe["close"] < dataframe["open"]) &  # Red candle.
                # (dataframe["rsi"] > 70) &  # RSI overbought.
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
                # Signal: Candle high crosses above maxima point.
                (qtpylib.crossed_above(dataframe["high"], dataframe["maxima"])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                # Signal: Candle low crosses below minima point.
                (qtpylib.crossed_below(dataframe["low"], dataframe["minima"])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return dataframe
