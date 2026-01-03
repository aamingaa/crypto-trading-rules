import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from dataload import data_load_v2
# import ccxt

# ==============================
# 1. 数据获取函数（Binance期货数据）
# ==============================
# def fetch_binance_futures_data(symbol="ETH/USDT", timeframe="1h", limit=1000):
#     """
#     从Binance获取期货K线数据
#     :param symbol: 交易对，默认ETH/USDT
#     :param timeframe: K线周期，如1h, 4h
#     :param limit: 获取数据条数
#     :return: 标准化的DataFrame，包含OHLCV+时间戳
#     """
#     # 初始化Binance期货交易所
#     exchange = ccxt.binance({
#         'enableRateLimit': True,  # 开启限流保护
#         'options': {
#             'defaultType': 'future'  # 期货市场
#         }
#     })
    
#     # 获取K线数据
#     ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
#     # 转换为DataFrame并标准化列名
#     df = pd.DataFrame(
#         ohlcv,
#         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
#     )
#     # 转换时间戳为可读格式
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     # 确保数据类型正确
#     df = df.astype({
#         'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
#     })
#     # 去除空值
#     df = dropna(df)
#     return df

# ==============================
# 2. 辅助指标实现（ta库未提供的部分）
# ==============================
def calculate_heikin_ashi(df):
    """
    计算Heikin Ashi（平均K线）
    :param df: 原始OHLCV DataFrame
    :return: 包含HA列的DataFrame
    """
    ha_df = df.copy()
    # HA收盘价 = (开+高+低+收)/4
    ha_df['ha_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    # HA开盘价：首行为原始开盘，后续为前一行HA开+HA收的均值
    ha_df['ha_open'] = np.nan
    ha_df.loc[0, 'ha_open'] = (ha_df.loc[0, 'open'] + ha_df.loc[0, 'close']) / 2
    for i in range(1, len(ha_df)):
        ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2
    # HA最高价 = max(高, HA开, HA收)
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    # HA最低价 = min(低, HA开, HA收)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    return ha_df

def calculate_qqe_mod(series, rsi_length=6, rsi_smoothing=5, qqe_factor=3.0):
    """
    实现QQE MOD核心计算逻辑
    :param series: 价格序列（如close）
    :param rsi_length: RSI周期
    :param rsi_smoothing: RSI平滑周期
    :param qqe_factor: QQE因子
    :return: (trend_line, smooth_rsi)
    """
    # 1. 计算RSI
    rsi = ta.momentum.RSIIndicator(series, window=rsi_length).rsi()
    # 2. 平滑RSI（EMA）
    smooth_rsi = ta.trend.EMAIndicator(rsi, window=rsi_smoothing).ema_indicator()
    # 3. 计算ATR RSI
    atr_rsi = abs(smooth_rsi.shift(1) - smooth_rsi)
    # 4. Wilders周期（len*2-1）
    wilders_len = rsi_length * 2 - 1
    smooth_atr_rsi = ta.trend.EMAIndicator(atr_rsi, window=wilders_len).ema_indicator()
    # 5. 动态ATR RSI
    dynamic_atr_rsi = smooth_atr_rsi * qqe_factor
    
    # 初始化长/短带和趋势方向
    long_band = np.zeros(len(series))
    short_band = np.zeros(len(series))
    trend_dir = np.zeros(len(series))
    
    # 逐行计算长/短带和趋势
    for i in range(1, len(series)):
        new_short_band = smooth_rsi.iloc[i] + dynamic_atr_rsi.iloc[i]
        new_long_band = smooth_rsi.iloc[i] - dynamic_atr_rsi.iloc[i]
        
        # 更新长带
        if smooth_rsi.iloc[i-1] > long_band[i-1] and smooth_rsi.iloc[i] > long_band[i-1]:
            long_band[i] = max(long_band[i-1], new_long_band)
        else:
            long_band[i] = new_long_band
        
        # 更新短带
        if smooth_rsi.iloc[i-1] < short_band[i-1] and smooth_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short_band)
        else:
            short_band[i] = new_short_band
        
        # 更新趋势方向
        cross_up = smooth_rsi.iloc[i] > short_band[i-1]  # 上穿短带
        cross_down = long_band[i-1] > smooth_rsi.iloc[i]  # 下穿长带
        
        if cross_up:
            trend_dir[i] = 1
        elif cross_down:
            trend_dir[i] = -1
        else:
            trend_dir[i] = trend_dir[i-1]
    
    # 趋势线 = 多头时取长带，空头时取短带
    trend_line = np.where(trend_dir == 1, long_band, short_band)
    return pd.Series(trend_line), smooth_rsi

# ==============================
# 3. 策略核心计算函数
# ==============================
def calculate_strategy_indicators(df, atr_period=9, atr_multiplier=3.9, ma_period=52,
                                  rsi_length_primary=6, rsi_smoothing_primary=5, qqe_factor_primary=3.0,
                                  rsi_length_secondary=6, rsi_smoothing_secondary=5, qqe_factor_secondary=1.61,
                                  threshold_primary=3.0, threshold_secondary=3.0,
                                  bollinger_length=50, bollinger_multiplier=0.35):
    """
    计算策略所需的所有指标，返回包含状态的DataFrame
    """
    df = df.copy()
    # ------------------------------
    # 3.1 计算SuperTrend（ta库直接调用）
    # ------------------------------
    st = ta.volatility.SuperTrendIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_period,
        multiplier=atr_multiplier
    )
    df['st_line'] = st.super_trend()  # SuperTrend线
    df['st_trend'] = np.where(df['close'] > df['st_line'], 1, -1)  # 1=多头，-1=空头
    # SuperTrend信号（趋势反转）
    df['st_buy_signal'] = (df['st_trend'] == 1) & (df['st_trend'].shift(1) == -1)
    df['st_sell_signal'] = (df['st_trend'] == -1) & (df['st_trend'].shift(1) == 1)

    # ------------------------------
    # 3.2 计算Heikin Ashi + EMA（简化版MA，优先用ta库的EMA）
    # ------------------------------
    df = calculate_heikin_ashi(df)
    # 计算HA各价格的EMA（简化Pine Script中的多MA类型，优先用EMA）
    df['ma_ha_open'] = ta.trend.EMAIndicator(df['ha_open'], window=ma_period).ema_indicator()
    df['ma_ha_close'] = ta.trend.EMAIndicator(df['ha_close'], window=ma_period).ema_indicator()
    df['ma_ha_high'] = ta.trend.EMAIndicator(df['ha_high'], window=ma_period).ema_indicator()
    df['ma_ha_low'] = ta.trend.EMAIndicator(df['ha_low'], window=ma_period).ema_indicator()
    # 计算趋势指标A的状态
    df['trend_ind_val'] = 100 * (df['ma_ha_close'] - df['ma_ha_open']) / (df['ma_ha_high'] - df['ma_ha_low'])
    df['tia_bullish'] = df['trend_ind_val'] > 0  # TIA看涨
    df['tia_bearish'] = df['trend_ind_val'] <= 0  # TIA看跌
    # 高低线（止损用）
    df['highest_body_line'] = df[['ma_ha_open', 'ma_ha_close']].max(axis=1)
    df['lowest_body_line'] = df[['ma_ha_open', 'ma_ha_close']].min(axis=1)

    # ------------------------------
    # 3.3 计算QQE MOD指标
    # ------------------------------
    # 主QQE
    prim_trend_line, prim_rsi = calculate_qqe_mod(
        df['close'],
        rsi_length=rsi_length_primary,
        rsi_smoothing=rsi_smoothing_primary,
        qqe_factor=qqe_factor_primary
    )
    # 副QQE
    _, sec_rsi = calculate_qqe_mod(
        df['close'],
        rsi_length=rsi_length_secondary,
        rsi_smoothing=rsi_smoothing_secondary,
        qqe_factor=qqe_factor_secondary
    )
    # 布林带（基于主QQE）
    bb_basis = ta.trend.SMAIndicator(prim_trend_line - 50, window=bollinger_length).sma_indicator()
    bb_dev = bollinger_multiplier * ta.volatility.StandardDeviation(prim_trend_line - 50, window=bollinger_length).standard_deviation()
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    # QQE状态
    df['qqe_bullish'] = (sec_rsi - 50 > threshold_secondary) & (prim_rsi - 50 > bb_upper)
    df['qqe_bearish'] = (sec_rsi - 50 < -threshold_secondary) & (prim_rsi - 50 < bb_lower)

    # ------------------------------
    # 3.4 综合状态（三者同向）
    # ------------------------------
    df['concrete_bullish'] = (df['st_trend'] == 1) & df['tia_bullish'] & df['qqe_bullish']
    df['concrete_bearish'] = (df['st_trend'] == -1) & df['tia_bearish'] & df['qqe_bearish']

    return df

def datalod():
    sym = 'ETHUSDT'
    data_dir = '/Users/aming/data/ETHUSDT/15m'
    start_date_train = '2025-01-01'
    end_date_train = '2025-06-01'
    start_date_test = '2025-06-01'
    end_date_test = '2025-09-01'
    timeframe =  '15m'

    
    df = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
                            timeframe=timeframe, file_path=None)

    column_mapping = {
        'o':'open',
        'h':'high',
        'l':'low',
        'c':'close',
        'vol':'volume',
        'vol_ccy':'quote_av',
        'trades':'count',
        'close_time':'close_time',
        'taker_buy_volume':'tb_base_av',
        'taker_buy_quote_volume':'tb_quote_av'

    }
    df = df.rename(columns=column_mapping)
    return df
# ==============================
# 4. 多时间周期策略整合
# ==============================
def run_multi_timeframe_strategy():
    """
    运行多时间周期策略（4H判断大趋势，1H入场）
    """
    # 1. 获取数据
    # 1H数据（入场周期）
    df_1h = fetch_binance_futures_data(symbol="ETH/USDT", timeframe="1h", limit=1000)
    # 4H数据（趋势判断周期）
    df_4h = fetch_binance_futures_data(symbol="ETH/USDT", timeframe="4h", limit=500)
    
    # 2. 计算各周期指标
    df_1h = calculate_strategy_indicators(df_1h)
    df_4h = calculate_strategy_indicators(df_4h)
    
    # 3. 对齐时间戳（将4H趋势状态映射到1H数据）
    # 转换时间戳为小时级，方便对齐
    df_1h['hour'] = df_1h['timestamp'].dt.floor('4H')  # 1H数据映射到所属的4H周期
    df_4h['hour'] = df_4h['timestamp'].dt.floor('4H')  # 4H数据的周期起始时间
    
    # 提取4H的核心状态（concrete_bullish/concrete_bearish）
    htf_status = df_4h[['hour', 'concrete_bullish', 'concrete_bearish']].set_index('hour')
    # 合并到1H数据中
    df_1h = df_1h.merge(htf_status, on='hour', how='left', suffixes=('', '_4h'))
    
    # 4. 生成最终交易信号
    # 买入信号：1H综合看涨 + 4H综合看涨
    df_1h['buy_signal'] = df_1h['concrete_bullish'] & df_1h['concrete_bullish_4h']
    # 卖出信号：1H综合看跌 + 4H综合看跌
    df_1h['sell_signal'] = df_1h['concrete_bearish'] & df_1h['concrete_bearish_4h']
    
    # 5. 输出结果（展示最后20条信号）
    print("=== 1H周期交易信号（最后20条）===")
    result_cols = ['timestamp', 'close', 'concrete_bullish', 'concrete_bullish_4h', 
                   'buy_signal', 'concrete_bearish', 'concrete_bearish_4h', 'sell_signal']
    print(df_1h[result_cols].tail(20))
    
    # 6. 统计信号数量
    print(f"\n=== 信号统计 ===")
    print(f"买入信号总数: {df_1h['buy_signal'].sum()}")
    print(f"卖出信号总数: {df_1h['sell_signal'].sum()}")
    
    return df_1h

# ==============================
# 5. 运行策略
# ==============================
if __name__ == "__main__":
    # 运行多时间周期策略
    strategy_result = run_multi_timeframe_strategy()