import pandas as pd
import numpy as np
import ta
from ta.trend import EMAIndicator, SMAIndicator, WMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from dataload import data_load_v2

sym = 'ETHUSDT'
data_dir = '/Users/aming/data/ETHUSDT/15m'
start_date_train = '2025-01-01'
end_date_train = '2025-06-01'
start_date_test = '2025-06-01'
end_date_test = '2025-09-01'
timeframe =  '15m'

df_15min = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
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
df_15min = df_15min.rename(columns=column_mapping)

# ==============================
# 1. 数据准备（模拟数据/真实数据）
# ==============================
def get_sample_data():
    """生成模拟的1小时ETH-USDT数据（可直接运行测试）"""
    date_range = pd.date_range(start='2025-01-01', end='2025-12-01', freq='1H')
    n = len(date_range)
    
    # 模拟OHLCV数据（带随机趋势，更贴近真实行情）
    np.random.seed(42)
    base_price = 2000
    close = base_price + np.cumsum(np.random.randn(n) * 2)  # 带趋势的收盘价
    high = close + np.random.rand(n) * 5  # 最高价
    low = close - np.random.rand(n) * 5   # 最低价
    open_ = np.roll(close, 1)            # 开盘价（前一根收盘价）
    open_[0] = close[0]
    volume = np.random.randint(1000, 10000, n)  # 成交量
    
    df = pd.DataFrame({
        'datetime': date_range,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('datetime', inplace=True)
    return df

# ==============================
# 2. 核心指标实现（重点：手动实现SuperTrend）
# ==============================
def calculate_supertrend(df, period=9, multiplier=3.9, change_atr=True):
    """
    手动实现SuperTrend（完全对齐原Pine Script逻辑）
    :param df: 包含open/high/low/close的DataFrame
    :param period: ATR周期（原代码默认9）
    :param multiplier: ATR乘数（原代码默认3.9）
    :param change_atr: True=用原生ATR，False=用TR的SMA（原代码参数）
    :return: 包含st_trend/st_bullish/st_bearish的DataFrame
    """
    df_copy = df.copy()
    
    # 1. 计算ATR（对应原代码的atr_val）
    if change_atr:
        # 原生ATR（Wilder式，EMA平滑）
        atr = AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=period)
        atr_val = atr.average_true_range()
    else:
        # TR的SMA（简单移动平均）
        tr = ta.volatility.TRIndicator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close']).true_range()
        atr_val = SMAIndicator(close=tr, window=period).sma_indicator()
    
    # 2. 计算中间价（对应原代码的src_val = (high+low)/2）
    src_val = (df_copy['high'] + df_copy['low']) / 2
    
    # 3. 计算初始上轨(up)和下轨(dn)
    up = src_val - multiplier * atr_val
    dn = src_val + multiplier * atr_val
    
    # 4. 轨线平滑（核心：和原Pine Script逻辑一致）
    up1 = up.shift(1).fillna(up)  # 前一根上轨，nz(up[1], up)
    dn1 = dn.shift(1).fillna(dn)  # 前一根下轨，nz(dn[1], dn)
    
    # 调整上轨：前收盘价>前上轨 → 取当前up和前up1的最大值，否则保持当前up
    up_adjusted = np.where(df_copy['close'].shift(1) > up1, np.maximum(up, up1), up)
    # 调整下轨：前收盘价<前下轨 → 取当前dn和前dn1的最小值，否则保持当前dn
    dn_adjusted = np.where(df_copy['close'].shift(1) < dn1, np.minimum(dn, dn1), dn)
    
    # 5. 判断趋势（trend=1看涨，trend=-1看跌）
    trend = np.ones(len(df_copy))  # 初始默认看涨
    for i in range(1, len(df_copy)):
        # 继承前一根趋势
        trend[i] = trend[i-1]
        # 趋势反转规则：
        # 前趋势看跌(-1) + 当前收盘价>前下轨 → 转看涨(1)
        if trend[i-1] == -1 and df_copy['close'].iloc[i] > dn1.iloc[i]:
            trend[i] = 1
        # 前趋势看涨(1) + 当前收盘价<前上轨 → 转看跌(-1)
        elif trend[i-1] == 1 and df_copy['close'].iloc[i] < up1.iloc[i]:
            trend[i] = -1
    
    # 6. 封装结果
    df_copy['st_up'] = up_adjusted
    df_copy['st_dn'] = dn_adjusted
    df_copy['st_trend'] = trend  # 1=看涨，-1=看跌
    df_copy['st_bullish'] = df_copy['st_trend'] == 1
    df_copy['st_bearish'] = df_copy['st_trend'] == -1
    
    return df_copy

def calculate_heikin_ashi(df):
    """计算Heikin-Ashi（平均K线），修复KeyError + 优化赋值警告"""
    ha_df = df.copy(deep=True)
    
    # 2. 计算HA收盘价（先添加列，避免后续KeyError）
    ha_df['ha_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    
    # 3. 计算HA开盘价（用loc赋值，避免iloc的警告）
    ha_df['ha_open'] = np.nan
    ha_df.loc[ha_df.index[0], 'ha_open'] = (ha_df.loc[ha_df.index[0], 'open'] + ha_df.loc[ha_df.index[0], 'close']) / 2
    for i in range(1, len(ha_df)):
        prev_idx = ha_df.index[i-1]
        curr_idx = ha_df.index[i]
        ha_df.loc[curr_idx, 'ha_open'] = (ha_df.loc[prev_idx, 'ha_open'] + ha_df.loc[prev_idx, 'ha_close']) / 2
    
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return ha_df


def calculate_custom_ma(series, ma_type='EMA', period=52):
    """计算自定义MA（适配原策略的Trend Indicator A）"""
    if ma_type == 'EMA':
        return EMAIndicator(close=series, window=period).ema_indicator()
    elif ma_type == 'SMA':
        return SMAIndicator(close=series, window=period).sma_indicator()
    elif ma_type == 'WMA':
        return WMAIndicator(close=series, window=period).wma()
    else:
        return EMAIndicator(close=series, window=period).ema_indicator()

def calculate_qqe_mod(df, rsi_length=6, rsi_smoothing=5, qqe_factor=3.0, threshold=3.0, bb_length=50, bb_multiplier=0.35):
    """计算QQE MOD指标（最终修正版：解决StandardDeviation不存在+ndarray转Series问题）"""
    # 1. RSI计算与平滑（输入是df['close']，本身是Series，无问题）
    rsi = RSIIndicator(close=df['close'], window=rsi_length).rsi()
    smooth_rsi = EMAIndicator(close=rsi, window=rsi_smoothing).ema_indicator()
    
    # 2. 动态ATR-RSI计算
    atr_rsi = abs(smooth_rsi.shift(1) - smooth_rsi)
    wilders_len = rsi_length * 2 - 1
    smooth_atr_rsi = EMAIndicator(close=atr_rsi, window=wilders_len).ema_indicator()
    dynamic_atr_rsi = smooth_atr_rsi * qqe_factor
    
    # 3. QQE轨线与趋势方向（初始化用np.zeros，但后续转Series）
    long_band = np.zeros(len(df))
    short_band = np.zeros(len(df))
    trend_dir = np.zeros(len(df))
    
    for i in range(1, len(df)):
        new_short_band = smooth_rsi.iloc[i] + dynamic_atr_rsi.iloc[i]
        new_long_band = smooth_rsi.iloc[i] - dynamic_atr_rsi.iloc[i]
        
        # 轨线平滑
        if smooth_rsi.iloc[i-1] > long_band[i-1] and smooth_rsi.iloc[i] > long_band[i-1]:
            long_band[i] = max(long_band[i-1], new_long_band)
        else:
            long_band[i] = new_long_band
            
        if smooth_rsi.iloc[i-1] < short_band[i-1] and smooth_rsi.iloc[i] < short_band[i-1]:
            short_band[i] = min(short_band[i-1], new_short_band)
        else:
            short_band[i] = new_short_band
        
        # 趋势判断
        cross_long = (long_band[i-1] > smooth_rsi.iloc[i]) and (long_band[i-2] <= smooth_rsi.iloc[i-1])
        cross_short = (smooth_rsi.iloc[i] > short_band[i-1]) and (smooth_rsi.iloc[i-1] <= short_band[i-2])
        
        if cross_short:
            trend_dir[i] = 1
        elif cross_long:
            trend_dir[i] = -1
        else:
            trend_dir[i] = trend_dir[i-1]
    
    # 4. 核心修正1：将ndarray转为pandas Series（对齐df索引）
    prim_trend = np.where(trend_dir == 1, long_band, short_band)
    prim_trend_series = pd.Series(prim_trend, index=df.index, name='prim_trend')
    prim_trend_shifted = prim_trend_series - 50  # 对应原代码的prim_trend - 50
    
    # 5. 布林带过滤（核心修正2：用pandas原生滚动标准差替代ta库不存在的类）
    bb_basis = SMAIndicator(close=prim_trend_shifted, window=bb_length).sma_indicator()
    # 修正：用pandas rolling.std()计算滚动标准差，和原逻辑一致
    bb_dev = bb_multiplier * prim_trend_shifted.rolling(window=bb_length).std(ddof=0)
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # 6. QQE看涨/看跌判断（处理空值，避免逻辑错误）
    qqe_bullish = (smooth_rsi - 50 > threshold) & (prim_trend_shifted > bb_upper)
    qqe_bearish = (smooth_rsi - 50 < -threshold) & (prim_trend_shifted < bb_lower)
    
    # 填充空值（滚动计算前bb_length个值为空，设为False）
    qqe_bullish = qqe_bullish.fillna(False)
    qqe_bearish = qqe_bearish.fillna(False)
    
    return {
        'qqe_bullish': qqe_bullish,
        'qqe_bearish': qqe_bearish
    }


# ==============================
# 3. 单周期指标整合 + 多周期处理
# ==============================
def calculate_single_tf_indicators(df, st_period=9, st_multiplier=3.9, ma_type='EMA', ma_period=52):
    """计算单周期（1H/4H）的所有指标"""
    df_copy = df.copy()
    
    # 1. SuperTrend（手动实现）
    df_copy = calculate_supertrend(df_copy, period=st_period, multiplier=st_multiplier)
    
    # 2. Trend Indicator A（Heikin-Ashi + MA）
    ha_df = calculate_heikin_ashi(df_copy)
    ha_open_ma = calculate_custom_ma(ha_df['ha_open'], ma_type, ma_period)
    ha_close_ma = calculate_custom_ma(ha_df['ha_close'], ma_type, ma_period)
    ha_high_ma = calculate_custom_ma(ha_df['ha_high'], ma_type, ma_period)
    ha_low_ma = calculate_custom_ma(ha_df['ha_low'], ma_type, ma_period)
    
    # 趋势强度计算
    trend_ind_val = 100 * (ha_close_ma - ha_open_ma) / (ha_high_ma - ha_low_ma)
    df_copy['tia_bullish'] = trend_ind_val > 0
    df_copy['tia_bearish'] = trend_ind_val <= 0
    
    # 止损用的高低线
    df_copy['curr_sl_high'] = np.maximum(ha_open_ma, ha_close_ma)
    df_copy['curr_sl_low'] = np.minimum(ha_open_ma, ha_close_ma)
    
    # 3. QQE MOD（主+次）
    qqe_primary = calculate_qqe_mod(df_copy, rsi_length=6, rsi_smoothing=5, qqe_factor=3.0, threshold=3.0)
    qqe_secondary = calculate_qqe_mod(df_copy, rsi_length=6, rsi_smoothing=5, qqe_factor=1.61, threshold=3.0)
    
    # QQE共振
    df_copy['qqe_bullish'] = qqe_primary['qqe_bullish'] & qqe_secondary['qqe_bullish']
    df_copy['qqe_bearish'] = qqe_primary['qqe_bearish'] & qqe_secondary['qqe_bearish']
    
    # 4. 多指标共振（核心：和原策略一致）
    df_copy['concrete_bull'] = df_copy['st_bullish'] & df_copy['tia_bullish'] & df_copy['qqe_bullish']
    df_copy['concrete_bear'] = df_copy['st_bearish'] & df_copy['tia_bearish'] & df_copy['qqe_bearish']
    
    return df_copy

def process_multi_tf(df_1h):
    """多周期处理：4H大趋势 + 1H入场"""
    # 1. 1H数据计算指标
    df_1h_with_ind = calculate_single_tf_indicators(df_1h)
    
    # 2. 1H重采样为4H
    df_4h = df_1h.resample('4H', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # 3. 4H数据计算指标
    df_htf_with_ind = calculate_single_tf_indicators(df_4h)
    
    # 4. 4H趋势对齐到1H（向前填充）
    htf_cols = ['concrete_bull', 'concrete_bear']

    df_htf_shifted = df_htf_with_ind[htf_cols].shift(1)
    # df_4h_aligned = df_htf_with_ind[htf_cols].shift(1)
    
    df_htf_aligned = df_htf_shifted.resample('1H').ffill()
    # .resample('1H').ffill()
    
    # 5. 合并到1H数据
    df_merged = df_1h_with_ind.join(df_htf_aligned, rsuffix='_htf')
    
    cols_to_fix = ['concrete_bull_htf', 'concrete_bear_htf']
    df_merged[cols_to_fix] = df_merged[cols_to_fix].fillna(False)

    # 6. 生成最终买卖信号
    df_merged['buy_signal'] = df_merged['concrete_bull'] & df_merged['concrete_bull_htf']
    df_merged['sell_signal'] = df_merged['concrete_bear'] & df_merged['concrete_bear_htf']
    
    return df_merged

# ==============================
# 4. 运行测试
# ==============================
if __name__ == '__main__':


    # 1. 获取模拟数据
    df_1h = df_15min.resample('1h', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


    
    # 2. 多周期处理 + 生成信号
    df_result = process_multi_tf(df_1h)
    
    # 3. 输出结果（查看信号）
    print("=== 策略信号结果（前20行）===")
    print(df_result.head(20))
    
    # 统计信号数量
    print(f"\n=== 信号统计 ===")
    print(f"买入信号总数：{df_result['buy_signal'].sum()}")
    print(f"卖出信号总数：{df_result['sell_signal'].sum()}")