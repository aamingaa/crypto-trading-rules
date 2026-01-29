import numpy
import tensorflow as tf
import autogluon as ag

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from features import triple_barrier as tb, getTA, tautil, microstructure_features as ms
import autogluon as ag

from sklearn import preprocessing
from sklearn.decomposition import PCA 
from features import triple_barrier as tb, getTA, tautil

from scipy.stats import norm, moment

import keras

import warnings
warnings.filterwarnings(action='ignore')

def get_hourly_vol(close, span0=100):
    """
    计算小时级波动率 (基于 EWM)
    
    Parameters
    ----------
    close : pd.Series
        小时级收盘价
    span0 : int
        EWM 的跨度 (span)。
        如果是小时数据，span=100 大约代表回顾过去 4 天 (100小时) 的波动情况。
    """
    # 1. 计算小时收益率 (r_t = p_t / p_{t-1} - 1)
    # 也就是当前小时相对于上一小时的涨跌幅
    returns = close.pct_change()
    
    # 2. 计算指数加权移动标准差
    vol = returns.ewm(span=span0).std()
    
    return vol


def figure_scatter(sc,close, title, cmap='bwr', figsize=(15,5)):
    plt.figure(figsize=figsize)
    plt.plot(close, linewidth=0.5,alpha=0.6)
    plt.scatter(close.loc[sc.index].index, close.loc[sc.index], c=sc,cmap=cmap, alpha=1)
    plt.colorbar()
    plt.title(title)
    plt.savefig('image/{}.png'.format(title))
    plt.show()
    

# df_raw = pd.read_csv('ethusd5min.csv')
# df = df_raw.set_index('timestamp')
# df.index = pd.to_datetime(df.index)

from dataload import data_load_v2

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

df = df.resample('1H').agg({'open':'first',
                            'high':'max',
                            'low':'min',
                            'close':'last',
                            'volume':'sum',
                            'quote_av':'sum',
                            'count':'sum',
                            'tb_base_av':'sum',
                            'tb_quote_av':'sum',
                            'close_time':'last'
                            })

close = pd.to_numeric(df.close)
open = pd.to_numeric(df.open)
high = pd.to_numeric(df.high)
low = pd.to_numeric(df.low)
volume = pd.to_numeric(df.volume)
buy_volume = pd.to_numeric(df.tb_base_av)


selected_params = {
    'vol_span': 24,
    'vol_span_days': 1,
    'cusum_multiplier': 1,
    'profit_taking': 2.5,
    'stop_loss': 0.5,
    'max_holding_hours': 48
}  # 或者 best_return

# 使用最佳参数重新计算
hourly_vol = get_hourly_vol(close, span0=int(selected_params['vol_span']))

cusum_thresh = hourly_vol.mean() * selected_params['cusum_multiplier']
log_prices = np.log(close)
cusum_sides = tb.cusum_filter_side(log_prices, threshold=cusum_thresh)

pt_sl = [selected_params['profit_taking'], selected_params['stop_loss']]
max_holding = [0, int(selected_params['max_holding_hours'])]

barrier = tb.get_barrier_fast(close, cusum_sides.index, pt_sl, 
                               max_holding=max_holding, 
                               target=hourly_vol, side = cusum_sides)

print(barrier.head())