'''
数据读取、降频处理和计算收益率模块
'''

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import time
import talib as ta
from enum import Enum
import re

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import time
import talib as ta
from enum import Enum
import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore, kurtosis, skew, yeojohnson, boxcox
from scipy.stats import tukeylambda, mstats
from sklearn.preprocessing import RobustScaler
import zipfile
from io import BytesIO


def data_load(sym: str) -> pd.DataFrame:
    '''数据读取模块'''
    file_name = '/home/etern/crypto/data/merged/merged/' + sym + '-merged-without-rfr-1m.csv'  
    z = pd.read_csv(file_name, index_col=1)[
        ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades',
               'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
               'taker_vol_lsr']]
    return z

class DataFrequency(Enum):
    """数据频率枚举"""
    MONTHLY = 'monthly'  # 月度数据
    DAILY = 'daily'      # 日度数据


def _generate_date_range(start_date: str, end_date: str, read_frequency: DataFrequency = DataFrequency.MONTHLY) -> List[str]:
    """
    生成日期范围列表
    
    参数:
    start_date: 起始日期
        - 月度格式: 'YYYY-MM' (如 '2020-01') 或 'YYYY-MM-DD' (自动转换为 'YYYY-MM')
        - 日度格式: 'YYYY-MM-DD' (如 '2020-01-01')
    end_date: 结束日期，格式同上
    frequency: 数据频率（月度或日度）
    
    返回:
    日期字符串列表
    """
    if read_frequency == DataFrequency.MONTHLY:
        # 兼容 'YYYY-MM' 和 'YYYY-MM-DD' 两种格式
        # 如果是 'YYYY-MM-DD' 格式，自动截取为 'YYYY-MM'
        new_start_date = start_date
        new_end_date = end_date
        if len(start_date) == 10:  # 'YYYY-MM-DD' 格式
            new_start_date = start_date[:7]
        if len(end_date) == 10:
            new_end_date = end_date[:7]
            
        start_dt = datetime.strptime(new_start_date, '%Y-%m')
        end_dt = datetime.strptime(new_end_date, '%Y-%m')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m'))
            # 移动到下一个月
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return date_list
    
    elif read_frequency == DataFrequency.DAILY:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)
        
        return date_list
    
    else:
        raise ValueError(f"不支持的数据频率: {frequency}")
    

def data_load_v2(sym: str, data_dir: str, start_date: str, end_date: str, 
                 timeframe: str = '1h', read_frequency: str = 'monthly',
                 file_path: Optional[str] = None) -> pd.DataFrame:
    """
    数据读取模块 V2 - 支持从多种时间粒度的数据文件读取
    
    参数:
    sym: 交易对符号，例如 'BTCUSDT'
    data_dir: 数据目录路径，例如 '/Volumes/Ext-Disk/data/futures/um/monthly/klines/BTCUSDT/1m'
    start_date: 起始日期
        - 月度格式: 'YYYY-MM' (如 '2020-01')
        - 日度格式: 'YYYY-MM-DD' (如 '2020-01-01')
    end_date: 结束日期，格式同上
    timeframe: 时间周期，默认 '1m'，可选 '5m', '1h' 等
    frequency: 数据频率，'monthly'（月度）或 'daily'（日度）
    file_path: 直接指定文件路径（支持 .feather / .zip / .csv），指定后将忽略其他参数
    
    返回:
    包含标准化列名的 DataFrame
    
    文件读取优先级:
    1. 如果指定 file_path，直接读取该文件
    2. 否则按日期范围读取，优先读取 .feather 格式文件（如果存在）
    3. 如果 .feather 不存在，则读取 .zip 文件，并自动缓存为 .feather
    
    示例:
    # 读取月度数据
    df = data_load_v2('BTCUSDT', '/path/to/monthly', '2020-01', '2024-09', frequency='monthly')
    
    # 读取日度数据
    df = data_load_v2('BTCUSDT', '/path/to/daily', '2020-01-01', '2020-01-31', frequency='daily')
    
    # 直接读取单个文件
    df = data_load_v2('BTCUSDT', '', '', '', file_path='/path/to/data.feather')
    """
    
    # 如果指定了直接文件路径，直接读取
    # if file_path:
    #     return _read_direct_file(file_path)
    
    # 解析频率参数
    try:
        freq_enum = DataFrequency(read_frequency.lower())
    except ValueError:
        raise ValueError(f"不支持的数据频率: {read_frequency}，仅支持 'monthly' 或 'daily'")
    
    # 生成日期范围
    date_list = _generate_date_range(start_date, end_date, freq_enum)
    
    # 读取所有时间段的数据
    df_list = []
    success_count = 0
    failed_count = 0
    
    for date_str in date_list:
        df = _read_single_period_data(sym, date_str, data_dir, timeframe, freq_enum)
        if df is not None:
            df_list.append(df)
            success_count += 1
        else:
            failed_count += 1
    
    # 检查是否成功读取到数据
    if not df_list:
        raise ValueError(f"未能成功读取任何数据文件，请检查路径和日期范围\n路径: {data_dir}\n日期: {start_date} ~ {end_date}")
    
    print(f"\n{'='*60}")
    print(f"读取完成: 成功 {success_count} 个，失败 {failed_count} 个")
    print(f"{'='*60}\n")
    
    # 合并所有数据
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"合并后总行数: {len(merged_df):,}")
    
    # 标准化列名和索引
    standardized_df = _standardize_dataframe_columns(merged_df)
    
    print(f"数据时间范围: {standardized_df.index.min()} 至 {standardized_df.index.max()}")
    print(f"{'='*60}\n")
    
    return standardized_df

def _read_single_period_data(sym: str, date_str: str, data_dir: str, timeframe: str = '1m',
                             frequency: DataFrequency = DataFrequency.MONTHLY) -> Optional[pd.DataFrame]:
    """
    读取单个时间段的数据（优先 feather，其次 zip）
    
    参数:
    sym: 交易对符号
    date_str: 日期字符串
    data_dir: 数据目录
    timeframe: 时间周期
    frequency: 数据频率
    
    返回:
    DataFrame 或 None
    """
    file_base_name, feather_path, zip_path = _build_file_paths(sym, date_str, data_dir, timeframe, frequency)
    
    # 优先读取 feather
    df = _read_feather_file(feather_path)
    if df is not None:
        return df
    
    # 如果 feather 不存在，读取 zip
    df = _read_zip_file(zip_path, file_base_name, save_feather=True)
    if df is not None:
        return df
    
    # 两种文件都不存在
    print(f"⚠ 警告：文件不存在，跳过: {file_base_name}")
    return None


def _build_file_paths(sym: str, date_str: str, data_dir: str, timeframe: str = '1m', 
                      frequency: DataFrequency = DataFrequency.MONTHLY) -> Tuple[str, str, str]:
    """
    构建文件路径
    
    参数:
    sym: 交易对符号
    date_str: 日期字符串
    data_dir: 数据目录
    timeframe: 时间周期 (如 '1m', '5m', '1h')
    frequency: 数据频率
    
    返回:
    (file_base_name, feather_path, zip_path) 元组
    """
    if frequency == DataFrequency.MONTHLY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    elif frequency == DataFrequency.DAILY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    else:
        raise ValueError(f"不支持的数据频率: {frequency}")
    
    # /Volumes/Ext-Disk/data/futures/um/monthly/klines/ETHUSDT/15m/2025/ETHUSDT-15m-2025-01.feather
    year = date_str.split('-')[0]
    feather_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.feather")
    zip_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.zip")
    
    return file_base_name, feather_path, zip_path

def _read_feather_file(feather_path: str) -> Optional[pd.DataFrame]:
    """
    读取 feather 格式文件
    
    参数:
    feather_path: feather 文件路径
    
    返回:
    DataFrame 或 None（如果读取失败）
    """
    if not os.path.exists(feather_path):
        return None
    
    try:
        df = pd.read_feather(feather_path)
        print(f"✓ 成功读取 feather: {os.path.basename(feather_path)}, 行数: {len(df)}")
        return df
    except Exception as e:
        print(f"✗ 读取 feather 文件失败: {os.path.basename(feather_path)}, 错误: {str(e)}")
        return None


def _read_zip_file(zip_path: str, file_base_name: str, save_feather: bool = True) -> Optional[pd.DataFrame]:
    """
    读取 zip 格式文件（内含 CSV）
    
    参数:
    zip_path: zip 文件路径
    file_base_name: 文件基础名称（不含扩展名）
    save_feather: 是否保存为 feather 格式以加速后续读取
    
    返回:
    DataFrame 或 None（如果读取失败）
    """
    if not os.path.exists(zip_path):
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取 zip 中的 csv 文件名
            csv_filename = f"{file_base_name}.csv"
            
            if csv_filename not in zip_ref.namelist():
                # 如果找不到，尝试使用第一个 csv 文件
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    csv_filename = csv_files[0]
                else:
                    print(f"✗ 在 {os.path.basename(zip_path)} 中找不到 CSV 文件")
                    return None
            
            # 读取 CSV 数据
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                print(f"✓ 成功读取 zip: {os.path.basename(zip_path)}, 行数: {len(df)}")
                
                # 可选：保存为 feather 格式以加速后续读取
                if save_feather:
                    feather_path = zip_path.replace('.zip', '.feather')
                    try:
                        df.to_feather(feather_path)
                        print(f"  → 已缓存为 feather 格式")
                    except Exception as e:
                        print(f"  → 保存 feather 文件失败: {str(e)}")
                
                return df
    
    except Exception as e:
        print(f"✗ 读取 zip 文件失败: {os.path.basename(zip_path)}, 错误: {str(e)}")
        return None
    

def _standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 DataFrame 列名并设置索引
    
    参数:
    df: 原始 DataFrame（包含 Binance 格式的列名）
    
    返回:
    标准化后的 DataFrame
    """
    # 将 open_time 转换为 datetime 并设置为索引
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    # df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    # df.set_index('close_time', inplace=True)
    
    # 列名映射：新列名 -> 旧列名
    # 新列名: open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
    # 旧列名: o, h, l, c, vol, vol_ccy, trades, oi, oi_ccy, toptrader_count_lsr, toptrader_oi_lsr, count_lsr, taker_vol_lsr
    column_mapping = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c',
        'volume': 'vol',
        'quote_volume': 'vol_ccy',
        'count': 'trades',
        'close_time': 'close_time',
    }
    
    df = df.rename(columns=column_mapping)
    
    # 选择需要的列，对于缺失的列用 0 填充
    required_columns = [
                            'o', 'h', 'l', 'c', 
                            'vol', 
                            'vol_ccy', 
                            'trades',
                        #    'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
                        #    'taker_vol_lsr', 
                            'close_time', 
                            'taker_buy_volume', 
                            'taker_buy_quote_volume'
                       ]
    
    # 为缺失的列添加默认值 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"⚠ 警告：列 '{col}' 不存在，已填充为 0")
    
    return df[required_columns]

def removed_zero_vol_dataframe(df):
    """
    打印并且返回-
    1. volume这一列为0的行组成的df
    2. low这一列的最小值
    3. volume这一列的最小值
    5. 去除掉volume=0的行的dataframe
    -------

    """
    # 将DataFrame的索引列设置为'datetime'
    df.index = pd.to_datetime(df.index)

    # 1. volume这一列为0的行组成的df
    volume_zero_df = df[df['vol'] == 0]
    print(f"Volume为0的行组成的DataFrame: {len(volume_zero_df)}")

    # 2. low这一列的最小值
    min_low = df['l'].min()
    print(f"Low这一列的最小值: {min_low}")

    # 3. volume这一列的最小值
    min_volume = df['vol'].min()
    print(f"Volume这一列的最小值: {min_volume}")

    # 5. 去除掉volume=0的行的dataframe
    removed_zero_vol_df = df[df['vol'] != 0]
    print(f"去除掉Volume为0的行之前的DataFrame length: {len(df)}")
    print(f"去除掉Volume为0的行之后的DataFrame length: {len(removed_zero_vol_df)}")

    return removed_zero_vol_df


def resample(z: pd.DataFrame, freq: str, closed: str = 'left', label: str = 'left') -> pd.DataFrame:
    '''
    这是不支持vwap的，默认读入的数据是没有turnover信息，自然也没有vwap的信息，不需要获取sym的乘数
    '''
    if freq == '15m':
        return z
    
    if freq != '1min' or freq != '1m':
        z.index = pd.to_datetime(z.index)
        # 注意closed和label参数
        z = z.resample(freq, closed=closed, label=label).agg({'o': 'first',
                                                               'h': 'max',
                                                               'l': 'min',
                                                               'c': 'last',
                                                               'vol': 'sum',
                                                               'vol_ccy': 'sum',
                                                               'trades': 'sum',
                                                            #    'oi': 'last', 
                                                            #    'oi_ccy': 'last', 
                                                            #    'toptrader_count_lsr':'last', 
                                                            #    'toptrader_oi_lsr':'last', 
                                                            #    'count_lsr':'last',
                                                            #    'taker_vol_lsr':'last'
                                                               })
        # 注意resample后,比如以10min为resample的freq，9:00的数据是指9:00到9:10的数据~~
        z = z.fillna(method='ffill')   
        z.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy','trades',
            #    'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
            #    'taker_vol_lsr'
               ]
        
        # 重要，这个删掉0成交的操作，不能给5分钟以内的freq进行操作，因为这种情况还是挺容易出现没有成交的，这会改变本身的分布
        # 使用正则表达式提取开头的数值部分, 判断freq的周期
        match = re.match(r"(\d+)", freq)
        if match:
            int_freq = int(match.group(1))
            if int_freq > 5:
                z = removed_zero_vol_dataframe(z)
        
        return z
    
    return z


def resample_with_offset(z: pd.DataFrame, freq: str, offset: pd.Timedelta = None, 
                        closed: str = 'left', label: str = 'left') -> pd.DataFrame:
    '''
    支持offset参数的resample函数 - 使用pandas原生offset参数，避免时间索引偏移的问题
    
    参数:
        z: 输入的DataFrame，必须有DatetimeIndex
        freq: 重采样频率，如 '1h', '2h', '30min'
        offset: 偏移量（pd.Timedelta），用于调整分桶起点
                例如：offset=pd.Timedelta(minutes=15) 会让1小时桶从 9:15, 10:15, 11:15... 开始
        closed: 区间闭合方式，'left' 或 'right'
        label: 标签位置，'left' 或 'right'
    
    返回:
        重采样后的DataFrame
    '''
    if freq == '15m':
        return z
    
    if freq != '1min' and freq != '1m':
        z.index = pd.to_datetime(z.index)
        
        # 使用pandas原生的offset参数，而不是偏移索引
        if offset is not None:
            z_resampled = z.resample(
                freq, 
                closed=closed, 
                label=label,
                offset=offset  # 🔑 关键：使用pandas原生offset参数
            ).agg({
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'vol': 'sum',
                'vol_ccy': 'sum',
                'trades': 'sum',
            })
        else:
            # 没有offset时，使用原有逻辑
            z_resampled = z.resample(freq, closed=closed, label=label).agg({
                'o': 'first',
                'h': 'max',
                'l': 'min',
                'c': 'last',
                'vol': 'sum',
                'vol_ccy': 'sum',
                'trades': 'sum',
            })
        
        # 前向填充NaN值
        z_resampled = z_resampled.fillna(method='ffill')
        z_resampled.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades']
        
        return z_resampled
    
    return z
    