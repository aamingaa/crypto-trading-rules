import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import os
from typing import Optional, List, Dict, Any
from dataload import data_load_v2

# 设置中文显示（风格参考 tools/MovingAverageStrategy.py）
plt.rcParams["font.family"] = ["Heiti TC", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 与 tools/MovingAverageStrategy.py 保持一致的默认路径（如你本机路径不同，可自行修改）
file_path = f"/Volumes/Ext-Disk/data/futures/um/daily/klines"
save_path = f"/Users/aming/project/python/crypto-workstation-v2/output/evaluate_metric"


def generate_date_range(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return date_list


def load_daily_data(start_date: str, end_date: str, interval: str, crypto: str = "BNBUSDT") -> pd.DataFrame:
    """
    读取日度切片的 K 线数据（zip CSV），并做基础清洗。
    说明：该函数复制自 tools/MovingAverageStrategy.py 的风格与逻辑，便于你直接复用现有数据落地。
    """
    date_list = generate_date_range(start_date, end_date)
    crypto_date_data = []
    suffix = "2025-01-01_2025-07-01"
    for date in date_list:
        crypto_date_data.append(pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip"))

    z = pd.concat(crypto_date_data, axis=0, ignore_index=True)
    z["open_time"] = pd.to_datetime(z["open_time"], unit="ms")
    z["close_time"] = pd.to_datetime(z["close_time"], unit="ms")

    z = z.sort_values(by="close_time", ascending=True)
    z = z.drop_duplicates("close_time").reset_index(drop=True)
    z["interval"] = interval
    return z


# ==========
# 指标基础函数（尽量对齐 TradingView ta.* 的语义）
# ==========

def _rma(x: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA（TradingView ta.rma/ta.atr 的平滑方式）。"""
    if length <= 0:
        raise ValueError("length 必须为正整数")
    alpha = 1.0 / float(length)
    return x.ewm(alpha=alpha, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range（TradingView ta.tr）。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """ATR（TradingView ta.atr）：TR 的 Wilder RMA。"""
    tr = _true_range(high, low, close)
    return _rma(tr, length)


def sma(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(window=length, min_periods=length).mean()


def ema(x: pd.Series, length: int) -> pd.Series:
    # TradingView EMA：alpha = 2/(n+1)，adjust=False
    return x.ewm(span=length, adjust=False).mean()


def wma(x: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average，权重 1..n（TradingView ta.wma）。"""
    if length <= 0:
        raise ValueError("length 必须为正整数")
    weights = np.arange(1, length + 1, dtype=np.float64)
    w_sum = weights.sum()

    def _apply(arr: np.ndarray) -> float:
        return float(np.dot(arr, weights) / w_sum)

    return x.rolling(window=length, min_periods=length).apply(_apply, raw=True)


def vwma(price: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume Weighted Moving Average（TradingView ta.vwma）。"""
    pv = (price * volume).rolling(window=length, min_periods=length).sum()
    vv = volume.rolling(window=length, min_periods=length).sum()
    return pv / vv.replace(0.0, np.nan)


def alma(x: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """ALMA（TradingView ta.alma）。"""
    if length <= 0:
        raise ValueError("length 必须为正整数")
    i = np.arange(length, dtype=np.float64)
    m = offset * (length - 1)
    s = length / float(sigma)
    w = np.exp(-((i - m) ** 2) / (2.0 * (s ** 2)))
    w /= w.sum()

    def _apply(arr: np.ndarray) -> float:
        return float(np.dot(arr, w))

    return x.rolling(window=length, min_periods=length).apply(_apply, raw=True)


def zlema(x: pd.Series, length: int) -> pd.Series:
    """ZLEMA：EMA(x + x - x[lag])，lag=floor((n-1)/2)（Pine 版写法）。"""
    lag = int(math.floor((length - 1) / 2))
    x2 = x + x - x.shift(lag)
    return ema(x2, length)


def hma(x: pd.Series, length: int) -> pd.Series:
    """HMA（TradingView ta.hma）：WMA(2*WMA(n/2) - WMA(n), sqrt(n))。"""
    n = int(length)
    half = max(int(n / 2), 1)
    sqrt_n = max(int(math.sqrt(n)), 1)
    return wma(2 * wma(x, half) - wma(x, n), sqrt_n)


def swma(x: pd.Series) -> pd.Series:
    """
    SWMA（TradingView ta.swma）：固定长度的对称加权移动平均线（4）。

    与 Pine/TradingView 精确对齐的定义：
    swma(x)[t] = x[t-3] * 1/6 + x[t-2] * 2/6 + x[t-1] * 2/6 + x[t] * 1/6

    等价 Pine（你给的示例）：
    pine_swma(x) => x[3]*1/6 + x[2]*2/6 + x[1]*2/6 + x[0]*1/6
    """
    weights = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float64)
    w_sum = weights.sum()

    def _apply(arr: np.ndarray) -> float:
        return float(np.dot(arr, weights) / w_sum)

    return x.rolling(window=4, min_periods=4).apply(_apply, raw=True)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成 Heikin Ashi K 线（等价于 Pine: ticker.heikinashi + request.security(... lookahead_off) 在同周期下的可观测结果）。
    注意：HA open 是递推的，所以这里必须顺序计算（无未来函数）。
    """
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.zeros_like(ha_close)
    ha_high = np.zeros_like(ha_close)
    ha_low = np.zeros_like(ha_close)

    # 初始值：常见做法用 (open+close)/2
    ha_open[0] = (o[0] + c[0]) / 2.0
    ha_high[0] = max(h[0], ha_open[0], ha_close[0])
    ha_low[0] = min(l[0], ha_open[0], ha_close[0])

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])

    out = pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        },
        index=df.index,
    )
    return out


def apply_ma(
    x: pd.Series,
    ma_type: str,
    length: int,
    alma_offset: float = 0.85,
    alma_sigma: float = 6.0,
    volume: Optional[pd.Series] = None,
) -> pd.Series:
    """
    对齐 Pine 版本 switch 的 MA 选择。
    - WMA：修正为 wma（Pine 注释里提到原版误用 vwma）
    - ZLEMA：ema(x + x - x[lag], length)
    - SWMA：固定 4 根对称权重
    """
    t = (ma_type or "EMA").upper()
    if t == "ALMA":
        return alma(x, length, offset=alma_offset, sigma=float(alma_sigma))
    if t == "HMA":
        return hma(x, length)
    if t == "SMA":
        return sma(x, length)
    if t == "SWMA":
        return swma(x)
    if t == "VWMA":
        if volume is None:
            raise ValueError("VWMA 需要 volume 序列")
        return vwma(x, volume, length)
    if t == "WMA":
        return wma(x, length)
    if t == "ZLEMA":
        return zlema(x, length)
    # 默认 EMA
    return ema(x, length)


def infer_bars_per_day(interval: str) -> float:
    """
    用 interval 推断日内 bar 数，用于年化 Sharpe。
    仅覆盖常见频率；如果你用的是更复杂的频率（如 45m），建议手动传 bars_per_day。
    """
    s = (interval or "").strip().lower()
    if s.endswith("m"):
        minutes = float(s[:-1])
        return 1440.0 / minutes
    if s.endswith("h"):
        hours = float(s[:-1])
        return 24.0 / hours
    if s.endswith("d"):
        days = float(s[:-1]) if s[:-1] else 1.0
        return 1.0 / days
    # 兜底：按 15m
    return 96.0


# ==========
# 核心：Supertrend + Trend Indicator A 分析器（带回测）
# ==========

class SupertrendTIAAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        crypto: Optional[str] = None,
        commission_rate: float = 0.0001,
        # Supertrend
        st_periods: int = 10,
        st_multiplier: float = 3.0,
        st_change_atr: bool = True,
        # Trend Indicator A
        tia_ma_type: str = "EMA",
        tia_ma_period: int = 9,
        tia_alma_offset: float = 0.85,
        tia_alma_sigma: int = 6,
        volume_col: Optional[str] = None,
        # 回测行为
        allow_short: bool = False,
    ):
        """
        Supertrend + Trend Indicator A 指标 & 回测分析器（风格参考 tools/MovingAverageStrategy.py）

        Parameters
        ----------
        data : pd.DataFrame
            需要包含: open_time, close_time, open, high, low, close
            如使用 VWMA，还需要 volume 列（可用 volume_col 指定）
        commission_rate : float
            手续费率（双向收取）
        st_change_atr : bool
            True: 使用 ATR（Wilder RMA），False: 使用 SMA(TR)（对应 Pine: st_changeATR ? ta.atr : ta.sma(ta.tr)）
        allow_short : bool
            是否允许做空（默认 False：仅做多，buy 开仓，sell 平仓）
        """
        self.data = data.copy()
        self.crypto = crypto
        self.commission_rate = float(commission_rate)

        self.st_periods = int(st_periods)
        self.st_multiplier = float(st_multiplier)
        self.st_change_atr = bool(st_change_atr)

        self.tia_ma_type = tia_ma_type
        self.tia_ma_period = int(tia_ma_period)
        self.tia_alma_offset = float(tia_alma_offset)
        self.tia_alma_sigma = int(tia_alma_sigma)

        self.allow_short = bool(allow_short)
        self.volume_col = volume_col

        # ---- 标准化时间列（兼容 open_time 已经是 index 的情况）
        # dataload.data_load_v2() 会把 open_time 设为 index，并且返回的 columns 不含 open_time
        if "open_time" not in self.data.columns:
            # 如果 index 已是 DatetimeIndex/时间戳索引，这里直接补成一列
            self.data["open_time"] = self.data.index

        # open_time：尽量转成 datetime
        self.data["open_time"] = pd.to_datetime(self.data["open_time"], errors="coerce")

        # close_time：有些数据是毫秒时间戳（int），pd.to_datetime 默认按 ns 会错，这里做一下判断
        if "close_time" in self.data.columns:
            ct = self.data["close_time"]
            if pd.api.types.is_numeric_dtype(ct):
                self.data["close_time"] = pd.to_datetime(ct, unit="ms", errors="coerce")
            else:
                self.data["close_time"] = pd.to_datetime(ct, errors="coerce")

        # 统一：按 open_time 排序，并把 open_time 设为 index（同时保留列，方便后续 df["open_time"].iloc[i] 使用）
        # self.data = self.data.sort_values("open_time")
        self.data = self.data.set_index("open_time", drop=False)

        # 初始化列
        self.data["signal"] = 0
        self.data["position"] = 0
        self.data["pnl"] = 0.0
        self.data["cumulative_pnl"] = 0.0
        self.data["strategy_return"] = 0.0
        self.data["cumulative_strategy"] = 1.0
        self.data["market_return"] = 0.0
        self.data["cumulative_market"] = 1.0

        self.trades: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.total_pnl: float = 0.0

    def _get_volume_series(self) -> Optional[pd.Series]:
        if self.volume_col and self.volume_col in self.data.columns:
            return self.data[self.volume_col].astype(float)
        for cand in ["volume", "vol", "quote_volume", "vol_ccy"]:
            if cand in self.data.columns:
                return self.data[cand].astype(float)
        return None

    def compute_indicators(self):
        """计算 Supertrend 与 Trend Indicator A（不生成交易信号）。"""
        df = self.data

        # ---- [1] Supertrend
        src = (df["high"].astype(float) + df["low"].astype(float)) / 2.0  # hl2
        tr = _true_range(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float))
        atr_sma_tr = tr.rolling(window=self.st_periods, min_periods=self.st_periods).mean()
        atr_wilder = atr(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float), self.st_periods)
        st_atr = atr_wilder if self.st_change_atr else atr_sma_tr

        basic_up = src - (self.st_multiplier * st_atr)
        basic_dn = src + (self.st_multiplier * st_atr)

        up = basic_up.values.astype(float)
        dn = basic_dn.values.astype(float)
        close = df["close"].astype(float).values

        # 递归更新（对齐 Pine 的 := 写法）
        for i in range(1, len(df)):
            up1 = up[i - 1] if np.isfinite(up[i - 1]) else up[i]
            dn1 = dn[i - 1] if np.isfinite(dn[i - 1]) else dn[i]

            if close[i - 1] > up1:
                up[i] = max(up[i], up1)
            # else: up[i] 保持 basic_up[i]

            if close[i - 1] < dn1:
                dn[i] = min(dn[i], dn1)
            # else: dn[i] 保持 basic_dn[i]

        trend = np.ones(len(df), dtype=np.int8)  # var int st_trend = 1
        for i in range(1, len(df)):
            up1 = up[i - 1] if np.isfinite(up[i - 1]) else up[i]
            dn1 = dn[i - 1] if np.isfinite(dn[i - 1]) else dn[i]
            prev = trend[i - 1]
            if prev == -1 and close[i] > dn1:
                trend[i] = 1
            elif prev == 1 and close[i] < up1:
                trend[i] = -1
            else:
                trend[i] = prev

        df["st_atr"] = st_atr
        df["st_up"] = up
        df["st_dn"] = dn
        df["st_trend"] = trend
        df["st_buy_signal"] = (df["st_trend"] == 1) & (df["st_trend"].shift(1) == -1)
        df["st_sell_signal"] = (df["st_trend"] == -1) & (df["st_trend"].shift(1) == 1)
        df["st_line"] = np.where(df["st_trend"] == 1, df["st_up"], df["st_dn"])

        # ---- [2] Trend Indicator A（Heikin Ashi + MA）
        ha = heikin_ashi(df)
        vol = self._get_volume_series()

        df["tia_ha_open"] = ha["ha_open"]
        df["tia_ha_high"] = ha["ha_high"]
        df["tia_ha_low"] = ha["ha_low"]
        df["tia_ha_close"] = ha["ha_close"]

        df["tia_ma_ha_open"] = apply_ma(
            df["tia_ha_open"],
            self.tia_ma_type,
            self.tia_ma_period,
            alma_offset=self.tia_alma_offset,
            alma_sigma=float(self.tia_alma_sigma),
            volume=vol,
        )
        df["tia_ma_ha_close"] = apply_ma(
            df["tia_ha_close"],
            self.tia_ma_type,
            self.tia_ma_period,
            alma_offset=self.tia_alma_offset,
            alma_sigma=float(self.tia_alma_sigma),
            volume=vol,
        )
        df["tia_ma_ha_high"] = apply_ma(
            df["tia_ha_high"],
            self.tia_ma_type,
            self.tia_ma_period,
            alma_offset=self.tia_alma_offset,
            alma_sigma=float(self.tia_alma_sigma),
            volume=vol,
        )
        df["tia_ma_ha_low"] = apply_ma(
            df["tia_ha_low"],
            self.tia_ma_type,
            self.tia_ma_period,
            alma_offset=self.tia_alma_offset,
            alma_sigma=float(self.tia_alma_sigma),
            volume=vol,
        )

        denom = (df["tia_ma_ha_high"] - df["tia_ma_ha_low"]).replace(0.0, np.nan)
        df["tia_trend"] = 100.0 * (df["tia_ma_ha_close"] - df["tia_ma_ha_open"]) / denom
        df["tia_trend"] = df["tia_trend"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.data = df
        return df

    def generate_signals(self):
        """
        生成交易信号（默认使用 Supertrend 的趋势翻转信号），并按“下一根K线开盘执行”回测。

        Pine 对齐说明：
        - st_buySignal / st_sellSignal 在当前 bar close 时刻可观察
        - 这里采用 next bar open 执行，避免未来函数
        """
        if "st_buy_signal" not in self.data.columns:
            self.compute_indicators()

        df = self.data
        df["signal"] = 0
        df.loc[df["st_buy_signal"], "signal"] = 1
        df.loc[df["st_sell_signal"], "signal"] = -1

        # 这里一定要用“按位置”的方式处理，避免 df 的 index 不是 0..N-1 时 df.at[i] 意外扩表
        exec_signal = df["signal"].shift(1).fillna(0).astype(int).to_numpy()

        position = [0]
        self.trades = []
        self.total_pnl = 0.0
        pnl = np.zeros(len(df), dtype=float)

        # position 语义：
        # - allow_short=False：0/1（空仓/持多）
        # - allow_short=True：-1/0/1（持空/空仓/持多）
        for i in range(1, len(df)):
            prev_pos = int(position[-1])
            open_price = float(df["open"].iloc[i])
            open_time = df["open_time"].iloc[i]
            current_pnl = 0.0

            # 开多
            if exec_signal[i] == 1 and prev_pos <= 0:
                # 如果允许做空且当前是空头，则先平空再开多（这里简化为直接翻仓）
                if self.allow_short and prev_pos == -1:
                    # 平空：盈亏 = 开仓价 - 平仓价
                    last_entry = self.trades[-1] if self.trades else None
                    gross_pnl = 0.0
                    if last_entry and last_entry.get("type") == "sell_short":
                        gross_pnl = last_entry["price"] - open_price
                    commission = open_price * self.commission_rate
                    net_pnl = gross_pnl - commission
                    self.total_pnl += net_pnl
                    current_pnl += net_pnl
                    self.trades.append(
                        {
                            "type": "buy_to_cover",
                            "time": open_time,
                            "price": open_price,
                            "commission": commission,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "index": i,
                        }
                    )

                position.append(1)
                commission = open_price * self.commission_rate
                current_pnl -= commission
                self.trades.append(
                    {
                        "type": "buy",
                        "time": open_time,
                        "price": open_price,
                        "commission": commission,
                        "index": i,
                    }
                )

            # 平多 / 开空
            elif exec_signal[i] == -1 and prev_pos >= 0:
                if prev_pos == 1:
                    # 平多
                    commission = open_price * self.commission_rate
                    last_buy = None
                    for t in reversed(self.trades):
                        if t.get("type") == "buy":
                            last_buy = t
                            break
                    gross_pnl = (open_price - last_buy["price"]) if last_buy else 0.0
                    net_pnl = gross_pnl - commission
                    self.total_pnl += net_pnl
                    current_pnl += net_pnl
                    self.trades.append(
                        {
                            "type": "sell",
                            "time": open_time,
                            "price": open_price,
                            "commission": commission,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "holding_period": (i - last_buy["index"]) if last_buy else 0,
                            "index": i,
                        }
                    )

                if self.allow_short:
                    # 开空
                    position.append(-1)
                    commission = open_price * self.commission_rate
                    current_pnl -= commission
                    self.trades.append(
                        {
                            "type": "sell_short",
                            "time": open_time,
                            "price": open_price,
                            "commission": commission,
                            "index": i,
                        }
                    )
                else:
                    position.append(0)
            else:
                position.append(prev_pos)

            # 注意：用 numpy 数组按位置写入，避免 df.at[i] 在 DatetimeIndex 下创建新行
            pnl[i] = current_pnl

        df["pnl"] = pnl
        df["position"] = position
        df["cumulative_pnl"] = df["pnl"].cumsum()

        df["market_return"] = df["open"].pct_change()
        df["strategy_return"] = df["market_return"] * df["position"].shift(1).fillna(0)
        df["cumulative_market"] = (1 + df["market_return"].fillna(0)).cumprod()
        df["cumulative_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()

        self.data = df
        return df

    def calculate_performance(self):
        """计算核心绩效指标（与 tools/MovingAverageStrategy.py 口径一致）。"""
        df = self.data

        # 完整平仓交易
        close_types = {"sell"}
        if self.allow_short:
            close_types = {"sell", "buy_to_cover"}
        close_trades = [t for t in self.trades if t.get("type") in close_types]

        profitable = [t for t in close_trades if t.get("net_pnl", 0.0) > 0]
        total_closes = len(close_trades)
        win_rate = len(profitable) / total_closes if total_closes > 0 else 0.0

        total_return = df["cumulative_strategy"].iloc[-1] - 1 if len(df) > 0 else 0.0

        returns = df["strategy_return"].dropna()
        interval = str(df["interval"].iloc[0]) if "interval" in df.columns and len(df) > 0 else "15m"
        annualization = infer_bars_per_day(interval) * 365.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = math.sqrt(annualization) * (returns.mean() / returns.std())
        else:
            sharpe = 0.0

        cumulative = df["cumulative_strategy"].dropna()
        if len(cumulative) > 0:
            peak = cumulative.expanding(min_periods=1).max()
            dd = (cumulative / peak) - 1.0
            max_dd = float(dd.min())
        else:
            max_dd = 0.0

        max_pnl = max([t.get("net_pnl", 0.0) for t in close_trades]) if close_trades else 0.0
        min_pnl = min([t.get("net_pnl", 0.0) for t in close_trades]) if close_trades else 0.0
        total_commission = sum(float(t.get("commission", 0.0)) for t in self.trades)

        self.performance_metrics = {
            "总PNL": float(self.total_pnl),
            "最大盈利PNL": float(max_pnl),
            "最大亏损PNL": float(min_pnl),
            "累计收益率": float(total_return),
            "夏普比率": float(sharpe),
            "最大回撤": float(max_dd),
            "胜率": float(win_rate),
            "总交易次数": int(total_closes),
            "总手续费支出": float(total_commission),
            "symbol": self.crypto,
        }
        return self.performance_metrics

    def plot_results(self, save_path: Optional[str] = None, show: bool = False):
        """绘制：价格+Supertrend、净值曲线、TIA trend。"""
        df = self.data

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), sharex=True)
        plt.ion()

        # 1) 价格 + Supertrend 线
        ax1.plot(df["close_time"], df["close"], label="收盘价", linewidth=1.2, color="black", alpha=0.9)

        if "st_up" in df.columns and "st_dn" in df.columns:
            bull = df["st_trend"] == 1
            bear = df["st_trend"] == -1
            ax1.plot(df.loc[bull, "close_time"], df.loc[bull, "st_up"], label="ST 牛线", linewidth=1.8, color="green")
            ax1.plot(df.loc[bear, "close_time"], df.loc[bear, "st_dn"], label="ST 熊线", linewidth=1.8, color="red")

        # 标注交易点
        buys = [t for t in self.trades if t.get("type") == "buy"]
        sells = [t for t in self.trades if t.get("type") == "sell"]
        if buys:
            ax1.scatter([t["time"] for t in buys], [t["price"] for t in buys], marker="^", s=70, color="green", label="Buy")
        if sells:
            ax1.scatter([t["time"] for t in sells], [t["price"] for t in sells], marker="v", s=70, color="red", label="Sell")

        ax1.set_title("Supertrend + 交易信号")
        ax1.grid(True)
        ax1.legend()

        # 2) 净值曲线
        ax2.plot(df["close_time"], df["cumulative_strategy"], label="策略累计收益", linewidth=1.5)
        ax2.plot(df["close_time"], df["cumulative_market"], label="市场累计收益", linewidth=1.5, alpha=0.8)
        ax2.set_title("累计收益对比")
        ax2.grid(True)
        ax2.legend()

        # 3) TIA trend
        if "tia_trend" in df.columns:
            ax3.plot(df["close_time"], df["tia_trend"], label="TIA Trend", linewidth=1.2, color="#26A69A")
            ax3.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
        ax3.set_title("Trend Indicator A（Heikin Ashi + MA）")
        ax3.grid(True)
        ax3.legend()

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def run_strategy(self, save_plot_path: Optional[str] = None):
        """一键运行：计算指标 -> 生成信号&回测 -> 输出绩效 -> 绘图。"""
        self.compute_indicators()
        self.generate_signals()
        metrics = self.calculate_performance()

        print("策略参数:")
        print(f"Supertrend: ATR周期={self.st_periods}, ATR倍数={self.st_multiplier}, 使用ATR={self.st_change_atr}")
        print(f"TIA: MA类型={self.tia_ma_type}, MA周期={self.tia_ma_period}")
        print(f"手续费率: {self.commission_rate:.4%} (双向收取)")
        print(f"允许做空: {self.allow_short}")

        print("\n绩效指标:")
        for k in ["累计收益率", "夏普比率", "最大回撤", "胜率", "总交易次数", "总手续费支出", "总PNL"]:
            if k in ["累计收益率", "胜率", "最大回撤"]:
                print(f"{k}: {metrics[k]:.2%}")
            else:
                print(f"{k}: {metrics[k]:.6f}")

        self.plot_results(save_path=save_plot_path, show=False)
        return metrics


def generate_sample_data(n_periods: int = 2000, freq: str = "15min") -> pd.DataFrame:
    """生成带成交量的示例K线数据，便于你在无真实数据环境下快速验证逻辑。"""
    start_time = datetime(2023, 1, 1, 0, 0)
    times = pd.date_range(start=start_time, periods=n_periods, freq=freq)

    base_price = 100.0
    trend = np.linspace(0, 30, n_periods)
    noise = np.random.normal(0, 1, n_periods).cumsum()
    close = base_price + trend + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.random.uniform(0, 1, n_periods)
    low = np.minimum(open_, close) - np.random.uniform(0, 1, n_periods)
    volume = np.random.lognormal(mean=10.0, sigma=0.3, size=n_periods)

    df = pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(freq),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "interval": "15m",
        }
    )
    return df


if __name__ == "__main__":
    # ====== 你可以选择：示例数据跑通，或接入你本机的真实数据 ======
    use_sample = False

    # if use_sample:
    #     print("使用示例数据运行 Supertrend + TIA ...")
    #     df_price = generate_sample_data(n_periods=3000, freq="15min")
    #     crypto = "SAMPLE"
    # else:
    #     start_date = "2025-01-01"
    #     end_date = "2025-06-30"
    #     crypto = "ETHUSDT"
    #     df_price = load_daily_data(start_date, end_date, "15m", crypto=crypto)

    sym = 'ETHUSDT'
    data_dir = '/Users/aming/data/ETHUSDT/15m'
    start_date_train = '2025-01-01'
    end_date_train = '2025-01-01'
    start_date_test = '2025-01-01'
    end_date_test = '2025-02-01'
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

    analyzer = SupertrendTIAAnalyzer(
        df,
        crypto=sym,
        commission_rate=0.0001,
        st_periods=10,
        st_multiplier=3.0,
        st_change_atr=True,
        tia_ma_type="EMA",
        tia_ma_period=9,
        tia_alma_offset=0.85,
        tia_alma_sigma=6,
        allow_short=False,
    )

    analyzer.run_strategy(save_plot_path=f"{save_path}/supertrend_tia_{sym}.png")


