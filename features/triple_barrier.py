import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def add_vertical_barrier(t_events, close, days=0, hours=0, minutes=0, seconds=0):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.
    Adding a Vertical Barrier
    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.
    This function creates a series that has all the timestamps of when the vertical barrier would be reached.
    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(days, hours, minutes, seconds))
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers


def forming_barriers(close, events, pt_sl, molecule): 
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.
    Triple Barrier Labeling Method
    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.
    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.
    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['exit']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    out['pt'] = pd.Series(dtype=events.index.dtype)
    out['sl'] = pd.Series(dtype=events.index.dtype)

    # Get events
    for loc, vertical_barrier in events_['exit'].fillna(close.index[-1]).items():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out


def forming_barriers_fast(close, events, pt_sl, molecule): 
    """
    Optimized version of forming_barriers using numpy and integer indexing.
    """
    events_ = events.loc[molecule]
    out = events_[['exit']].copy()

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events_.index, data=np.inf)

    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events_.index, data=-np.inf)

    close_vals = close.values
    close_index = close.index
    
    start_indices = close_index.get_indexer(events_.index)
    end_indices = close_index.get_indexer(events_['exit'].fillna(close_index[-1]))
    
    sides = events_['side'].values
    pt_vals = profit_taking.values
    sl_vals = stop_loss.values
    
    sl_dates = [pd.NaT] * len(events_)
    pt_dates = [pd.NaT] * len(events_)

    for i in range(len(events_)):
        s_idx = start_indices[i]
        e_idx = end_indices[i]
        if s_idx < 0 or e_idx < 0:
            continue
            
        path_prices = close_vals[s_idx : e_idx + 1]
        path_returns = (path_prices / close_vals[s_idx] - 1.0) * sides[i]
        
        sl_hits = np.where(path_returns < sl_vals[i])[0]
        if len(sl_hits) > 0:
            sl_dates[i] = close_index[s_idx + sl_hits[0]]
            
        pt_hits = np.where(path_returns > pt_vals[i])[0]
        if len(pt_hits) > 0:
            pt_dates[i] = close_index[s_idx + pt_hits[0]]

    out['sl'] = sl_dates
    out['pt'] = pt_dates
    return out


def get_barrier(close, enter, pt_sl, max_holding, target=None, side=None):
    """

    :param close: (pd.Series) Close prices
    :param enter: (pd.Series) of entry points. These are timestamps that will seed every triple barrier.
        These are the timestamps
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. (if target is not None, pt_sl is ratio)
    :param target: (pd.Series) target rate
    :param max_holding: (2 element list) [days, hours]
    :param side: (pd.Series) Side of the bet (long/short) as decided by the primary model.
        1 if long, -1 if short
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['exit'] is event's endtime
            -events['side'] implies the algo's position side
            -events['ret'] is return of each bet
    """

    # 1) Get target
    if target is None:
        target_ = pd.Series(1,index=enter)
    
    else:
        target_ = target.reindex(enter)

    # 2) Get vertical barrier (max holding period)
    vertical_barrier = add_vertical_barrier(enter, close, days=max_holding[0], hours=max_holding[1])

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1.0, index=target_.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target_.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df
    events = pd.concat({'exit': vertical_barrier, 'trgt': target_,'side': side_}, axis=1)

    # Apply Triple Barrier
    first_touch_dates = forming_barriers(close, events, pt_sl_, events.index)
    

    for ind in events.index:
        events.at[ind, 'exit'] = first_touch_dates.loc[ind, :].dropna().min()

    events_x = events.dropna(subset=['exit'])

    out_df = pd.DataFrame(index=events.index)
    out_df['exit'] = events['exit']
    out_df['price'] = close
    out_df['ret'] = 0
    out_df.loc[events_x.index,'ret'] = (np.log(close.loc[events_x['exit'].array].array) - np.log(close.loc[events_x.index])) * events['side']
    out_df['side'] = events['side']
    return out_df


def get_barrier_fast(close, enter, pt_sl, max_holding, target=None, side=None):
    """
    Optimized version of get_barrier.
    """
    if target is None:
        target_ = pd.Series(1.0, index=enter)
    else:
        target_ = target.reindex(enter)

    vertical_barrier = add_vertical_barrier(enter, close, days=max_holding[0], hours=max_holding[1])

    if side is None:
        side_ = pd.Series(1.0, index=target_.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target_.index)
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'exit': vertical_barrier, 'trgt': target_,'side': side_}, axis=1)
    first_touch_dates = forming_barriers_fast(close, events, pt_sl_, events.index)
    
    events['exit'] = first_touch_dates.min(axis=1)
    events_x = events.dropna(subset=['exit'])

    out_df = pd.DataFrame(index=events.index)
    out_df['exit'] = events['exit']
    out_df['price'] = close.reindex(events.index)
    out_df['side'] = events['side']
    out_df['ret'] = 0.0
    
    if not events_x.empty:
        exit_prices = close.loc[events_x['exit']].values
        entry_prices = close.loc[events_x.index].values
        out_df.loc[events_x.index, 'ret'] = (np.log(exit_prices) - np.log(entry_prices)) * events_x['side'].values
        
    return out_df

def grid_pt_sl(pt,sl,close,enter,max_holding,side):
    """
    :param pt: list of profit taking target rate
    :param sl: list of stop loss target rate
    :return: (pd.DataFrame) Cumulative Returns of each pt_sl
            row = profit taking target rate, columns = stop loss target rate
    """
    out = np.ones((len(pt),len(sl)))
    df = pd.DataFrame(out)
    df.index = pt
    df.columns = sl
    for i in pt:
        for j in sl:
            pt_sl = [i,j]
            df.loc[i,j] = get_barrier_fast(close,enter,pt_sl,max_holding,side).ret.cumsum()[-1]
    return df

def get_wallet(close, barrier, initial_money=0, bet_size=None):
    """
    :param close: series of price
    :param barrier: DataFrame from get_barrier()
                barrier must include column 'exit'
    :return: (pd.DataFrame) Cumulative Returns of each pt_sl
            row = profit taking target rate, columns = stop loss target rate
    """
    close = close.round(2)
    if bet_size is None:
        bet_size = pd.Series(np.ones(len(close)),index=close.index)
    bet_amount = bet_size.loc[barrier.index]
    spend = bet_amount*close.loc[barrier.index]
    receive = pd.Series(close.loc[barrier.exit.dropna()].values, index=barrier.dropna(subset=['exit']).index)*bet_amount
    receive = receive.fillna(0)
    close_exit = pd.Series(receive.loc[barrier.index].values,index=barrier.exit).groupby(by='exit',axis=0).sum()
    close_exit = close_exit.rename('money_receive')
    
    wallet_0 = pd.DataFrame({'exit':barrier.exit,'price':close,'money_spent':spend})
    wallet = wallet_0.join(close_exit).fillna(0)
    wallet = wallet.drop(index= wallet.loc[wallet.money_spent+wallet.money_receive==0].index)
    
    buy_amount = bet_amount
    buy_amount = buy_amount.rename('buy_amount')
    sell_amount = (wallet.money_receive/wallet.price).round()
    sell_amount = sell_amount.rename('sell_amount')
    
    n_stock = ((wallet.money_spent/wallet.price).round()-(wallet.money_receive/wallet.price).round()).cumsum()
    n_stock = n_stock.rename('n_stock')
    
    inventory = (-wallet.money_spent+wallet.money_receive).cumsum() + initial_money
    inventory = inventory.rename('cash_inventory')
    
    out = wallet.join([buy_amount,sell_amount,n_stock,inventory])
    out = out.fillna(0)
    return out


def get_wallet_v2(close, barrier, initial_money=10000, bet_size=None, fee_rate=0.0006, slippage=0.0001):
    """
    高度可靠的钱包/持仓核算函数（针对 15min 加密货币回测优化）
    
    :param close: 原始价格序列 (pd.Series)
    :param barrier: get_barrier 的输出 (pd.DataFrame)，必须包含 'exit' 列
    :param initial_money: 初始账户现金
    :param bet_size: 下注数量（单位：枚/个）。如果是 None，则默认为每笔交易 1 个单位
    :param fee_rate: 单边手续费率 (0.0006 = 万六)
    :param slippage: 滑点百分比 (0.0001 = 0.01%)
    """
    # 确保索引是 datetime 格式
    close.index = pd.to_datetime(close.index)
    barrier.index = pd.to_datetime(barrier.index)
    
    # 1. 准备下注数据
    if bet_size is None:
        bet_size = pd.Series(1.0, index=barrier.index)
    else:
        bet_size = bet_size.loc[barrier.index]

    # 2. 计算【买入】详情 (Entry)
    # 考虑滑点后的买入成交价
    buy_price = close.loc[barrier.index] * (1 + slippage)
    # 总支出 = 成交额 + 手续费
    money_spent = (bet_size * buy_price) * (1 + fee_rate)
    
    # 3. 计算【卖出】详情 (Exit)
    # 仅处理有明确退出信号的交易
    exited_mask = barrier['exit'].notna()
    exit_times = pd.to_datetime(barrier.loc[exited_mask, 'exit'])
    entry_times_of_exited = barrier.loc[exited_mask].index
    
    # 退出时的市场价
    exit_market_prices = close.loc[exit_times].values
    # 考虑滑点后的卖出成交价
    sell_price = exit_market_prices * (1 - slippage)
    # 总收入 = 成交额 - 手续费
    money_received_vals = (bet_size.loc[entry_times_of_exited].values * sell_price) * (1 - fee_rate)
    
    # 4. 数据聚合 (处理同一时间戳可能存在的多笔交易)
    # 买入聚合
    entry_agg = pd.DataFrame({
        'money_spent': money_spent,
        'buy_units': bet_size
    }).groupby(level=0).sum()
    
    # 卖出聚合 (注意：卖出动作发生在 exit_times)
    exit_agg = pd.DataFrame({
        'money_receive': money_received_vals,
        'sell_units': bet_size.loc[entry_times_of_exited].values
    }, index=exit_times).groupby(level=0).sum()
    
    # 5. 构建完整的时间序列账本
    # 使用完整的 close.index 确保不遗漏任何时间点，这对 pyfolio 计算波动率至关重要
    wallet = pd.DataFrame(index=close.index)
    wallet = wallet.join(entry_agg).join(exit_agg).fillna(0)
    
    # 6. 计算核心指标 (Level 累积值)
    # 当前持仓数量 (累加买入 - 累加卖出)
    wallet['n_stock'] = (wallet['buy_units'] - wallet['sell_units']).cumsum()
    
    # 现金账户余额 (初始资金 - 支出 + 收入)
    wallet['cash_inventory'] = (-wallet['money_spent'] + wallet['money_receive']).cumsum() + initial_money
    
    # 账户总价值 (Equity) = 现金 + 持仓市值
    wallet['price'] = close
    wallet['total_equity'] = wallet['cash_inventory'] + (wallet['n_stock'] * wallet['price'])
    
    # 7. 计算收益率 (Returns)
    # pyfolio 需要的是每日收益率，这里先计算 15min 收益率
    wallet['returns'] = wallet['total_equity'].pct_change().fillna(0)
    
    return wallet

import pandas as pd
import numpy as np

def get_wallet_ratio_based_simple(close, barrier, bet_size=None, initial_money=10000, max_pos=1, fee_rate=0.0006, slippage=0.0001):
    """
    基于资金占比 (bet_size as ratio) 的回测系统 - 简化版
    去掉 numpy 加速，使用原生 pandas .loc 查找，逻辑更直观
    """
    # 1. 数据清洗与索引格式化
    close.index = pd.to_datetime(close.index)
    barrier.index = pd.to_datetime(barrier.index)
    
    # 2. 处理 bet_size
    # 如果没传，或者传入的是 None，构造一个全为 1.0 的 Series
    if bet_size is None:
        bet_size = pd.Series(1.0, index=barrier.index)
    else:
        bet_size.index = pd.to_datetime(bet_size.index)
        # 确保 bet_size 覆盖 barrier 的索引，缺失填 1.0
        bet_size = bet_size.reindex(barrier.index).fillna(1.0)

    # 3. 初始化变量
    current_cash = initial_money
    current_pos = 0.0      # 当前持有的币数
    
    open_positions = []    # 持仓队列 [{'exit_time':..., 'units':...}]
    real_trades = []       # 交易流水
    
    # 4. 遍历每一个买入信号
    # 直接遍历 barrier 的索引
    for t_entry in barrier.index:
        
        # 保护：如果信号时间不在 close 数据里，跳过
        if t_entry not in close.index:
            continue
            
        # 获取当前行数据
        # 使用 .loc 直接查找，速度较慢但可读性高
        t_exit = barrier.loc[t_entry, 'exit']
        current_price = close.loc[t_entry]
        
        # --- A. 先处理卖出 (释放资金) ---
        # 检查 open_positions 里是否有需要在此刻(或之前)平仓的
        still_open = []
        for pos in open_positions:
            # 如果预定的卖出时间早于或等于当前时间，说明该卖了
            if pos['exit_time'] <= t_entry:
                exit_t = pos['exit_time']
                
                # 确定卖出价格
                if exit_t in close.index:
                    price_exit = close.loc[exit_t]
                else:
                    # 如果退出时间在K线图中找不到（比如停牌或数据缺失），用当前价兜底
                    price_exit = current_price 
                
                # 执行卖出
                sell_price = price_exit * (1 - slippage)
                revenue = (pos['units'] * sell_price) * (1 - fee_rate)
                
                current_cash += revenue
                current_pos -= pos['units']
                
                real_trades.append({
                    'time': exit_t, 
                    'type': 'sell', 
                    'cash_delta': revenue, 
                    'pos_delta': -pos['units']
                })
            else:
                # 还没到时间，继续持有
                still_open.append(pos)
        
        # 更新持仓列表
        open_positions = still_open
        
        # --- B. 计算当前账户总权益 (Total Equity) ---
        # 权益 = 现金 + 持仓市值 (用当前 entry 价格估算)
        total_equity = current_cash + (current_pos * current_price)
        
        # --- C. 决定买入数量 ---
        
        # 1. 检查最大持仓单数限制
        if len(open_positions) >= max_pos:
            continue
            
        # 2. 获取本单计划占比 (Ratio)
        ratio = bet_size.loc[t_entry]
        
        if ratio <= 0: continue
        
        # 3. 计算【目标交易金额】
        # 逻辑：我有多少总身家 * 我想下注的百分比
        target_money_to_spend = total_equity * ratio
        
        # 4. 计算【理论需要买入的单位数】
        buy_price = current_price * (1 + slippage)
        # 公式推导：花费 = 数量 * 单价 * (1+费率)  => 数量 = 花费 / (单价 * (1+费率))
        target_units = target_money_to_spend / (buy_price * (1 + fee_rate))
        
        # 5. 【资金兜底检查】 (Reality Check)
        # 就算你想买100万，但你现金只有50万，那你最多只能买50万
        max_affordable_units = current_cash / (buy_price * (1 + fee_rate))
        
        # 最终下单数量取较小值 (理想 vs 现实)
        final_units = min(target_units, max_affordable_units)
        
        # 过滤过小的碎股
        if final_units < 0.000001:
            continue
            
        # --- D. 执行买入 ---
        cost = (final_units * buy_price) * (1 + fee_rate)
        
        current_cash -= cost
        current_pos += final_units
        
        open_positions.append({
            'exit_time': t_exit,
            'units': final_units
        })
        
        real_trades.append({
            'time': t_entry, 
            'type': 'buy',
            'cash_delta': -cost, 
            'pos_delta': final_units
        })

    # === 5. 数据聚合与输出 ===
    # 如果没交易，直接返回空账本
    if not real_trades:
        empty_wallet = pd.DataFrame(index=close.index)
        empty_wallet['total_equity'] = initial_money
        return empty_wallet

    # 聚合交易流水
    df_trades = pd.DataFrame(real_trades)
    # 按时间聚合现金和持仓变动
    df_agg = df_trades.groupby('time')[['cash_delta', 'pos_delta']].sum()
    
    # 扩展到完整时间轴
    wallet = pd.DataFrame(index=close.index)
    wallet = wallet.join(df_agg).fillna(0)
    
    # 计算累计状态
    wallet['cash_inventory'] = wallet['cash_delta'].cumsum() + initial_money
    wallet['n_stock'] = wallet['pos_delta'].cumsum()
    
    # 计算总资产曲线
    wallet['price'] = close
    wallet['total_equity'] = wallet['cash_inventory'] + (wallet['n_stock'] * wallet['price'])
    
    # 计算收益率
    wallet['returns'] = wallet['total_equity'].pct_change().fillna(0)
    
    return wallet


def show_results(wallet):
    """
    

    Parameters
    ----------
    wallet : dataframe from get_wallet()

    Returns
    -------
    show results

    """
    initial_invest = wallet.cash_inventory[0]+wallet.money_spent[0]
    cash_in_hand = wallet.cash_inventory[-1]
    stock_owned = wallet.n_stock[-1]
    stock_price_now = wallet.price[-1]
    total_asset = cash_in_hand + stock_owned*stock_price_now
    total_gain = total_asset-initial_invest
    total_return = total_gain/initial_invest
    tcost = wallet.money_receive.sum() * 0.003 + 0 # 0.3% tax, no fee
    
    print("Your initial investment money : {}".format(initial_invest))
    print("You now have cash : {}".format(cash_in_hand))
    print("You now have n of stocks : {}".format(stock_owned))
    print("Your total asset (cash+stock) now : {}".format(total_asset))
    print("Total gain : {}".format(total_gain))
    print("Transaction costs : {}".format(tcost))
    print("Total profit : {}".format(total_gain-tcost))

def get_plot_wallet(close,barrier,wallet):
    
    plot_df = close.to_frame().join(barrier)
    ret_abs = plot_df.ret.abs()
    plot_df['ret_size']=ret_abs
    ret_sign = np.sign(plot_df.ret)
    dfret = ret_sign.to_frame()
    dfret[dfret.ret==1] = 'profit'
    dfret[dfret.ret==-1] = 'loss'
    dfret[dfret.ret==0] = 'exit point'
    plot_df['This bet is']=dfret.ret
    plot_wallet = wallet.join(plot_df.dropna()[['This bet is','ret_size']])
    plot_wallet = plot_wallet.reset_index()
    plot_wallet = plot_wallet.fillna({'This bet is':'exit point','ret_size':0})
    plot_wallet = plot_wallet.rename(columns={'timestamp':'Date'})
    return plot_wallet

def get_metalabel(barrier):
    
    """
    Parameters
    ----------
    barrier : dataframe
        from get_barrier()

    Returns
    -------
    series of meta-label (1 profit(go) 0 loss(pass))

    """
    retsign = np.sign(barrier.ret)
    retsign = retsign.loc[retsign!=0]
    out = .5*(retsign+1)
    out = out.rename('label')
    return out

def plot(close,barrier,wallet):   
    plot_wallet = get_plot_wallet(close,barrier,wallet)
    initial_invest = wallet.cash_inventory[0]+wallet.money_spent[0]
    cash_in_hand = wallet.cash_inventory[-1]
    stock_owned = wallet.n_stock[-1]
    stock_price_now = wallet.price[-1]
    total_asset = cash_in_hand + stock_owned*stock_price_now
    total_gain = total_asset-initial_invest
    total_return = total_gain/initial_invest
    
    fig = px.scatter(plot_wallet, x="Date", y="price", size='buy_amount', color='This bet is'
                     ,size_max=10, hover_data=['exit','n_stock','cash_inventory','money_spent','money_receive'],color_discrete_sequence=["red", "black", "blue"])
    
    fig.update_xaxes(ticklabelmode="period")
    fig.update_layout(title_text='Now having cash + stock value = {}, Total gain = {}'.format(total_asset.round(2),(total_gain).round(2)))
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name="Close Price",opacity=0.4))
    
    fig.show()

