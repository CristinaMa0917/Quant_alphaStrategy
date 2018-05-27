
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import os
import time
signal_df = pd.read_csv('alpha_ml/lgb_roll_month.csv', dtype={"ticker": np.str, "tradeDate": np.str},index_col=0, encoding='GBK')
signal_df['ticker'] = signal_df['ticker'].apply(lambda x: str(x).zfill(6))
signal_df['ticker'] = signal_df['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
signal_df = signal_df[[u'ticker', u'tradeDate', u'factor']]

signal_df.head()


# In[ ]:

import time
from CAL.PyCAL import * 

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2013-03-29'                       # 回测起始时间
end = '2017-12-31'                         # 回测结束时间
benchmark = 'ZZ500'                        # 策略参考标准
universe =set_universe('ZZ500')+set_universe('HS300')           # 证券池，支持股票和基金
capital_base = 10000000                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  

factor_data = signal_df[['ticker', 'tradeDate', 'factor']]     # 读取因子数据
factor_data = factor_data.set_index('tradeDate', drop=True)
q_dates = factor_data.index.values

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}

# ---------------回测参数部分结束----------------

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)
# 运行结果
results = {}

# 调整参数(选取股票的集成因子五分位数)，进行快速回测
for quantile_five in range(1, 6):
    
    # ---------------策略逻辑部分----------------
    
    def initialize(context):                   # 初始化虚拟账户状态
        pass

    def handle_data(context): 
        account = context.get_account('fantasy_account')
        current_universe = context.get_universe('stock', exclude_halt=True)
        pre_date = context.previous_date.strftime("%Y%m%d")
        if pre_date not in q_dates:            
            return

        # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
        q = factor_data.ix[pre_date].dropna()
        q = q.set_index('ticker', drop=True)
        q = q.ix[current_universe]
        
        q_min = q['factor'].quantile((quantile_five-1)*0.2)
        q_max = q['factor'].quantile(quantile_five*0.2)
        my_univ = q[(q['factor']>=q_min) & (q['factor']<q_max)].index.values

       # 交易部分
        positions = account.get_positions()
        sell_list = [stk for stk in positions if stk not in my_univ]
        for stk in sell_list:
            account.order_to(stk,0)
        
        # 在目标股票池中的，等权买入
        for stk in my_univ:
            account.order_pct_to(stk, 1.0/len(my_univ))


    # 生成策略对象
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # ---------------策略定义结束----------------
    
    # 开始回测
    bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)

    # 保存运行结果，1为因子最强组，5为因子最弱组
    results[6 - quantile_five] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt}    

    print ('backtesting for group %s..................................' % str(quantile_five)),
print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


# In[ ]:

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()

for qt in results:
    bt = results[qt]['bt']

    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0   # 总头寸每日回报率
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return                 # 总头寸每日超额回报率
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()                # 总头寸对冲指数后的净值序列
    data['portfolio'] = data.portfolio_return + 1.0     
    data['portfolio'] = data.portfolio.cumprod()          # 总头寸不对冲时的净值序列
    data['benchmark'] = data.benchmark_return + 1.0
    data['benchmark'] = data.benchmark.cumprod()          # benchmark的净值序列
    results[qt]['hedged_max_drawdown'] = max([1 - v/max(1, max(data['excess'][:i+1])) for i,v in enumerate(data['excess'])])  # 对冲后净值最大回撤
    results[qt]['hedged_volatility'] = np.std(data['excess_return'])*np.sqrt(252)
    results[qt]['hedged_annualized_return'] = (data['excess'].values[-1])**(252.0/len(data['excess'])) - 1.0
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))
    

ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"因子不同五分位数分组选股净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"因子不同五分位数分组选股对冲中证500指数后净值走势", fontproperties=font, fontsize=16)

# results 转换为 DataFrame
results_pd = pd.DataFrame(results).T.sort_index()

results_pd = results_pd[[u'alpha', u'beta', u'information_ratio', u'sharpe', u'annualized_return', u'max_drawdown',  
                         u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']]

cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'),
        (u'纯股票多头时', u'最大回撤'), (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
results_pd.columns = pd.MultiIndex.from_tuples(cols)
results_pd.index.name = u'五分位组别'
results_pd


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



