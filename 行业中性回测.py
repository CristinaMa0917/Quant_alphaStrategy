
# coding: utf-8

# In[ ]:

#读入合成的因子文件
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np

from CAL.PyCAL import *
sns.set_style('white')

def load_factor(path):
    factors = pd.read_csv(path, index_col=0, dtype={'ticker':np.str, 'tradedate':np.str})
    factors['ticker'] = factors['ticker'].apply(lambda x: str(x).zfill(6))
    factors['ticker'] = factors['ticker'].apply(lambda x:x + '.XSHG' if x[0] == '6' else x+'.XSHE')
    factors['tradeDate'] = pd.to_datetime(factors['tradeDate'], format='%Y%m%d')
    factors.rename(columns={"value":"factor"}, inplace=True)
    factors = factors.pivot(index='tradeDate', columns='ticker', values='factor')
    factors.index = factors.index.strftime('%Y-%m-%d')
    return factors
factors_sxgb = load_factor('alpha_ml/stacking_xgb0524.csv')
factors_lgb_roll = load_factor('alpha_ml/lgbr_active.csv')
factors_lgb = load_factor('alpha_ml/lgb0524.csv')


# In[ ]:

#计算60个组合的持仓个股和权重信息
start = u"20130329"
end = u"20171229"

import pandas as pd
import time

t1 = time.time()
############################## 获取基准权重、行业分类数据 ############################

stock_list = DataAPI.EquGet(equTypeCD=u"A",secID=u"",ticker=u"",listStatusCD=u"L,S,DE",field=u"",pandas="1")
stock_list = stock_list['secID'].tolist()

# 中证500权重
zz500_weight = DataAPI.IdxCloseWeightGet(secID=u"",ticker=u"000905",beginDate=start,endDate=end,field=u"effDate,consID,weight",pandas="1")
zz500_weight.set_index('effDate', inplace=True)

# 沪深300权重
hs300_weight = DataAPI.IdxCloseWeightGet(secID=u"",ticker=u"000300",beginDate=start,endDate=end,field=u"effDate,consID,weight",pandas="1")
hs300_weight.set_index('effDate', inplace=True)

# 股票所属行业
start_date = start
end_date = end
cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date).sort('calendarDate')
cal_dates = cal_dates[cal_dates['isMonthEnd']==1]
trade_month_list = cal_dates['calendarDate'].values.tolist()
data_list = []
tcount = 0
for date in trade_month_list:
    if tcount %12 == 0:
        print date
    date_ = ''.join(date.split('-'))
    tmp = DataAPI.RMExposureDayGet(secID=stock_list, beginDate=date_, endDate=date_)
    data_list.append(tmp)
    tcount +=1
data = pd.concat(data_list, axis=0)
indu = data.iloc[:, [0, 2] + range(15, 15+28)]
indu['tradeDate'] = indu['tradeDate'].apply(lambda x: x[:4]+"-"+x[4:6]+"-"+x[6:])
indu = indu.set_index('tradeDate')
t2 = time.time()
print "Time cost: %s seconds" %(t2-t1)


# In[ ]:

# 非行业中性下，计算权重的详细算法

def calc_wts_n(df, n):
    '''
    df: 某一期因子值的dataframe，列为 ['tradeDate', 'secID', 'factor_value']
    输出: dataframe，因子排名前n的个股，以及个股的买入权重, 列为  ['tradeDate', 'secID', 'factor_value', 'h_wts']
    '''
    n_df = df.sort_values(by='factor_value', ascending=False).head(n)
    hold_wts = 1.0/n
    n_df['h_wts'] = hold_wts
    return n_df

# 非行业中性下，得到持仓标的和权重
def portfolio_simple_long_only(tfactor, holding_num):
    '''
    tfactor: 股票及对应因子值的dataframe，列为：['tradeDate', 'secID', 'factor_value']
    holding_num: 买入的股票个数
    输出：dataframe, 每一期买入的个股及个股权重，列为: ['tradeDate', 'secID', 'h_wts']
    '''
    tfactor = tfactor.copy()
    tmp_frame = tfactor.groupby(['tradeDate']).apply(calc_wts_n, holding_num)
    del tmp_frame['tradeDate']
    tmp_frame = tmp_frame.reset_index()
    tmp_frame = tmp_frame[['tradeDate', 'secID', 'h_wts']]
    return tmp_frame[['tradeDate', 'secID', 'h_wts']].pivot(index='tradeDate', columns='secID', values='h_wts')

# 行业中性下，计算权重的详细算法
def calc_wts_neu_n(df, n):
    '''
    df: 某一期、某个行业因子值的dataframe，列为 ['tradeDate', 'secID', 'factor_value']
    输出: dataframe，因子排名前n的个股，以及个股的买入权重, 列为  ['tradeDate', 'secID', 'factor_value', 'h_wts', 'hold_n'(持仓个数)]
    '''
    indu_stock_count = len(df)
    num = min(n, indu_stock_count)
    hold_wts = df['wts'].values[0]/num
    n_df = df.sort_values(by='factor_value', ascending=False).head(num)
    n_df['h_wts'] = hold_wts
    n_df['hold_n'] = num
    return n_df

# 行业中性下，得到持仓标的和权重
def portfolio_indu_long_only(bm_weight, tfactor, tindu, holding_num=5):
    '''
    bm_weight: dataframe，基准的行业权重，effDate(index), 列为：consID,weight
    tfactor: 股票及对应因子值的dataframe，列为：['tradeDate', 'secID', 'factor_value']
    holding_num: 买入的股票个数
    输出：dataframe, 每一期买入的个股及个股权重，列为: ['tradeDate', 'secID', 'h_wts']
    '''
    tfactor = tfactor.copy()
    # 拿到每一期，行业的权重
    bm_wts = bm_weight.copy()
    bm_wts.reset_index(inplace=True)
    bm_wts.columns = ['tradeDate', 'secID', 'wts']
    bm_wts = bm_wts.merge(tindu, on=['tradeDate', 'secID'], how='inner')
    indu_total_wts = bm_wts.groupby(['tradeDate', 'indu'])['wts'].sum()/100
    indu_total_wts = indu_total_wts.reset_index()

    #合并上行业在bm中的权重
    tfactor = tfactor.merge(indu_total_wts, on=['tradeDate', 'indu'], how='left')

    tmp_frame = tfactor.groupby(['tradeDate', 'indu']).apply(calc_wts_neu_n, holding_num)
    del tmp_frame['indu']
    del tmp_frame['tradeDate']
    tmp_frame = tmp_frame.reset_index()
    tmp_frame = tmp_frame[['tradeDate', 'secID', 'h_wts']]
    return tmp_frame[['tradeDate', 'secID', 'h_wts']].pivot(index='tradeDate', columns='secID', values='h_wts')


# In[ ]:

import os
import pickle
######################################### 计算15个组合的持仓及权重 #####################################

t1 = time.time()

# 包含基准权重的字典
bm_wts_dict = {
    "HS300":hs300_weight,
    "ZZ500":zz500_weight
}

# 存储不同模型持仓权重的字典
all_factor = {
                'factors_sxgb': factors_sxgb,'factors_lgb_roll': factors_lgb_roll,'factors_lgb': factors_lgb
             }

# 存储各行业中性组合持仓及权重的字典
all_weights_neu = {
                     'factors_sxgb':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]},
                    'factors_lgb_roll':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]},
                    'factors_lgb':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]}
                    }

# 存储各行业中性组合持仓及权重的字典
all_weights = {
                 'factors_sxgb':[pd.DataFrame() for i in range(1)],
          'factors_lgb_roll':[pd.DataFrame() for i in range(1)],
    'factors_lgb':[pd.DataFrame() for i in range(1)]
              }

# 计算行业中性组合的持仓和权重
print "calc neu wts..."
for factor_name in all_factor.keys():
    # 拿到因子文件
    factor_frame = all_factor[factor_name]
    
    # 格式转换
    tfactor= factor_frame.copy()
    tfactor.reset_index(inplace=True)
    tfactor = pd.melt(tfactor, id_vars=['index'], value_vars=list(factor_frame.columns))
    tfactor.columns = ['tradeDate', 'secID', 'factor_value']

    # 合并上行业标签
    tindu = indu.copy()
    tindu.reset_index(inplace=True)
    tindu = pd.melt(tindu, id_vars=['tradeDate', 'secID'])
    tindu = tindu[tindu.value==1]
    del tindu['value']
    tindu.columns = ['tradeDate', 'secID', 'indu']
    tfactor = tfactor.merge(tindu, on=['tradeDate', 'secID'], how='inner')
    
    # 遍历持仓个数，计算对应的持仓个股和持仓权重
    for t, hold_num in enumerate([2, 5, 10, 15, 20]):#[2, 5, 10, 15, 20]
        # 遍历行业中性对应的指数
        for type_universe in ['HS300', 'ZZ500']:
            wts = portfolio_indu_long_only(bm_wts_dict[type_universe], tfactor, tindu, hold_num)
            all_weights_neu[factor_name][type_universe][t] = wts

t2 = time.time()
print u"行业中性组合 Time cost: %s seconds" %(t2-t1)

# 计算非行业中性组合的持仓和权重
print "calc non-neu wts..."
for factor_name in all_factor.keys():
    factor_frame = all_factor[factor_name]
    # 格式转换
    tfactor= factor_frame.copy()
    tfactor.reset_index(inplace=True)
    tfactor = pd.melt(tfactor, id_vars=['index'], value_vars=list(factor_frame.columns))
    tfactor.columns = ['tradeDate', 'secID', 'factor_value']
    # 遍历持仓数
    for t, hold_num in enumerate([50]): #[20, 50, 100, 150, 200]
        wts = portfolio_simple_long_only(tfactor, hold_num)
        all_weights[factor_name][t] = wts

t3 = time.time()

print u"非行业中性组合 Time cost: %s seconds"%(t3-t2)

################################# 将上面存储的权重dict存储下来，便于后面的回测使用 ##############################
save_dir = "./alpha_ml"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join("./alpha_ml/","weights_neu.txt"), 'wb') as fHandler:
    pickle.dump(all_weights_neu, fHandler)

with open(os.path.join("./alpha_ml/","weights_noneu.txt"), 'wb') as fHandler:
    pickle.dump(all_weights, fHandler)


print "Total time cost:%s seconds" %(t3-t1)


# In[ ]:

import os
import threading
import pickle
import time
import numpy as np
import pandas as pd
from quartz.context.parameters import SimulationParameters
from quartz.backtest_tools import get_backtest_data
from quartz.backtest import backtest 

start = u"20130329"
end = u"20171229"

############################### 将存储的权重dict读取出来 ######################################
def read_wts(): 
    '''
    将上面得到的行业中性、非行业中性组合持仓标的和权重信息载入
    
    返回：
    两个dict，分别对应行业中性组合、非行业中性组合的持仓标的和权重
    返回字典的格式为：all_weights_neu['factors_lstm']['HS300'][0] 为dataframe，列为：['tradeDate', 'secID', 'h_wts'] 
    '''
    # 行业中性组合的持仓标的和权重
    all_weights_neu = {
                         'factors_sxgb':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]},
        'factors_lgb_roll':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]},
        'factors_lgb':{"HS300":[pd.DataFrame() for i in range(5)], "ZZ500":[pd.DataFrame() for i in range(5)]}####change
                        }
    
    # 非行业中性组合的持仓标的和权重
    all_weights = {
                     'factors_sxgb':[pd.DataFrame() for i in range(1)],
         'factors_lgb_roll':[pd.DataFrame() for i in range(1)],
         'factors_lgb':[pd.DataFrame() for i in range(1)]####change
                  }

    with open(os.path.join("./alpha_ml/","weights_neu.txt"), 'rb') as fHandler:
        all_weights_neu = pickle.load(fHandler)

    with open(os.path.join("./alpha_ml/","weights_noneu.txt"), 'rb') as fHandler:
        all_weights = pickle.load(fHandler)
    return all_weights_neu, all_weights


############################### 回测行情预加载相关函数和设置 ######################################
'''由于回测数量大，预加载数据可以节省回测时间'''


# 行情预加载函数
def preload_market_service(start, end, universe, benchmark, max_history_window=30, **kwargs):
    parameters = {
        'start': start,
        'end': end,
        'universe': universe,
        'benchmark': benchmark,
        'max_history_window': max_history_window
    }
    sim_parameters = SimulationParameters(**parameters)
    market_service = get_backtest_data(sim_parameters)
    market_service.rolling_load_daily_data(sim_parameters.trading_days)
    return market_service


# 基于预加载的行情服务进行批量回测的相关设置
def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    pre_date = context.previous_date.strftime("%Y-%m-%d")
    if pre_date not in weights.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return

    # 组合构建                
    wts = weights.loc[pre_date, :].dropna()
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk,0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])

        
def get_account():
    accounts = {
                'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
            }
    return accounts

# 得到并存储回测信息
def get_backtest_result(params_dict):
    save_dir = params_dict['save_dir']
    factor_name = params_dict['factor_name']
    loc_index = params_dict['loc_index']
    type_universe = params_dict['type_universe']
    weights = params_dict['weights']
    
    global preloaded_market_service, A_universe
    account = get_account()
    
    print "backtesting %s, %s, %s" %(factor_name, loc_index, type_universe)
    try:
        # 调用优矿quartz进行回测
        result = backtest(start=start,end=end, benchmark= 'ZZ500', max_history_window=30,
                         preload_data=preloaded_market_service, weights=weights, freq='d',refresh_rate=Monthly(1), accounts=account, initialize=initialize, handle_data=handle_data, universe=A_universe)

        # 存储回测结果
        bt = result[0]
        tmp = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]

        if type_universe is None:
            holding_list = [20] ###### change
            holding_num = holding_list[loc_index]
            save_file = os.path.join(save_dir, "%s_%s_%s.csv"%(factor_name, "noneu", str(holding_num)))
        else:
            holding_list = [2, 5, 10, 15, 20] #[2, 5, 10, 15, 20]
            holding_num = holding_list[loc_index]
            save_file = os.path.join(save_dir, "%s_%s_%s.csv"%(factor_name, type_universe, str(holding_num)))

        tmp.to_csv(save_file, index=False)
    except Exception, err:
        print "Error", err

start_time = time.time()
save_dir = "alpha_ml"
target_universe = ['ZZ500', 'HS300']
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
t1 = time.time()




# 得到预加载的数据
print "loading universe data"
A_universe = set_universe('ZZ500')+set_universe('HS300')  ### change A_universe = DynamicUniverse('A')


print "loading market service data ..."
load_params = {
                'start': start,
                'end': end,
                'universe': A_universe,
                'benchmark': 'ZZ500',
                'max_history_window': 30
        }

# 预加载
preloaded_market_service = preload_market_service(**load_params)


# 读取之前的权重数据
print "reading wts ..."
all_weights_neu, all_weights = read_wts()


################################### 回测行业中性的组合 #######################################
# 行业中性的存储变量
result_neu = {
                  'factors_sxgb':{"HS300":{}, "ZZ500":{}} ,
                  'factors_lgb_roll':{"HS300":{}, "ZZ500":{}},
                  'factors_lgb':{"HS300":{}, "ZZ500":{}},  ##################################change !!!!!
             }

for name in result_neu.keys():
    for type_universe in target_universe:
        for i, weights in enumerate(all_weights_neu[name][type_universe]):
            # 设置回测参数

            tmp_dict = {
                "save_dir":save_dir,
                "factor_name":name,
                "loc_index":i,
                "type_universe":type_universe,
                "weights":weights
            }
            tstart = time.time()
            get_backtest_result(tmp_dict)
            tend = time.time()
            print "finished one round, time:%s"%(tend - tstart)

# ################################### 回测非行业中性的组合 #######################################
result = {'factors_sxgb':{},'factors_lgb_roll':{},'factors_lgb':{}}  #########################chaneg!
for name in result.keys():
    for i, weights in enumerate(all_weights[name]):
        tmp_dict = {
                "save_dir":save_dir,
                "factor_name":name,
                "loc_index":i,
                "type_universe":None,
                "weights":weights
            }
        tstart = time.time()
        get_backtest_result(tmp_dict)
        tend = time.time()
        print "finished one round, time:%s"%(tend - tstart)
t2 = time.time()
print "Time cost: ", t2 - t1


# 在同等条件下，对benchmark（沪深300、中证500）进行回测

# In[ ]:

weights_zz500 = pd.DataFrame()
weights_hs300 = pd.DataFrame

all_weight_bench = {'hs300': weights_hs300, 'zz500': weights_zz500}

import quartz as qz
from quartz.api import *

start = '2013-03-29'                       # 回测起始时间
end = '2017-12-31'                         # 回测结束时间
universe = ['000001.XSHE']        # 证券池，支持股票和基金、期货
benchmark = 'HS300'                        # 策略参考基准
freq = 'd'                                 # 'd'表示使用日频率回测，'m'表示使用分钟频率回测
refresh_rate = 1                           # 执行handle_data的时间间隔

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    
bt, perf, stock = qz.backtest(start=start, 
                 end=end, 
                 benchmark=benchmark, 
                 universe=universe, 
                 capital_base=100000.0, 
                 initialize=initialize, 
                 handle_data=handle_data, 
                 refresh_rate=1, 
                 freq='d', 
                 security_base={}, 
                 security_cost={}, 
                 max_history_window=(30, 241), 
                 accounts=accounts)
benchmark_return_hs300 = bt[['tradeDate','benchmark_return']]

# benchmark数据预加载
save_dir = "alpha_ml"
tmp = pd.read_csv('%s//factors_sxgb_HS300_5.csv'%save_dir)
benchmark_return_zz500 = tmp['benchmark_return']
benchmark_return_hs300 = benchmark_return_hs300['benchmark_return']


# 行业中性组合的指标对比
# 
# 此处的测试指标是基于超额收益，即将持仓组合收益用基准收益进行对冲之后的结果
# 

# In[ ]:

import os

# 计算超额收益的指标
def get_hedge_result(benchmark_return, columns=[2, 5, 10, 15, 20], bm='noneu'):
    '''
    benchmark_return: dataframe, 基准的收益序列, index为日期，列为portfolio_value， 即基准的净值
    columns: list, 每个行业的持仓个数
    bm: 保留字段
    返回： 年化超额收益、超额收益最大回撤、超额收益的信息比率、超额收益的calmar比率
    '''
    annual_excess_return = pd.DataFrame()
    excess_return_max_drawdown = pd.DataFrame()
    excess_return_ir = pd.DataFrame()

    for factor_name in ['factors_sxgb','factors_lgb_roll','factors_lgb']: ###### change!!!!
        tmp_annual_excess_return = pd.DataFrame(columns=columns, index=[factor_name])
        tmp_excess_return_max_drawdown = pd.DataFrame(columns=columns, index=[factor_name])
        tmp_excess_return_ir = pd.DataFrame(columns=columns, index=[factor_name])

        for qt, num in enumerate(columns):
            bt = pd.read_csv(os.path.join(save_dir, "%s_%s_%s.csv"%(factor_name, bm, str(num))))
            tmp = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
            tmp['portfolio_return'] = tmp['portfolio_value'] / tmp['portfolio_value'].shift(1) - 1.0   # 总头寸每日回报率
            tmp['portfolio_return'].ix[0] = tmp['portfolio_value'].ix[0] / 10000000.0 - 1.0
            tmp['excess_return'] = tmp['portfolio_return'] - benchmark_return                # 总头寸每日超额回报率
            tmp['excess'] = tmp['excess_return'] + 1.0
            tmp['excess'] = tmp['excess'].cumprod()

            tmp_annual_excess_return.iloc[0, qt] = tmp['excess'].iloc[-1]**(252.0/len(tmp)) - 1.0
            tmp_excess_return_max_drawdown.iloc[0, qt] = max([1 - v/max(1, max(tmp['excess'][:t+1])) for t,v in enumerate(tmp['excess'])])
            tmp_excess_return_ir.iloc[0, qt] = tmp_annual_excess_return.iloc[0, qt] / np.std(tmp['excess_return']) / np.sqrt(252)

        annual_excess_return = annual_excess_return.append(tmp_annual_excess_return)
        excess_return_max_drawdown = excess_return_max_drawdown.append(tmp_excess_return_max_drawdown)
        excess_return_ir = excess_return_ir.append(tmp_excess_return_ir)

    annual_excess_return = annual_excess_return.convert_objects(convert_numeric=True)
    
    excess_return_max_drawdown = excess_return_max_drawdown.convert_objects(convert_numeric=True)
    
    excess_return_ir = excess_return_ir.convert_objects(convert_numeric=True)
    calmar_ratio = annual_excess_return / excess_return_max_drawdown
    
    return annual_excess_return, excess_return_max_drawdown, excess_return_ir, calmar_ratio

# 画热力图
def heatmap_plot(data_set, ax, title=None):
    ax = sns.heatmap(data_set, ax=ax, alpha=1.0, center=0.0, annot_kws={"size": 7}, linewidths=0.01, 
                     linecolor='white', linewidth=0, cmap=cm.gist_rainbow_r, cbar=False, annot=True)
    y_label = data_set.index.tolist()[::-1]
    x_label = data_set.columns.tolist()
    ax.set_yticklabels(y_label, minor=False, fontproperties=font, fontsize=10)
    ax.set_xticklabels(x_label, minor=False, fontproperties=font, fontsize=10)
    if title:
        ax.set_title(title, fontproperties=font, fontsize=16)
    return ax

# 分别以中证500、沪深300作为基准，计算行业中性组合对冲之后的回测指标
annual_excess_return_500, excess_return_max_drawdown_500, excess_return_ir_500, calmar_ratio_500 = get_hedge_result(benchmark_return_zz500, bm='ZZ500')
annual_excess_return_300, excess_return_max_drawdown_300, excess_return_ir_300, calmar_ratio_300 = get_hedge_result(benchmark_return_hs300, bm='HS300')


# In[ ]:

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np

from CAL.PyCAL import *
sns.set_style('white')

fig = plt.figure(figsize=(24, 5))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

ax1 = heatmap_plot(annual_excess_return_500, ax1, title=u'年化超额收益率（行业中性）')
ax2 = heatmap_plot(excess_return_max_drawdown_500, ax2, title=u'超额收益最大回撤（行业中性）')
ax3 = heatmap_plot(excess_return_ir_500, ax3, title=u'信息比率（行业中性）')
ax4 = heatmap_plot(calmar_ratio_500, ax4, title=u'Calmar比率（行业中性）')

# 下图横轴为每个行业中，买入的股票个数，纵轴为模型名，从上到下依次为 线性模型、rnn模型、gru模型、lstm模型 


# 以中证500行业中性组合为例进行了展示，用户可以指定其它组合
# 
# 收益计算说明：组合收益为对冲中证500指数之后的超额收益
# 
# 回撤计算说明：回撤值= -(前面最高净值 - 当前净值)/(前面最高净值)

# In[ ]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CAL.PyCAL import *    # CAL.PyCAL中包含font

save_dir = "./alpha_ml"


# 计算净值和回撤
def get_pf(path):
    '''
    path: 组合回测数据文件的存储路径
    返回： 净值序列和最大回撤序列
    '''
    bt = pd.read_csv(path)
    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']].set_index('tradeDate')
    data.index = pd.to_datetime(data.index)
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()

    df_cum_rets = data['excess']
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    return data, underwater


# # 读取组合回测数据
data_sxgb, underwater_sxgb = get_pf(os.path.join(save_dir, "factors_sxgb_ZZ500_5.csv"))
data_lgbR, underwater_lgbR = get_pf(os.path.join(save_dir, "factors_lgb_roll_ZZ500_5.csv"))
data_lgb, underwater_lgb = get_pf(os.path.join(save_dir, "factors_lgb_ZZ500_5.csv"))

# 画图展示
fig = plt.figure(figsize=(14, 6))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid(True)
ax1.set_ylim(-0.2, 0.2)
ax1.fill_between(underwater_sxgb.index, 0, np.array(underwater_sxgb), color='r')
ax1.fill_between(underwater_lgbR.index, 0, np.array(underwater_lgbR), color='k')
ax1.fill_between(underwater_lgb.index, 0, np.array(underwater_lgb), color='g')

(data_sxgb['excess']-1).plot(ax=ax2, label='stacking-xgb', color='r')
(data_lgbR['excess']-1).plot(ax=ax2, label='lgb_roll', color='k')
(data_lgb['excess']-1).plot(ax=ax2, label='lgb', color='g')
ax2.set_ylim(-4, 4)
ax2.legend(loc='best')
s = ax1.set_title(u"对冲组合超额收益走势（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
s = ax2.set_ylabel(u"累计超额收益（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_xlabel(u"红线组合：中证500行业中性、每个行业买5只、对冲中证500指数", fontproperties=font, fontsize=16)


# 这个五分位回测结果跟之前一样



