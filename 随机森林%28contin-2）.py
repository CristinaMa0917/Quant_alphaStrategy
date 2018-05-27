
# coding: utf-8

# ### 3.1、数据准备 </font>
# ---
# - 该部分耗时 **10分钟左右**
# 
# ---
# 该部分内容为：
# - 从uqer的DataAPI中获取70个因子的数值，取得因子对应的下一期的股价涨跌数据，并将数据进行对齐
# 
# - 对因子进行缺失值填充、winsorize,neutralize,standardize处理
# 
# - 对上述数据进行处理，方便后续分类模型进行训练、测试

# ##### 3.1.1 获取因子数据和股价涨跌数据

# - 获取因子的原始数据值, 并将所需要的数据都存下来，便于后面的模型参数调整、优化，节省时间。生成raw_data/factor_chpct.csv 
# 
# - 数据文件的格式为：
# ![factor_chpct.csvc](http://odqb0lggi.bkt.clouddn.com/560a3e12f9f06c597165ef9c/01bf8e8a-2026-11e8-927b-0242ac140002)
# 
# 
# - 该段代码用了多线程加速(代码62行：ThreadPool(processes=16)），可以根据用户自己运行环境进行调整线程数。

# In[ ]:

universe = set_universe('HS300')+set_universe('ZZ500')
end = '20180517'


# In[ ]:



import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

#定义70个因子
factors = [b'Beta60', b'OperatingRevenueGrowRate', b'NetProfitGrowRate', b'NetCashFlowGrowRate', b'NetProfitGrowRate5Y',
           b'TVSTD20',
           b'TVSTD6', b'TVMA20', b'TVMA6', b'BLEV', b'MLEV', b'CashToCurrentLiability', b'CurrentRatio', b'REC',
           b'DAREC', b'GREC',
           b'DASREV', b'SFY12P', b'LCAP', b'ASSI', b'LFLO', b'TA2EV', b'PEG5Y', b'PE', b'PB', b'PS', b'SalesCostRatio',
           b'PCF', b'CETOP',
           b'TotalProfitGrowRate', b'CTOP', b'MACD', b'DEA', b'DIFF', b'RSI', b'PSY', b'BIAS10', b'ROE', b'ROA',
           b'ROA5', b'ROE5',
           b'DEGM', b'GrossIncomeRatio', b'ROECut', b'NIAPCut', b'CurrentAssetsTRate', b'FixedAssetsTRate', b'FCFF',
           b'FCFE', b'PLRC6',
           b'REVS5', b'REVS10', b'REVS20', b'REVS60', b'HSIGMA', b'HsigmaCNE5', b'ChaikinOscillator',
           b'ChaikinVolatility', b'Aroon',
           b'DDI', b'MTM', b'MTMMA', b'VOL10', b'VOL20', b'VOL5', b'VOL60', b'RealizedVolatility', b'DASTD', b'DDNSR',
           b'Hurst']

def get_factor_by_day(tdate):
    '''
    获取给定日期的因子信息
    参数： 
        tdate, 时间，格式%Y%m%d
    返回:
        DataFrame, 返回给定日期的70个因子值
    '''
    cnt = 0
    while True:
        try:
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate,secID=universe,ticker=u"",field=['ticker', 'tradeDate'] + factors,pandas="1")
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace("-", ""))

            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate)
                break


if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate="20070101", endDate=end, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]

    print("begin to get factor value for each stock...")
    # # 取得每个月末日期，所有股票的因子值
    pool = ThreadPool(processes=16)
    date_list = [tdate.replace("-", "") for tdate in trade_date.calendarDate.values if tdate < "20180430"]
    frame_list = pool.map(get_factor_by_day, date_list)
    pool.close()
    pool.join()
    print ("ALL FINISHED")

    factor_csv = pd.concat(frame_list, axis=0)
    factor_csv.reset_index(inplace=True, drop=True)
    stock_list = np.unique(factor_csv.ticker.values)

    ########################## 取得个股和指数的行情数据 ################################
    print("\nbegin to get price ratio for stocks and index ...")
    # 个股绝对涨幅
    chgframe = DataAPI.MktEqumAdjGet(secID=u"", ticker=stock_list, monthEndDate=u"", isOpen=u"", beginDate=u"20070131",
                                      endDate=end, field=['ticker', 'endDate', 'tradeDays', 'chgPct', 'return'], pandas="1")
    
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))

    # 沪深300指数涨幅
    hs300_chg_frame = DataAPI.MktIdxmGet(beginDate=u"20070131", endDate=end, indexID=u"000300.ZICN", ticker=u"",
                                         field=['ticker', 'endDate', 'chgPct'], pandas="1")
    hs300_chg_frame['endDate'] = hs300_chg_frame['endDate'].apply(lambda x: x.replace("-", ""))
    hs300_chg_frame.head()

    # 得到个股的相对收益
    hs300_chg_frame.columns = ['HS300', 'endDate', 'HS300_chgPct']
    pframe = chgframe.merge(hs300_chg_frame, on=['endDate'], how='left')
    pframe['active_return'] = pframe['chgPct'] - pframe['HS300_chgPct']
    pframe = pframe[['ticker', 'endDate', 'return', 'active_return']]
    pframe.rename(columns={"return": "abs_return"}, inplace=True)

    ################################ 对齐数据 ################################
    print("begin to align data ...")
    # 得到月度关系
    month_frame = trade_date[['calendarDate', 'isOpen']]
    month_frame['prev_month_end'] = month_frame['calendarDate'].shift(1)
    month_frame = month_frame[['prev_month_end', 'calendarDate']]
    month_frame.columns = ['month_end', 'next_month_end']
    month_frame.dropna(inplace=True)
    month_frame['month_end'] = month_frame['month_end'].apply(lambda x: x.replace("-", ""))
    month_frame['next_month_end'] = month_frame['next_month_end'].apply(lambda x: x.replace("-", ""))

    # 对齐月度关系
    factor_frame = factor_csv.merge(month_frame, left_on=['tradeDate'], right_on=['month_end'], how='left')

    # 得到个股下个月的涨幅数据
    factor_frame = factor_frame.merge(pframe, left_on=['ticker', 'next_month_end'], right_on=['ticker', 'endDate'])

    del factor_frame['month_end']
    del factor_frame['endDate']

    ################################ 数据存储下来 ################################
    factor_frame.to_csv(os.path.join(raw_data_dir, 'factor_chpct_2.csv'), chunksize=1000)

    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))


# In[ ]:

df = pd.read_csv(os.path.join(raw_data_dir, 'factor_chpct_2.csv'))
df.tail()


# #### 3.1.2 对数据进行winsorize, neutralize, standardize （6分钟）

# - winsorize
# 	- 上界值=因子均值+5*|平均值（因子值-因子均值）|，下界值=因子均值-5*|平均值（因子值-因子均值）|，超过上下界的值用上下界值填充
# 
# 
# - 对数据空值进行填充： 用同期申万一级**行业的均值**进行空值填充
# 
# - neutralize和standardize
# 	- 直接调用优矿的neutralize函数进行中性化，中性时候不包括'BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY'以和研报一致
# 	
# 	- 对中性化后的因子进行标准化，直接调用优矿的standardize函数
#  
# - 处理后的文件存储在 raw_data/after_prehandle.csv, 文件的数据格式如下：
# ![图片注释](http://odqb0lggi.bkt.clouddn.com/560a3e12f9f06c597165ef9c/b9d5730e-20ee-11e8-85a4-0242ac140002)

# In[ ]:


import pandas as pd
import numpy as np
import os
import shutil
import multiprocessing
import time
import gevent
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


######################################### 通用变量设置 #########################################
start_time = time.time()
raw_data_dir = "./raw_data"

pre_handle_dir = "./pre_handle_data"  # 存放中间数据
if not os.path.exists(pre_handle_dir):
    os.mkdir(pre_handle_dir)

# 申万一级行业分类
sw_map_frame = DataAPI.EquIndustryGet(industryVersionCD=u"010303", industry=u"", secID=u"", ticker=u"", intoDate=u"",field=[u'ticker', 'secShortName', 'industry', 'intoDate', 'outDate', 'industryName1', 'industryName2', 'industryName3', 'isNew'], pandas="1")
sw_map_frame = sw_map_frame[sw_map_frame.isNew == 1]
    

# 读入原始因子
input_frame = pd.read_csv(os.path.join(raw_data_dir, u'factor_chpct_2.csv'),
                          dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str}, index_col=0)

# 得到因子名
extra_list = ['ticker', 'tradeDate', 'next_month_end', 'abs_return', 'active_return']
factor_name = [x for x in input_frame.columns if x not in extra_list]

print('init data done, cost time: %s seconds' % (time.time()-start_time))

################################### 定义数据处理的一些基本函数 ##################################

def paper_winsorize(v, upper, lower):
    '''
    winsorize去极值，给定上下界
    参数:    
        v: Series, 因子值
        upper: 上界值
        lower: 下界值
    返回:
        Series, 规定上下界后因子值
    '''
    if v > upper:
        v = upper
    elif v < lower:
        v = lower
    return v

def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    media_v = cdate_input.median()
    for a_factor in factor_name:
        dm = media_v[a_factor]
        new_factor_series = abs(cdate_input[a_factor] - dm)  # abs(di-dm)
        dm1 = new_factor_series.median()
        upper = dm + 5 * dm1
        lower = dm - 5 * dm1
        cdate_input[a_factor] = cdate_input[a_factor].apply(lambda x: paper_winsorize(x, upper, lower))
    return cdate_input


def nafill_by_sw1(cdate_input):
    '''
    用申万一级的均值进行填充
    参数:
        cdate_input: 因子值，DataFrame
    返回:
        DataFrame, 填充缺失值后的因子值
    '''
    func_input = cdate_input.copy()
    func_input = func_input.merge(sw_map_frame[['ticker', 'industryName1']], on=['ticker'], how='left')
    
    func_input.loc[:, factor_name] = func_input.loc[:, factor_name].fillna(func_input.groupby('industryName1')[factor_name].transform("mean"))
    
    return func_input.fillna(0.0)


def winsorize_fillna_date(tdate):
    '''
    对某一天的数据进行去极值，填充缺失值
    参数:
        tdate： 时间， 格式为 %Y%m%d
    返回:
        DataFrame, 去极值，填充缺失值后的因子值
    '''
    cnt = 0
    while True:
        try:
            cdate_input = input_frame[input_frame.tradeDate == tdate]
            # print("####Running single_date for %s" % tdate)
            # winsorize
            cdate_input = winsorize_by_date(cdate_input)

            # 缺失值填充, 用同行业的均值
            cdate_input = nafill_by_sw1(cdate_input)
            cdate_input.set_index('ticker', inplace=True)

            return cdate_input
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                cdate_input = input_frame[input_frame.tradeDate == tdate]
                # 缺失值填充, 用同行业的均值
                cdate_input = nafill_by_sw1(cdate_input)
                cdate_input.set_index('ticker', inplace=True)
                return cdate_input
            
            
def standardize_neutralize_factor(input_data):
    '''
    行业、市值中性化，并进行标准化
    参数: 
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回:
        DataFrame, 行业、市值中性化，并进行标准化后的因子值
    '''
    cdate_input, tdate = input_data
    for a_factor in factor_name:
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = standardize(neutralize(cdate_input[a_factor], target_date=tdate,
                        exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY']))
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break
    
    return cdate_input

            
if __name__ == "__main__":
    ############################################ 对每期的数据进行处理 ###########################################
    # 遍历每个月末日期，对因子进行去极值、空值填充
    print('winsorize factor data...')
    pool = Pool(processes=8)
    date_list = [tdate for tdate in np.unique(input_frame.tradeDate.values) if int(tdate) > 20061231]
    dframe_list = pool.map(winsorize_fillna_date, date_list)

    # 遍历每个月末日期，利用协程对因子进行标准化，中性化处理
    print('standardize & neutralize factor...')
    jobs = [gevent.spawn(standardize_neutralize_factor, value) for value in zip(dframe_list, date_list)]
    gevent.joinall(jobs)
    new_dframe_list = [e.value for e in jobs]
    print('standardize neutralize factor finished!')
    
            
    # 将不同月份的数据合并到一起
    all_frame = pd.concat(new_dframe_list, axis=0)
    all_frame.reset_index(inplace=True)

    # 存储下来
    all_frame.to_csv(os.path.join(raw_data_dir, "after_prehandle_2.csv"), encoding='gbk', chunksize=1000)
    end_time = time.time()
    print("\nData handle finished! Time Cost:%s seconds" % (end_time - start_time))


# In[ ]:

d = pd.read_csv(os.path.join(raw_data_dir, "after_prehandle_2.csv"))
d.tail()


# #### 3.1.3 模型数据准备(2分钟)

# - 给原始数据打上标签，在每个月末截面期，选取下月收益排名前30%的股票作为正例（𝑦=1），后30%的股票作为负例（𝑦=−1），其余的股票标签为0.
# 
# - 处理后的文件存储在 raw_data/dataset.csv, 文件的数据格式如下：
# ![图片注释](http://odqb0lggi.bkt.clouddn.com/55adb48bf9f06c94497ec025/a05797c4-21aa-11e8-85a4-0242ac140002)

# In[ ]:

import pandas as pd
import numpy as np
import os
import time

start_time = time.time()
raw_data_dir = "./raw_data"

def get_label_by_return(filename):
    '''
    对下期收益打标签，涨幅top 30%为+1，跌幅top 30%为-1
    参数:
        filename：csv文件名，为上诉步骤保存的因子值
    返回:
        DataFrame, 打完标签后的数据
    '''
    df = pd.read_csv(filename, dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str},index_col=0, encoding='gb2312').fillna(0.0)

    new_df = None
    for date, group in df.groupby('tradeDate'):
        quantile_30 = group['active_return'].quantile(0.3)
        quantile_70 = group['active_return'].quantile(0.7)

        def _get_label(x):
            if x >= quantile_70:
                return 1
            elif x <= quantile_30:
                return -1
            else:
                return 0

        group.loc[:, 'label'] = group.loc[:, 'active_return'].apply(lambda x : _get_label(x))

        if new_df is None:
            new_df = group
        else:
            new_df = pd.concat([new_df, group],ignore_index=True)

    return new_df
new_df = get_label_by_return(os.path.join(raw_data_dir, "after_prehandle_2.csv"))
new_df['year'] = new_df['next_month_end'].apply(lambda x: int(int(x)/10000))
new_df.to_csv(os.path.join(raw_data_dir, "dataset_2.csv"), encoding='gbk', chucksize=1000)

print("Done, Time Cost:%s seconds" % (time.time() - start_time))


# In[ ]:

new_df.tail()


# ### 3.2、模型因子合成 </font>
# ---
# - 该部分耗时 **2分钟左右**
# 
# ---
# 
# 该部分分为2个阶段
# -  第一个阶段训练模型, 并合成因子
# -  第二个阶段分析模型的准确度

# #### 3.2.1 模型训练(1分钟左右)

# - 为了能让模型及时抓取到市场的变化，我们采用了七个阶段滚动回测方法。模型训练区间为20070101至20171231，按年份分为7个子区间
# - 数据划分和回测设置的示意图如下（由于因子数据从2007年开始，因此前面几年的训练数据稍短一些）：
# ![图片注释](http://odqb0lggi.bkt.clouddn.com/560a3e12f9f06c597165ef9c/9579776c-2071-11e8-927b-0242ac140002)
# 
# - 根据华泰研报中的思路，在训练中会丢弃掉涨跌幅处于中间位置(label=0)的样本，以减少随机噪声的影响

# In[ ]:

#--------------------------------加载第一部分的预处理数据-----------------------------------------------
import os
import pandas as pd
import numpy as np


factors = [b'Beta60', b'OperatingRevenueGrowRate', b'NetProfitGrowRate', b'NetCashFlowGrowRate', b'NetProfitGrowRate5Y', b'TVSTD20',
           b'TVSTD6', b'TVMA20', b'TVMA6', b'BLEV', b'MLEV', b'CashToCurrentLiability', b'CurrentRatio', b'REC', b'DAREC', b'GREC',
           b'DASREV', b'SFY12P', b'LCAP', b'ASSI', b'LFLO', b'TA2EV', b'PEG5Y', b'PE', b'PB', b'PS', b'SalesCostRatio', b'PCF', b'CETOP',
           b'TotalProfitGrowRate', b'CTOP', b'MACD', b'DEA', b'DIFF', b'RSI', b'PSY', b'BIAS10', b'ROE', b'ROA', b'ROA5', b'ROE5',
           b'DEGM', b'GrossIncomeRatio', b'ROECut', b'NIAPCut', b'CurrentAssetsTRate', b'FixedAssetsTRate', b'FCFF', b'FCFE', b'PLRC6',
           b'REVS5', b'REVS10', b'REVS20', b'REVS60', b'HSIGMA', b'HsigmaCNE5', b'ChaikinOscillator', b'ChaikinVolatility', b'Aroon',
           b'DDI', b'MTM', b'MTMMA', b'VOL10', b'VOL20', b'VOL5', b'VOL60', b'RealizedVolatility', b'DASTD', b'DDNSR', b'Hurst']
raw_data_dir = "./raw_data"
df = pd.read_csv(os.path.join(raw_data_dir, "dataset.csv"), dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str},index_col=0, encoding='GBK')
df.head()


# In[ ]:

#--------------------------------模型训练、验证、测试阶段-----------------------------------------------

import time
import os

def get_train_val_test_data(df, year):
    '''
    #给定年份，拆分训练、验证、测试集
    参数:
        df: 第一部分处理后的数据，存储在./raw_data/dataset.csv
        year: 年份，是上述7个滚动测试阶段的区别字段
    返回:
        train_df,  test_df: 均为DataFrame，对应着训练、测试的数据集
    '''
    
    back_year = max(2007, year-6)
    train_df = df[(df['year']>=back_year) & (df['year']<year)]
    
    test_df = df[df['year']==year]
    
    return train_df, test_df


def format_feature_label(origin_df, is_filter=True):
    '''
    转换为模型输入格式
    参数:
        origin_df: 原始输入数据，DataFrame
        is_filter: 是否需要过滤label为0的数据
    返回:
        feature, np.array, 对应着更改格式后的数据特征
        label, np.array, 对应着更改格式后的数据标签
    '''
    
    if is_filter:
        origin_df = origin_df[origin_df['label']!=0]
        #模型的label输入范围替换成[0, 1]，比较直观，需要对原始label进行替换
        origin_df['label'] = origin_df['label'].replace(-1, 0)
        
    feature = np.array(origin_df[factors])
    label = np.array(origin_df['label'])

    return feature, label

def write_factor_to_csv(df, predict_score, year, filename):
    '''
    记录模型预测分数为因子值，输出
    参数:
        df： 原始数据， DataFrame
        predict_score: 模型预测的分数
        year: 年份
        filename: 需要保存的文件名称
    '''
    
    df['factor'] = predict_score
    df = df.loc[:, ['ticker', 'tradeDate', 'label', 'factor']]
    is_header = True
    if year != 2011:
        is_header = False
    
    df.to_csv(filename, mode='a+', encoding='utf-8', header=is_header)


from sklearn.ensemble import RandomForestClassifier  
def get_rf_result(train_data, train_label, test_data): #model 4
    rf = RandomForestClassifier(n_estimators = 50,min_samples_split = 50,min_samples_leaf =13,                                n_jobs = 8,max_depth = 14)
    rf.fit(train_data, train_label)
    predict_score = rf.predict_proba(test_data)[:, 1]   
    return predict_score


def pipeline():
    '''
    对7个阶段分别进行训练测试，并保存测试的因子合成值
    返回:
        boost_model_list, list结构，每个阶段汇总的模型集合
    '''
    
    t0 = time.time()
    raw_data_dir = "./raw_data"
    
    linear_file = os.path.join(raw_data_dir, "factor_linear_1.csv")
    try:
        os.remove(linear_file) #删除历史因子文件，以防冲突
        os.remove(pca_linear_file)
    except Exception as e:
        pass
    
    boost_model_list = []
    for year in range(2011, 2018):
        print('training model for %s' % year)
        t1 = time.time()
        #构建训练测试数据
        train_df, test_df = get_train_val_test_data(df, year)
        train_feature, train_label = format_feature_label(train_df)
        test_feature, test_label = format_feature_label(test_df, False)
        
        #线性逻辑回归模型训练，得到因子值输出
        predict_score = get_rf_result(train_feature, train_label, test_feature)
        write_factor_to_csv(test_df, predict_score, year, linear_file)
        
        print('------------------ finish year: %s, time cost: %s seconds--------------' % (year, time.time() - t1))
    
    print('Done, Time cost: %s seconds' % (time.time() - t0))

    
pipeline()


# #### 3.2.2 模型结果分析(1分钟)

# - 计算知7个阶段的平均准确率在56%左右，平均AUC在59%左右
# - AUC的具体意义参见文档[AUC释义](https://www.zhihu.com/question/39840928)

# In[ ]:

from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def get_test_auc_acc(filename):
    '''
    计算二分类模型样本外的ACC与AUC，按照日期统计
    返回:
        acc_list: 样本外的预测准确率集合
        auc_list: 样本外的预测AUC集合
        mean_acc: 样本外的平均预测准确率
        mean_auc: 样本外的平均预测AUC
    '''
    
    df = pd.read_csv(filename)
    #只查看原有label为+1, -1的数据
    df = df[df['label'] != 0]
    df.loc[:, 'predict'] = df.loc[:, 'factor'].apply(lambda x : 1 if x > 0.5 else -1)

    acc_list = []  #保存每个月份的准确率
    auc_list = []  #保存每个月份的AUC指标
    for date, group in df.groupby('tradeDate'):
        df_correct = group[group['predict'] == group['label']]
        correct = len(df_correct) * 1.0 / len(group)
        auc =  roc_auc_score(np.array(group['label']), np.array(group['factor']))
        acc_list.append([date, correct])
        auc_list.append([date, auc])
        
    acc_list = sorted(acc_list, key=lambda x: x[0], reverse=False)
    mean_acc = sum([item[1] for item in acc_list]) / len(acc_list)
    
    auc_list = sorted(auc_list, key=lambda x: x[0], reverse=False)
    mean_auc = sum([item[1] for item in auc_list]) / len(auc_list)
    
    return acc_list, auc_list, round(mean_acc, 2), round(mean_auc, 2)

def plot_accuracy_curve(filename):
    '''
    画图
    '''
    acc_list, auc_list, mean_acc, mean_auc = get_test_auc_acc(filename)

    plt.plot([datetime.strptime(str(item[0]), '%Y%m%d') for item in acc_list], [item[1] for item in acc_list], '-bo')
    plt.plot([datetime.strptime(str(item[0]), '%Y%m%d') for item in auc_list], [item[1] for item in auc_list], '-ro')

    plt.legend([u"acc curve: mean_acc:%s"%mean_acc, u"auc curve: mean auc:%s"%mean_auc], loc='upper left', handlelength=2, handletextpad=0.5, borderpad=0.1)
    plt.ylim((0.3, 0.8))
    plt.show()

plot_accuracy_curve(os.path.join(raw_data_dir, "factor_linear.csv"))


# ### 3.3、因子测试 
# * 对因子进行五分位分组，并进行回测
# * 耗时8分钟左右

# In[ ]:

import pandas as pd
import numpy as np
signal_df = pd.read_csv("raw_data/factor_linear.csv", dtype={"ticker": np.str, "tradeDate": np.str},index_col=0, encoding='GBK')
signal_df['ticker'] = signal_df['ticker'].apply(lambda x: str(x).zfill(6))
signal_df['ticker'] = signal_df['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
signal_df = signal_df[[u'ticker', u'tradeDate', u'factor']]

signal_df.head()


# In[ ]:

import time
from CAL.PyCAL import * 

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2011-01-01'                       # 回测起始时间
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



