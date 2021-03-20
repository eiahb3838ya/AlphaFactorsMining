import pickle
import pandas as pd
import numpy as np
import os
from Tool import Logger, GeneralData, Factor,globalVars
from GeneticPogramming.utils import rowwise_corrcoef
from GetData import load_data, align_all_to
from GeneticPogramming.Backtest.HFBacktest import HFBacktest,HFSingleFactorBacktest



PROJECT_ROOT = 'F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\factors'

with open(os.path.join(os.path.join(os.path.join(PROJECT_ROOT, "20210306_18_37_48"),"found_factors"), "log(subtract_(open,high)).pickle"),'rb') as f:
    factorData = pickle.load(f)
with open('F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\factors\\20210306_18_37_48\\found_factors\\'
          'log(subtract_(open, high)).pickle', 'rb') as f:
    factorData = pickle.load(f)

factor_arr = factorData.generalData

globalVars.initialize()
load_data("materialData",
          "F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\h5\\materialData_newData.h5")

PERIOD_START = "2017-01-01"
PERIOD_END = "2019-01-01"

# 读取无风险利率
rf_df = pd.read_csv('F:\\Keira\\AlphaSignalFromMachineLearning\\GeneticPogramming\\Backtest\\rf.csv',
                    index_col=0,parse_dates = True)
# get the return to compare
# 定義用來放進 evaluation function 的 收益率
open_ = globalVars.materialData['open']
shiftedPctChange_df = open_.to_DataFrame().pct_change().shift(-2)  # 使用後天到明天開盤價的 pctChange 作為 收益率
periodShiftedPctChange_df = shiftedPctChange_df.loc[PERIOD_START:PERIOD_END]

factor_df= pd.DataFrame(factor_arr,index=periodShiftedPctChange_df.index, columns=periodShiftedPctChange_df.columns)
rf_df = rf_df.reindex(index=factor_df.index)
#adjusted return
adj_return_df = periodShiftedPctChange_df - rf_df


def ret_group(factor :pd.DataFrame, shiftedPctChange:pd.DataFrame, group_num) -> float:
    holding_period = 5
    num_prod = int(len(factor.columns) / group_num)
    backtest_period = factor.index
    ret = pd.DataFrame(index=backtest_period, columns=np.array(range(1, 1 + group_num)), dtype='object')
    for i in range(len(backtest_period)):
        change_date = backtest_period[i]
        if np.mod(i, holding_period) == 0:  # 换仓日,排序记录每组的序号
            prod_ID_group = dict()
            rank = np.argsort(-factor.iloc[i, :])  # 倒序
            for group in range(1, group_num):
                prod_ID_group[group] = rank[(group - 1) * num_prod:group * num_prod]
            prod_ID_group[group_num] = rank[(group_num - 1) * num_prod:]
        daily_return = shiftedPctChange.iloc[i, :]
        for group in range(1, group_num + 1):
            ret.loc[change_date, group] = np.nanmean(daily_return[prod_ID_group[group]])
    return ret

#多头
#累计收益(超额)
def long_return(factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    group_num =5
    ret = ret_group(factor, shiftedPctChange, group_num)
    nav = (ret+1).cumprod()
    total_return = nav.iloc[-1,:]- 1
    print(total_return.iloc[0])
    print(total_return.iloc[group_num-1])
    return max(total_return.iloc[0],total_return.iloc[group_num-1])


#年化收益（超额）
def annual_return(factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    annual_rtn = ((1+long_return(factor , shiftedPctChange))*250/len(factor.index))-1
    return annual_rtn


#最大回撤（超额）
def max_drawdown (factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    group_num = 5
    ret = ret_group(factor, shiftedPctChange, group_num)
    nav = (ret + 1).cumprod()
    max_return = np.fmax.accumulate(nav)
    max_drawdown = (np.nanmin((nav - max_return) / max_return,axis=0))[0]
    return max_drawdown

#Sharpe比率（超额）
def sharpe_ratio (factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    group_num = 5
    ret = ret_group(factor, shiftedPctChange, group_num)
    sharpe_ratio = np.divide(np.mean(ret.iloc[:,0]),np.std(ret.iloc[:,0])) * np.sqrt(250)
    return sharpe_ratio
#
# #信息比率
# def IR (factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
#     group_num = 5
#     ret = ret_group(factor, shiftedPctChange, group_num)
#     sharpe_ratio = np.divide(np.mean(ret.iloc[:,0]),np.std(ret.iloc[:,0])) * np.sqrt(250)
#     return sharpe_ratio

#Calmer比率(超额)
def calmar_ratio (factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    calmer_ratio = -np.divide(annual_return(factor, shiftedPctChange), max_drawdown (factor, shiftedPctChange))
    return calmer_ratio


def IC (factor :pd.DataFrame, shiftedPctChange:pd.DataFrame) -> float:
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    return corr_np




#年化alpha（%）
#Beta
#跟踪误差（%）


