# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

在這裡定義所有對單因子評價的方式，輸入的因子是已經輸出成 GeneralData 的因子值
而後面也可以使用其他的數據， typically 就是用收益率做 ic, icir 等等
回傳一個數值

@author: eiahb
"""


import os

import numpy as np
from copy import copy, deepcopy
from Tool.GeneralData import GeneralData
import pandas as pd
from GeneticPogramming.utils import rowwise_corrcoef
from sklearn import metrics



# evaluate function 评价函数
def ic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    # print("fail eval {} of {} days".format(corr_np.mask.sum(), shiftedPctChange.generalData.shape[0]))
    if corr_np.mask.sum() > len(corr_np)/2:
        print("trouble factor with unevaluate days {} out of {} days".format(corr_np.mask.sum(), len(corr_np)))
        ic = -1
    else:
        ic = np.abs(np.nanmean(corr_np))
    return(ic)

def rankic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:
    corr_np = rowwise_corrcoef(np.argsort(factor.generalData,axis=1), np.argsort(shiftedPctChange.generalData,axis=1))
    # print("fail eval {} of {} days".format(corr_np.mask.sum(), shiftedPctChange.generalData.shape[0]))
    if corr_np.mask.sum() > len(corr_np)/2:
        print("trouble factor with unevaluate days {} out of {} days".format(corr_np.mask.sum(), len(corr_np)))
        ic = -1
    else:
        ic = np.nanmean(corr_np)
    return(ic)

def icir_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float: 
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    if not corr_np.mask.all():
        ic = np.nanmean(corr_np)

        icir = ic / np.nanstd(corr_np)
        icir_y = icir * np.sqrt(250)
        return(icir_y)
    else:
        return(-1)    



def ic_long_ir(factor : GeneralData, shiftedPctChange:GeneralData) -> float:
    holding_period = 5
    group_num = 5
    num_prod = int(len(factor.columnNames) / group_num)
    backtest_period = factor.timestamp
    n = len(backtest_period )
    assert n!= 0
    ret = pd.DataFrame(index=backtest_period, columns=list(range(1, 1 + group_num)), type='object')
    for i in range(n):
        change_date = backtest_period[i]
        if np.mod(i, holding_period) == 0:  # 换仓日,排序记录每组的序号
            prod_ID_group = dict()
            rank = np.argsort(-factor.generalData[i, :])  # 倒序
            for group in range(1, group_num):
                prod_ID_group[group] = rank[(group - 1) * num_prod:group * num_prod]
            prod_ID_group[group_num] = rank[(group_num - 1) * num_prod:]
        daily_return = shiftedPctChange.generalData[i, :]
        for group in range(1, group_num + 1):
            ret.loc[change_date, group] = np.nanmean(daily_return[prod_ID_group[group]])

    if rankic_evaluator(factor, shiftedPctChange) > 0.3:
        return np.divide(np.mean(ret.iloc[:,0]),np.std(ret.iloc[:,0])) * np.sqrt(250)
    # elif rankic_evaluator(factor, shiftedPctChange) < -0.3:
    #     return np.divide(np.mean(ret.iloc[:,-1]), np.std(ret.iloc[:,-1])) * np.sqrt(250)
    else:
        return -1000


def long_return(factor : GeneralData, shiftedPctChange:GeneralData) -> float:
    holding_period = 5
    group_num = 5
    num_prod = int(len(factor.columnNames)/group_num)
    backtest_period = factor.timestamp
    ret = pd.DataFrame(index = backtest_period, columns=np.array(range(1,1+group_num)),dtype='object')
    for i in range(len(backtest_period)):
        change_date = backtest_period[i]
        if np.mod(i,holding_period) == 0:#换仓日,排序记录每组的序号
            prod_ID_group = dict()
            rank = np.argsort(-factor.generalData[i, :])  #倒序
            for group in range(1, group_num):
                prod_ID_group[group] = rank[(group-1)*num_prod:group*num_prod]
            prod_ID_group[group_num] = rank[(group_num - 1) * num_prod:]
        daily_return = shiftedPctChange.generalData[i, :]
        for group in range(1, group_num + 1):
            ret.loc[change_date,group] = np.nanmean(daily_return[prod_ID_group[group]])
    nav = (ret+1).cumprod()
    total_return = nav.iloc[-1,:]-1
    return max(total_return.iloc[0],total_return.iloc[group_num-1])
    # return (((total_return.iloc[0]+1)*1/3)-1)




def long_short_return(factor : GeneralData, shiftedPctChange:GeneralData) -> float:
    holding_period = 5
    group_num = 5
    num_prod = int(len(factor.columnNames)/group_num)
    backtest_period = factor.timestamp
    ret = pd.DataFrame(index = backtest_period, columns=np.array(range(1,1+group_num)),dtype='object')
    for i in range(len(backtest_period)):
        change_date = backtest_period[i]
        if np.mod(i,holding_period) == 0:#换仓日,排序记录每组的序号
            prod_ID_group = dict()
            rank = np.argsort(-factor.generalData[i, :])  #倒序
            for group in range(1, group_num):
                prod_ID_group[group] = rank[(group-1)*num_prod:group*num_prod]
            prod_ID_group[group_num] = rank[(group_num - 1) * num_prod:]
        daily_return = shiftedPctChange.generalData[i, :]
        for group in range(1, group_num + 1):
            ret.loc[change_date,group] = np.nanmean(daily_return[prod_ID_group[group]])
    nav = (ret+1).cumprod()
    # print(nav)
    # print(nav.iloc[-1,:])
    # print('end')
    # print(nav.iloc[0,:])
    total_return = nav.iloc[-1,:] - 1
    return abs(total_return.iloc[0]-total_return.iloc[group_num-1])


def mutual_info(factor : GeneralData, shiftedPctChange:GeneralData) -> float:
    num = 10000
    def label(A,num = 10000):
        Min = np.nanmin(A)
        Max = np.nanmax(A)
        delta = (Max-Min)/num
        A_label = pd.DataFrame(index =range(len(A)), columns = ['label'])
        for i in range(num-1):
            A_label.loc[(A>=Min+i*delta)&(A<Min+(i+1)*delta),'label'] = i+1
        A_label.loc[A>=Min+(num-1)*delta,'label'] = num
        return A_label
    backtest_period = factor.timestamp
    mutual_info = pd.DataFrame(index=backtest_period, columns=['mutual_info'])
    for i in range(len(backtest_period)):
        A = factor.generalData[i, :]
        B = shiftedPctChange.generalData[i, :]
        M = pd.concat([label(A,num), label(B,num)], axis=1)
        mutual_info.iloc[i,0] = metrics.normalized_mutual_info_score(M.iloc[:, 0], M.iloc[:, 1])
    return mutual_info.mean()