# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:00:18 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
from tool.GeneralData import GeneralData
import copy
from genetic_programming import utils  # get_strided, get_maskAllNaN

import warnings

warnings.filterwarnings("ignore")


def delay(this: GeneralData, aNum: int = 1) -> GeneralData:
    assert aNum >= 0
    outputToReturn = copy.copy(this)
    tmp_copy = outputToReturn.generalData.copy()

    # tmp_copy[-aNum:, :] = np.nan
    roll = np.roll(tmp_copy, aNum, axis=0)
    roll[: aNum, :] = tmp_copy[:aNum, :]
    outputToReturn.generalData = roll
    return outputToReturn


# 𝑑𝑒𝑙𝑡𝑎(𝑎, 𝑏) 𝑎 − 𝑑𝑒𝑙𝑎𝑦(𝑎, 𝑏)
def delta(this: GeneralData, aNum: int = 1) -> GeneralData:
    assert aNum >= 0
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    tmp_copy = outputToReturn.generalData.copy()

    # tmp_copy[-aNum:, :] = np.nan
    roll = np.roll(tmp_copy, aNum, axis=0)
    roll[: aNum, :] = tmp_copy[:aNum, :]

    # outputToReturn.generalData = np.subtract(tmp_copy, np.roll(tmp_copy, aNum, axis=0))
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.subtract(tmp_copy, roll)
    return outputToReturn


# 𝑑𝑒𝑙𝑡𝑎(𝑎, 𝑏) 𝑎 − 𝑑𝑒𝑙𝑎𝑦(𝑎, 𝑏)
def pct_change(this: GeneralData, aNum: int = 1) -> GeneralData:
    assert aNum >= 0
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    tmp_copy = outputToReturn.generalData.copy()

    # tmp_copy[-aNum:, :] = np.nan
    roll = np.roll(tmp_copy, aNum, axis=0)
    roll[: aNum,:] = tmp_copy[:aNum, :]

    # delay = np.roll(tmp_copy, aNum, axis=0)
    # delta = np.subtract(tmp_copy, np.roll(tmp_copy, aNum, axis=0))
    delay = roll
    delta = np.subtract(tmp_copy, roll)


    outputToReturn.generalData = np.divide(delta, delay)

    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn


# def ts_argmax(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0

#     outputToReturn = copy.copy(this)

#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     masked, isAllNaN = utils.get_maskAllNaN(strided)
#     arg = np.argmax(masked, axis = 1).astype(float)
#     arg[isAllNaN] = np.nan

#     outputToReturn.generalData = arg

#     return outputToReturn

# def ts_argmin(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0

#     outputToReturn = copy.copy(this)

#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     masked, isAllNaN = utils.get_maskAllNaN(strided)
#     arg = np.argmin(masked, axis = 1).astype(float)
#     arg[isAllNaN] = np.nan

#     outputToReturn.generalData = arg

#     return outputToReturn


# # 𝑡𝑠_𝑎𝑟𝑔𝑚𝑎𝑥𝑚𝑖𝑛(𝑎, 𝑏) 2 过去 b 天 a 最大值的下标-过去 b 天 a 最小值的下标
# def ts_argmaxmin(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0

#     outputToReturn = copy.copy(this)

#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     masked, isAllNaN = utils.get_maskAllNaN(strided)
#     argmax = np.argmax(masked, axis = 1).astype(float)
#     argmin = np.argmin(masked, axis = 1).astype(float)
#     arg = np.subtract(argmax, argmin)
#     arg[isAllNaN] = np.nan

#     outputToReturn.generalData = arg

#     return outputToReturn

def ts_max(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0

    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    max_ = np.nanmax(strided, axis=1)
    outputToReturn.generalData = max_
    return outputToReturn


def ts_min(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    min_ = np.nanmin(strided, axis=1)
    outputToReturn.generalData = min_
    return outputToReturn

def ts_decay_linear(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    todivide = (1+rollingDaysN)*rollingDaysN/2
    weight = np.array(range(1,rollingDaysN+1))/todivide
    toReturn = np.average(strided, axis=1,weights= weight)
    outputToReturn.generalData = toReturn
    return outputToReturn

def ts_argmax(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    temparray = np.zeros((strided.shape[0], strided.shape[2]))
    for i in range(strided.shape[0]):
        tempArg = np.argsort(strided[i],axis=0)
        argmax = tempArg[-1,:]
        temparray[i] = argmax
    outputToReturn.generalData = temparray
    return outputToReturn

def ts_argmin(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    temparray = np.zeros((strided.shape[0], strided.shape[2]))
    for i in range(strided.shape[0]):
        tempArg = np.argsort(strided[i],axis=0)
        argmax = tempArg[0,:]
        temparray[i] = argmax
    outputToReturn.generalData = temparray
    return outputToReturn


# 𝑡𝑠_𝑚𝑎𝑥𝑚𝑖𝑛_𝑛𝑜𝑟𝑚(𝑎, 𝑏)过去 b 天 a 的 maxmin 标准化，即
# (𝑎 − 𝑡𝑠_𝑚𝑖𝑛(𝑎, 𝑏))/(𝑡𝑠_𝑚𝑎𝑥(𝑎, 𝑏) − 𝑡𝑠_𝑚𝑖𝑛(𝑎, 𝑏))
def ts_maxmin_norm(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    max_ = np.nanmax(strided, axis=1)
    min_ = np.nanmin(strided, axis=1)
    toScale = np.subtract(toStride2DArray, min_)
    scaler = np.subtract(max_, min_)

    # if the min and the max are the same value, the function will return 0,
    # witch makes sense because this means the max and min values are the same through out the rolling days
    # and the factor x - min(x) should be represented by 0
    outputToReturn.generalData = np.divide(toScale, scaler, out=np.full_like(toScale, 0), where=scaler != 0)

    return outputToReturn


def ts_mean(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    mean = np.nanmean(strided, axis=1)
    outputToReturn.generalData = mean
    return outputToReturn


def ts_std(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    std = np.nanstd(strided, axis=1)
    outputToReturn.generalData = std
    return outputToReturn


def ts_zscore(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)
    mean = ts_mean(this, rollingDaysN)
    std = ts_std(this, rollingDaysN)
    outputToReturn.generalData = np.divide(mean.generalData,std.generalData)
    return outputToReturn
#
# def ts_median(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     nanmedian = np.nanmedian(strided, axis=1)
#     outputToReturn.generalData = nanmedian
#     return outputToReturn


def ts_prod(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)

    masked, isAllNaN = utils.get_maskAllNaN(strided)

    cumprod = np.nanprod(strided, axis=1)
    cumprod[isAllNaN] = np.nan

    outputToReturn.generalData = cumprod
    return outputToReturn


def ts_sum(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    outputToReturn = copy.copy(this)
    toStride2DArray = outputToReturn.generalData
    strided = utils.get_strided(toStride2DArray, rollingDaysN)

    masked, isAllNaN = utils.get_maskAllNaN(strided)

    sum_ = np.nansum(strided, axis=1)
    sum_[isAllNaN] = np.nan

    outputToReturn.generalData = sum_
    return outputToReturn


# def ts_rank(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0
#     outputToReturn = copy.copy(this)
#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#
#     masked, isAllNaN = utils.get_maskAllNaN(strided)
#
#     rank = np.argmax(np.argsort(strided, axis=1), axis=1).astype(float)
#     rank[isAllNaN] = np.nan
#     outputToReturn.generalData = rank + 1
#     return outputToReturn
#
# #
# # (𝑎 − 𝑡𝑠_𝑚𝑒𝑎𝑛(𝑎, 𝑏))/𝑡𝑠_𝑠𝑡𝑑(𝑎, 𝑏)
# def ts_t(this: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert rollingDaysN >= 0
#
#     outputToReturn = copy.copy(this)
#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     mean = np.nanmean(strided, axis=1)
#     std = np.nanstd(strided, axis=1)
#     toScale = np.subtract(toStride2DArray, mean)
#     outputToReturn.generalData = np.divide(toScale, std, out=np.full_like(toScale, 0), where=std != 0)
#     return outputToReturn





































