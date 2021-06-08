# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:27:20 2020

@author: Evan Hu (Yi Fan Hu)

"""


import numpy as np
from tool.GeneralData import GeneralData
import copy
import warnings
warnings.filterwarnings("ignore")

# def max_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape
#
#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
#     return outputToReturn
#
# def min_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape
#
#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.minimum(this.generalData, that.generalData)
#     return outputToReturn

def add_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.add(this.generalData, that.generalData)
    return outputToReturn

def subtract_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.subtract(this.generalData, that.generalData)
    return outputToReturn

def multiply_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.multiply(this.generalData, that.generalData)
    return outputToReturn

def divide_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape
    nrows, ncols = this.generalData.shape
    nan2DArray = np.full((nrows, ncols), np.nan)
    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.divide(this.generalData, that.generalData,
                                           out = np.full_like(nan2DArray, 10000),where = that.generalData!=0)
    return outputToReturn

def rank_sub(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape
    outputToReturn = copy.copy(this)
    this = copy.copy(this)
    that = copy.copy(that)
    thisTemp = (np.argsort(np.argsort(this.generalData)) + 1) \
                                 / this.generalData.shape[1]
    thatTemp = (np.argsort(np.argsort(that.generalData)) + 1) \
                                / this.generalData.shape[1]
    outputToReturn.generalData = thisTemp - thatTemp
    return outputToReturn


def rank_div(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape
    outputToReturn = copy.copy(this)
    this = copy.copy(this)
    that = copy.copy(that)
    thisTemp = (np.argsort(np.argsort(this.generalData)) + 1) \
                                 / this.generalData.shape[1]
    thatTemp = (np.argsort(np.argsort(that.generalData)) + 1) \
                                / this.generalData.shape[1]
    outputToReturn.generalData = np.divide(thisTemp,thatTemp)
    return outputToReturn


# def max_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape

#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
#     return outputToReturn

# def max_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape

#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
#     return outputToReturn