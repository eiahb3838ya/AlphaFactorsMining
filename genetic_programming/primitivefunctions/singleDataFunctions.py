# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:41:16 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
from tool.GeneralData import GeneralData
import copy
import warnings

warnings.filterwarnings("ignore")


# def sign(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.sign(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn


############################################### cause of overflow ##############################################
# def exp(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.exp(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn
############################################### cause of overflow ##############################################

# 带符号的对数，𝑠𝑖𝑔𝑛(𝑎) ∗ 𝑙𝑜𝑔(𝑎𝑏𝑠(𝑎))
# def s_log(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     #
#     outputToReturn.generalData = np.sign(np.log(np.abs(outputToReturn.generalData)+1))
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn

# # 带符号的开方，𝑠𝑖𝑔𝑛(𝑎) ∗ 𝑠𝑞𝑟𝑡(𝑎𝑏𝑠(𝑎))
# def s_sqrt(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.sign(outputToReturn.generalData) * np.sqrt(np.abs(outputToReturn.generalData))
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn


# Return the element-wise square of the input.
# def square(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.square(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn


# # Return the cube-root of an array, element-wise.
# def cbrt(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.cbrt(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     # return outputToReturn


# def cube(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.power(outputToReturn.generalData, 3)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn


def log(this: GeneralData) -> GeneralData:
    outputToReturn = copy.copy(this)
    min_axis0 = np.nanmin(outputToReturn.generalData, axis=0, keepdims=True)
    outputToReturn.generalData = np.add(outputToReturn.generalData, np.abs(min_axis0)) + 1
    outputToReturn.generalData = np.log(outputToReturn.generalData)
    return outputToReturn


def abs_(this: GeneralData) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.abs(outputToReturn.generalData)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn


# a 减截面均值
# def demean(this: GeneralData) -> GeneralData:
#     outputToReturn = copy.copy(this)
#     mean_axis1 = np.nanmean(outputToReturn.generalData, axis=1, keepdims=True)
#     outputToReturn.generalData = np.subtract(outputToReturn.generalData, mean_axis1)
#     return outputToReturn

#
# def negative(this: GeneralData) -> GeneralData:
#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.multiply(outputToReturn.generalData, -1)
#     return outputToReturn


# def positive(this: GeneralData) -> GeneralData:
#     outputToReturn = copy.copy(this)
#     # outputToReturn.generalData = np.multiply(outputToReturn.generalData, -1)
#     return(outputToReturn)

def rank(this: GeneralData) -> GeneralData:
    outputToReturn = copy.copy(this)
    outputToReturn.generalData = (np.argsort(np.argsort(outputToReturn.generalData)) + 1)\
                                 /outputToReturn.generalData.shape[1]
    return outputToReturn




def scale(this: GeneralData) -> GeneralData:
    outputToReturn = copy.copy(this)
    sum_axis1 = np.nansum(outputToReturn.generalData, axis=1, keepdims=True)
    outputToReturn.generalData = np.divide(outputToReturn.generalData, sum_axis1)
    return outputToReturn

def sigmoid(this: GeneralData) -> GeneralData:
    outputToReturn = copy.copy(this)
    sigmoid = 1 / (1 + np.exp(np.multiply(outputToReturn.generalData, -1)))
    outputToReturn.generalData = sigmoid
    return outputToReturn
#
#
#
# def center_scale(this: GeneralData) -> GeneralData:
#     outputToReturn = copy.copy(this)
#     mean_axis1 = np.nanmean(outputToReturn.generalData, axis=1, keepdims=True)
#     std_axis1 = np.nanstd(outputToReturn.generalData, axis=1, keepdims=True)
#     toScale = np.subtract(outputToReturn.generalData, mean_axis1)
#     outputToReturn.generalData = np.divide(toScale, std_axis1, out=np.full_like(toScale, 0), where=std_axis1 != 0)
#     return outputToReturn

# Trigonometric sine, element-wise.
# def sin(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.sin(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn

# # Cosine element-wise.
# def cos(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.cos(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn

# # Compute tangent element-wise.
# def tan(this: GeneralData) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     outputToReturn.generalData = np.tan(outputToReturn.generalData)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn






















































