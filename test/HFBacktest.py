# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:02:51 2020

@author: Alfred
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')

class HFBacktest(object):
    
    def __init__(self, price, tradePoint='close'):
        self.factorDict = {}
        self.price = price
        self.tradePoint = tradePoint
        self.rts = self.price / self.price.shift(1) - 1
        self.rtsRank = self.rts.rank(method='dense', axis=1)
        
    def addFactor(self, factorName, factorExposure):
        self.factorDict[factorName] = factorExposure
        
    def orthogonalizeFactors(self):
        self.factorNameList = [key for key in self.factorDict.keys()]
        if len(self.factorNameList) == 1:
            return
        else:
            print('Start Orthogonalizing Factors.')
            for factorIdx in tqdm(range(1, len(self.factorDict)), desc='Factor to Be Orthogonalized: '):
                toRegFactor = self.factorDict[self.factorNameList[factorIdx]].copy(deep=True)
                for idxToReg in tqdm(range(factorIdx), desc='Factor to Regress: '):
                    regFactor = self.factorDict[self.factorNameList[idxToReg]]
                    for date in tqdm(toRegFactor.index.tolist(), desc='Regression Date: '):
                        notNanBool = toRegFactor.loc[date].notnull() & regFactor.loc[date].notnull()
                        lr = LinearRegression()
                        lr.fit(toRegFactor.loc[date, notNanBool].values.reshape(notNanBool.sum(), 1), 
                               regFactor.loc[date, notNanBool].values.reshape(notNanBool.sum(), 1))
                        toRegFactor.loc[date, notNanBool] = (toRegFactor.loc[date, notNanBool] - lr.coef_[0][0] * regFactor.loc[date, notNanBool] - 
                                                             lr.intercept_[0] * np.ones(notNanBool.sum()))
                self.factorDict[self.factorNameList[factorIdx]] = toRegFactor.sub(toRegFactor.mean(axis=1), axis=0).divide(toRegFactor.std(axis=1), axis=0)
        print('Orthogonalization Completed.')
            
    def analyzeSingleFactor(self, layerNum=10, positionPct=0.1):
        self.singleFactorTest = {}
        self.factorModeList = []
        self.factorIC = pd.DataFrame(index=self.rts.index[1:])
        for factorName in self.factorNameList:
            print('Single Factor Backtest for %s' % factorName)
            factorExposure = self.factorDict[factorName]
            factorAnalysis = HFSingleFactorBacktest(factorName, factorExposure, self.price, self.tradePoint)
            factorAnalysis.analyze(layerNum, positionPct)
            
            self.singleFactorTest[factorName] = {}
            self.singleFactorTest[factorName]['factorRank'] = factorAnalysis.factorRank
            self.singleFactorTest[factorName]['groupRts'] = factorAnalysis.groupRts
            self.singleFactorTest[factorName]['groupCumRts'] = factorAnalysis.groupCumRts
            self.factorModeList.append(factorAnalysis.factorMode)
            self.factorIC = pd.concat([self.factorIC, factorAnalysis.IC], axis=1)
            self.singleFactorTest[factorName]['IC'] = factorAnalysis.IC
            self.singleFactorTest[factorName]['rankIC'] = factorAnalysis.rankIC
            self.singleFactorTest[factorName]['ICPerformance'] = factorAnalysis.ICPerformance
            self.singleFactorTest[factorName]['longPosition'] = factorAnalysis.longPosition
            self.singleFactorTest[factorName]['shortPosition'] = factorAnalysis.shortPosition
            self.singleFactorTest[factorName]['longRts'] = factorAnalysis.longRts
            self.singleFactorTest[factorName]['shortRts'] = factorAnalysis.shortRts
            self.singleFactorTest[factorName]['longShortRts'] = factorAnalysis.longShortRts
            self.singleFactorTest[factorName]['performance'] = factorAnalysis.performance
            
        print('Single Factor Backtest Completed.')
        
    def analyzeMultiFactor(self, composeMode='XGBoost'):
        self.composeMultiFactor(composeMode)
        print('Multi-Factor Backtest: ')
        self.multiFactorAnalysis = HFSingleFactorBacktest('ComposedFactor', self.composedFactor, 
                                                          self.price.loc[self.composedFactor.index], self.tradePoint)
        self.multiFactorAnalysis.analyze()
        print('Multi-Factor Backtest Completed.')
        
    def composeMultiFactor(self, composeMode):
        print('Start Composing Multi-Factors...')
        print('    Composing Model: %s' % composeMode)
        if composeMode == 'XGBoost':
            composedFactor = self.calcXGBComposedFactor()
        elif composeMode == 'equalWeight':
            composedFactor = self.calcEWComposedFactor()
        self.composedFactor = composedFactor.sub(composedFactor.mean(axis=0), 
                                                 axis=1).divide(composedFactor.std(axis=0), axis=1)
        print('Composing Completed.')
    
    def calcXGBComposedFactor(self):
        composedFactor = pd.DataFrame(index=self.factorDict[self.factorNameList[0]].index[30:], 
                                      columns=self.factorDict[self.factorNameList[0]].columns)
        startTime = time.time()
        for stockIdx in tqdm(range(len(composedFactor.columns))):
            for timestampIdx in range(30, self.factorDict[self.factorNameList[0]].shape[0]):
                factorOnOneStock = np.zeros((30, len(self.factorNameList)))
                for factorIdx in range(len(self.factorNameList)):
                    factorOnOneStock[:, factorIdx] = self.factorDict[self.factorNameList[factorIdx]].iloc[timestampIdx-30:timestampIdx, stockIdx].values
                if np.isnan(factorOnOneStock).sum() > 0:
                    continue
                testFactor = factorOnOneStock[-1:, :]
                if self.factorDict[self.factorNameList[factorIdx]].index[timestampIdx][8:12] == '1019':
                    trainFactor = factorOnOneStock[:-1, :]
                    label = (self.rts.iloc[timestampIdx-29:timestampIdx, stockIdx] > 0).astype(int).values
                    XGBModel = XGBClassifier(max_depth=3, 
                                             learning_rate=0.05, 
                                             subsample=0.8, 
                                             random_state=10)
                    trainedXGBModel = XGBModel.fit(trainFactor, label)
                composedFactor.iloc[timestampIdx-30, stockIdx] = trainedXGBModel.predict_proba(testFactor)[0, -1]
                # trainFactor = factorOnOneStock[:-1, :]
                # testFactor = factorOnOneStock[-1:, :]
                # label = (self.rts.iloc[timestampIdx-29:timestampIdx, stockIdx] > 0).astype(int).values
                # XGBModel = XGBClassifier(max_depth=6, 
                #                          learning_rate=0.05, 
                #                          subsample=0.8, 
                #                          random_state=10)
                # trainedXGBModel = XGBModel.fit(trainFactor, label)
                # composedFactor.iloc[timestampIdx-30, stockIdx] = trainedXGBModel.predict_proba(testFactor)[0, -1]
                
        endTime = time.time()
        print('Time Taken to Compose Factors: %s' % (endTime - startTime))
        return composedFactor
    
    def calcEWComposedFactor(self):
        composedFactor = self.factorDict[self.factorNameList[0]] / len(self.factorNameList)
        for factorName in self.factorNameList[1:]:
            composedFactor += self.factorDict[factorName] / len(self.factorNameList)
        return composedFactor
        
        
        
class HFSingleFactorBacktest(object):
    
    def __init__(self, factorName, factorExposure, price, tradePoint='close'):
        self.factorName = factorName
        self.factorExposure = factorExposure
        self.price = price
        self.tradePoint = tradePoint
        self.rts = self.price / self.price.shift(1) - 1
        self.rtsRank = self.rts.rank(method='dense', axis=1)
        
    def analyze(self, layerNum=10, positionPct=0.1, turnoverLimit=0.5):
        print('Start Backtest for %s' % self.factorName)
        self.factorRank = self.factorExposure.rank(axis=1, method='dense')
        
        self.groupBacktest(layerNum)
    
        self.ICBacktest()

        self.generateLongShortPosition(positionPct, turnoverLimit)

        self.calcLongShortPerformance()
        self.longShortBacktest()
            
    def groupBacktest(self, layerNum):
        print('Hierarchical Bakctest for %s' % self.factorName)
        self.calcGroupRts(layerNum)
        self.plotGroupBacktestFigure()
            
    def calcGroupRts(self, layerNum):
        groupCumRts = pd.DataFrame(index=self.factorRank.index)
        groupRts = pd.DataFrame(index=self.factorRank.index)
        for layerIdx in range(1, layerNum + 1):
            groupName = 'group%s' % layerIdx
            groupPosition = pd.DataFrame(data=np.zeros(self.factorRank.shape), index=self.factorRank.index, 
                                         columns=self.factorRank.columns)
            if self.factorRank.max(skipna=True).max(skipna=True) < layerNum:
                print('Number of Layers Excess Number of Ranks.')
                return 
            elif self.factorRank.max(skipna=True).max(skipna=True) == layerNum:
                groupPosition[self.factorRank == layerIdx] = 1
            else:
                groupPosition[(self.factorRank.sub(self.factorRank.max(axis=1) * (layerIdx-1) / layerNum, axis=0) > 0) & 
                              (self.factorRank.sub(self.factorRank.max(axis=1) * layerIdx / layerNum, axis=0) <= 0)] = 1
                    
            groupRts[groupName] = np.hstack((0, np.nanmean(groupPosition.iloc[:-1, :].values * self.rts.iloc[1:, :].values, axis=1)))
            groupCumRts[groupName] = (1 + groupRts[groupName]).cumprod() - 1
        if groupCumRts.iloc[-1, 0] < groupCumRts.iloc[-1, -1]:
            self.factorMode = 1
        else:
            self.factorMode = -1
        self.groupRts = groupRts
        self.groupCumRts = groupCumRts
    
    def plotGroupBacktestFigure(self):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('%s Hierarchical Backtest' % self.factorName)
        ax1.bar(self.groupCumRts.columns, 100 * self.groupCumRts.iloc[-1], color='blue', edgecolor='black')
        ax1.set_ylabel('cum returns (%)')
        ax1.set_title('%s Hierarchical Cum Returns Bar' % self.factorName)
        for groupIdx in range(self.groupRts.shape[1]):
            ax2.plot(self.groupCumRts.index, 100 * self.groupCumRts['group%s' % (groupIdx+1)], linewidth=3)
        ax2.legend(self.groupCumRts.columns)
        ax2.set_title('%s Hierarchical Cum Returns Plot' % self.factorName)
        ax2.set_ylabel('cum returns (%)')
        ticks = np.arange(0, self.groupCumRts.shape[0], 90)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels=self.groupCumRts.index[ticks], rotation=45)
        plt.show()
        
    # def ICBacktest(self):
    #     print('IC Bakctest for %s' % self.factorName)
    #     self.calcIC()
    #     self.calcRankIC()
    #     print('%s IC performance is below:' % self.factorName)
    #     ICPerformance = pd.DataFrame( index=[0], columns=['IC_mean', 'ICIR', 'IC_tValue', 'rankIC_mean'], dtype='object')
    #     ICPerformance['IC_mean'] = self.IC.mean()
    #     ICPerformance['ICIR'] = self.IC.mean() / self.IC.std()
    #     t_stats = stats.ttest_1samp(self.IC, 0)
    #     ICPerformance['IC_tValue'] = round(t_stats.statistic, 4)
    #     ICPerformance['rankIC_mean'] = self.rankIC.mean()
    #     ICPerformance = ICPerformance.set_index(['IC_mean'])
    #     ICPerformance.index.name = None
    #     ICPerformance.columns.name = 'IC_mean'
    #     print(ICPerformance)
    #     self.ICPerformance = ICPerformance
    #
    #     self.plotICFigures()

    def ICBacktest(self):
        print('IC Bakctest for %s' % self.factorName)
        self.calcIC()
        self.calcRankIC()
        print('%s IC performance is below:' % self.factorName)
        #ICPerformance = pd.DataFrame( index=[0], columns=['IC_mean', 'ICIR', 'IC_tValue', 'rankIC_mean'], dtype='object')
        ICPerformance = pd.DataFrame(np.zeros((1, 4)))
        ICPerformance.columns = ['IC_mean', 'ICIR', 'IC_tValue', 'rankIC_mean']
        ICPerformance['IC_mean'] = self.IC.mean()
        ICPerformance['ICIR'] = self.IC.mean() / self.IC.std()
        t_stats = stats.ttest_1samp(self.IC, 0)
        ICPerformance['IC_tValue'] = round(t_stats.statistic, 4)
        ICPerformance['rankIC_mean'] = self.rankIC.mean()
        ICPerformance = ICPerformance.set_index(['IC_mean'])
        ICPerformance.index.name = None
        ICPerformance.columns.name = 'IC_mean'
        print(ICPerformance)
        self.ICPerformance = ICPerformance

        self.plotICFigures()

    #
    def calcIC(self):
        IC = pd.Series(index=self.rts.index[1:], dtype='float64', name=self.factorName)
        for dateIdx in range(len(self.rts.index[1:])):
            corrDt = pd.DataFrame(list(zip(self.factorExposure.iloc[dateIdx], self.rts.iloc[dateIdx+1]))).dropna()
            corrMat = corrDt.corr()
            if corrMat.isna().iloc[0, 1]:
                IC[dateIdx] = 0
            else:
                IC[dateIdx] = corrMat.iloc[0, 1]
        self.IC = IC
                
    def calcRankIC(self):                
        rankIC = pd.Series(index=self.rts.index[1:], dtype='float64')
        for dateIdx in range(len(self.rts.index[1:])):
            corrDt = pd.DataFrame(list(zip(self.factorRank.iloc[dateIdx], self.rtsRank.iloc[dateIdx+1]))).dropna()
            corrMat = corrDt.corr()
            if corrMat.isna().iloc[0, 1]:
                rankIC[dateIdx] = 0
            else:
                rankIC[dateIdx] = corrMat.iloc[0, 1]
        self.rankIC = rankIC
    
    def plotICFigures(self):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(self.IC.index, self.IC, linewidth=3, label='IC curve')
        ax1.axhline(y=self.IC.mean(), color='g', linewidth=3, linestyle='-', label='IC mean')
        ax1.legend()
        ax1.set_title('%s IC curve' % self.factorName)
        ticks = np.arange(0, self.IC.shape[0], 90)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels=self.IC.index[ticks], rotation=45)
        
        ax2.plot(self.rankIC.index, self.rankIC, linewidth=3, label='rankIC curve')
        ax2.axhline(y=self.rankIC.mean(), color='g', linewidth=3, linestyle='-', label='rankIC mean')
        ax2.legend()
        ax2.set_title('%s rankIC curve' % self.factorName)
        ticks = np.arange(0, self.rankIC.shape[0], 90)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels=self.rankIC.index[ticks], rotation=45)
        plt.subplots_adjust(left=0.125, right=0.9, top=0.92, wspace=0.5, hspace=0.5)
        plt.show()
        
    def generateLongShortPosition(self, positionPct, turnoverLimit):
        positionStockNum = (self.factorRank.notnull().sum(axis=1) * positionPct).astype(int)
        upperRank, lowerRank = self.generateUpperLowerRank(positionStockNum)
        
        upperPosition = self.factorRank.sub(upperRank, axis=0) >= 0
        lowerPosition = self.factorRank.sub(lowerRank, axis=0) <= 0
        
        # self.longPosition = pd.DataFrame(index=upperPosition.index, columns=upperPosition.columns)
        # self.shortPosition = pd.DataFrame(index=upperPosition.index, columns=upperPosition.columns)
        # self.longPosition.iloc[0] = upperPosition.iloc[0] if self.factorMode == 1 else lowerPosition.iloc[0]
        # self.shortPosition.iloc[0] = lowerPosition.iloc[0] if self.factorMode == 1 else upperPosition.iloc[0]
        
        # for timestampIdx in tqdm(range(1, upperPosition.shape[0]), desc='Generating Long-Short Position:'):
        #     if (upperPosition.index[timestampIdx][8:12] == '1456'):# | (upperPosition.index[timestampIdx][8:12] == '1456'):
        #         self.longPosition.iloc[timestampIdx] = upperPosition.iloc[timestampIdx] if self.factorMode == 1 else lowerPosition.iloc[timestampIdx]
        #         self.shortPosition.iloc[timestampIdx] = lowerPosition.iloc[timestampIdx] if self.factorMode == 1 else upperPosition.iloc[timestampIdx]
        #     else:
        #         self.longPosition.iloc[timestampIdx] = self.longPosition.iloc[timestampIdx-1]
        #         self.shortPosition.iloc[timestampIdx] = self.shortPosition.iloc[timestampIdx-1]
        
        if self.factorMode == 1:
            self.longPosition = upperPosition
            self.shortPosition = lowerPosition
        else:
            self.longPosition = lowerPosition
            self.shortPosition = upperPosition
            
        # self.reduceTurnover(positionStockNum, turnoverLimit)
        
    def reduceTurnover(self, positionStockNum, turnoverLimit):
        timestampList = self.factorRank.index.tolist()
        oldLongPosition = self.longPosition.iloc[0].copy()
        oldShortPosition = self.shortPosition.iloc[0].copy()
        for timestampIdx in tqdm(range(1, len(timestampList)), desc='Start Reducing Turnover:'):
            newLongPosition = self.longPosition.iloc[timestampIdx].copy()
            newShortPosition = self.shortPosition.iloc[timestampIdx].copy()
            turnoverThreshold = int(turnoverLimit * positionStockNum[timestampIdx-1])
            if (oldLongPosition ^ newShortPosition).sum() >= turnoverThreshold:
                toDeleteStockLoc = np.where((oldLongPosition==1) & (newLongPosition==0))[0]
                toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
                toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
                toAddStockLoc = np.where((oldLongPosition==0) & (newLongPosition==1))[0]
                toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
                toAddFactorRank = toAddFactorExposure.rank(method='min')
                if self.factorMode == 1:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] - 
                                                         turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
                else:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank.shape[0] - 
                                                     turnoverThreshold / 2]
                newLongPosition.iloc[toReserveStockLoc] = 1
                newLongPosition.iloc[toIgnoreStockLoc] = 0
                self.longPosition.iloc[timestampIdx] = newLongPosition
                
            if (oldShortPosition ^ newShortPosition).sum() >= turnoverThreshold:
                toDeleteStockLoc = np.where((oldShortPosition==0) & (newShortPosition==1))[0]
                toDeleteFactorExposure = self.factorExposure.iloc[timestampIdx, toDeleteStockLoc]
                toDeleteFactorRank = toDeleteFactorExposure.rank(method='min')
                toAddStockLoc = np.where((oldShortPosition==0) & (newShortPosition==1))[0]
                toAddFactorExposure = self.factorExposure.iloc[timestampIdx, toAddStockLoc]
                toAddFactorRank = toAddFactorExposure.rank(method='min')
                if self.factorMode == 1:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank <= turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank >= toDeleteFactorRank - 
                                                     turnoverThreshold / 2]
                else:
                    toReserveStockLoc = toDeleteStockLoc[toDeleteFactorRank >= toAddFactorRank.shape[0] - 
                                                         turnoverThreshold / 2]
                    toIgnoreStockLoc = toAddStockLoc[toAddFactorRank <= turnoverThreshold / 2]
                newShortPosition.iloc[toReserveStockLoc] = 1
                newShortPosition.iloc[toIgnoreStockLoc] = 0
                self.shortPosition.iloc[timestampIdx] = newShortPosition
            oldLongPosition = newLongPosition.copy()
            oldShortPosition = newShortPosition.copy()
        
    def generateUpperLowerRank(self, positionStockNum):
        upperRank = np.zeros(self.factorRank.shape[0])
        lowerRank = np.zeros(self.factorRank.shape[0])
        for dateIdx in range(self.factorRank.shape[0]):
            numOfRanks = self.factorRank.iloc[dateIdx].value_counts()
            numOfRanks = numOfRanks.reindex(index=numOfRanks.index.sort_values(ascending=False))
            upperStopRankIdx = 0
            upperNum = numOfRanks.iloc[upperStopRankIdx]
            while (upperNum < positionStockNum[dateIdx]):
                upperStopRankIdx += 1
                upperNum += numOfRanks.iloc[upperStopRankIdx]
            upperRank[dateIdx] = numOfRanks.index[upperStopRankIdx]
            
            lowerStopRankIdx = -1
            lowerNum = numOfRanks.iloc[lowerStopRankIdx]
            while (lowerNum < positionStockNum[dateIdx]):
                lowerStopRankIdx += -1
                lowerNum += numOfRanks.iloc[lowerStopRankIdx]
            lowerRank[dateIdx] = numOfRanks.index[lowerStopRankIdx]
        return upperRank, lowerRank
    
    def calcLongShortPerformance(self):
        stampTaxRate = 0
        longRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.longPosition.index)
        shortRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.longPosition.index)
        longTurnover = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.longPosition.index)
        shortTurnover = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.longPosition.index)
        longShortRts = pd.Series(data=np.zeros(self.longPosition.shape[0]), index=self.longPosition.index)
        oldLongPosition = self.longPosition.iloc[0]
        oldShortPosition = self.shortPosition.iloc[0]
        shortedPosition = pd.Series(data=np.full((self.shortPosition.shape[1],), False), index=self.shortPosition.columns)
        for dateIdx in range(1, self.longPosition.shape[0]):
            newLongPosition = self.longPosition.iloc[dateIdx]
            newShortPosition = self.shortPosition.iloc[dateIdx]
            longTurnoverPosition = oldLongPosition ^ newLongPosition
            shortTurnoverPosition = shortedPosition ^ oldShortPosition
            longTurnover.iloc[dateIdx-1] = longTurnoverPosition.sum() / oldLongPosition.sum()
            shortTurnover.iloc[dateIdx-1] = shortTurnoverPosition.sum() / oldShortPosition.sum()
            longRts.iloc[dateIdx] = (((self.price.iloc[dateIdx] * (oldLongPosition - longTurnoverPosition * stampTaxRate)).sum() - (self.price.iloc[dateIdx-1] * oldLongPosition).sum()) / 
                                      ((self.price.iloc[dateIdx-1] * oldLongPosition).sum() + (self.price.iloc[dateIdx] * longTurnoverPosition).sum() * stampTaxRate))
            shortRts.iloc[dateIdx] = (((self.price.iloc[dateIdx-1] * (oldShortPosition - shortTurnoverPosition * stampTaxRate)).sum() - (self.price.iloc[dateIdx] * oldShortPosition).sum()) / 
                                      ((self.price.iloc[dateIdx] * oldShortPosition).sum() + (self.price.iloc[dateIdx-1] * shortTurnoverPosition).sum() * stampTaxRate))
            shortedPosition = oldShortPosition
            oldLongPosition = newLongPosition
            oldShortPosition = newShortPosition
        longShortRts = longRts + shortRts
        
        self.longRts = longRts
        self.shortRts = shortRts
        self.longShortRts = longShortRts
        self.longTurnover = longTurnover
        self.shortTurnover = shortTurnover
    
    def longShortBacktest(self):
        print('Long Short Portfolio Backtest for %s' % self.factorName)
        performance = pd.DataFrame(index=[0], columns=['cumRts(%)', 'annualVol(%)', 'maxDrawdown(%)', 'winRate(%)', 'SharpeRatio'])
        longShortNetValue = (1 + self.longShortRts).cumprod()
        performance['cumRts(%)'] = round(100 * (longShortNetValue.iloc[-1] - 1), 2)
        performance['annualVol(%)'] = round(100 * self.longShortRts.std() * ((237*250)**0.5), 2)
        maxNetValue = longShortNetValue.expanding().max()
        self.drawdown = longShortNetValue / maxNetValue - 1
        performance['maxDrawdown(%)'] = round(-100 * self.drawdown.min(), 2)
        performance['winRate(%)'] = round(100 * (self.longShortRts > 0).sum() / self.longShortRts.shape[0], 2)
        performance['SharpeRatio'] = round(self.longShortRts.mean() / self.longShortRts.std(), 4)
        performance.set_index(['cumRts(%)'], inplace=True)
        performance.index.name = None
        performance.columns.name = 'cumRts(%)'
        
        self.plotRtsFigure()
        self.plotTurnover()
        
        print(performance)
        self.performance = performance
    
    def plotRtsFigure(self):
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 10))
        ax1.bar(self.longShortRts.index, 100 * self.longShortRts, color='red', label='single term long-short rts')
        ax1.set_ylabel('single term returns(%)')
        ax1.legend()
        ax1Right = ax1.twinx()
        ax1Right.plot((1 + self.longShortRts).cumprod() - 1, linewidth=3, label='long-short cum rts')
        ax1Right.plot((1 + self.longRts).cumprod() - 1, linewidth=3, label='long cum rts')
        ax1Right.plot((1 + self.shortRts).cumprod() - 1, linewidth=3, label='short cum rts')
        ax1Right.set_ylabel('cum returns(%)')
        ax1Right.legend()
        ticks = np.arange(0, self.longShortRts.shape[0], 90)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels=self.longShortRts.index[ticks])
        fig.autofmt_xdate(rotation=45)
        
        ax2.plot((1 + self.longShortRts).cumprod() - 1, linewidth=3, color='red', label='long-short cum returns')
        ax2.set_ylabel('cum returns(%)')
        ax2.legend()
        ax2Right = ax2.twinx()
        ax2Right.fill_between(self.drawdown.index, 100 * self.drawdown, 0, color='grey', label='drawdown')
        ax2Right.set_ylabel('drawdown(%)')
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels=self.longShortRts.index[ticks])
        fig.autofmt_xdate(rotation=45)
        ax2Right.legend()
        
        plt.suptitle('%s Backtest Performance: ' % self.factorName)
        plt.show()
        
    def plotTurnover(self):
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(20, 25))
        plt.subplots_adjust(left=0.125, right=0.9, bottom= 0.2, top=0.9, wspace=0.2, hspace=0.5)
        
        ax1.fill_between(self.longTurnover.index[1:], self.longTurnover.iloc[1:] * 100, 0, facecolor='pink', alpha=0.4)
        ticks = np.arange(0, self.longTurnover.shape[0], 90)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels=self.longTurnover.index[ticks+1], rotation=45)
        ax1.set_title('Turnover on the Long Side')
        
        ax2.fill_between(self.shortTurnover.index[1:], self.shortTurnover.iloc[1:] * 100, 0, facecolor='green', alpha=0.4)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels=self.longTurnover.index[ticks+1], rotation=45)
        ax2.set_title('Turnover on the Short Side')
        
        ax3.fill_between(self.longTurnover.index[1:], self.longTurnover.iloc[1:] * 100, 0, facecolor='pink', alpha=0.4)
        ax3.fill_between(self.shortTurnover.index[1:], self.shortTurnover.iloc[1:] * 100, 0, facecolor='green', alpha=0.4)
        ax3.legend(['Long', 'Short'])
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(labels=self.longTurnover.index[ticks+1], rotation=45)
        ax3.set_title('Turnover on Both Sides')
        
        plt.suptitle('Turnover Analysis')
        plt.show()
        
#%%
# close = pd.read_csv('filled_data/close_filled.csv')
# close.set_index(['time'], inplace=True)
# close.index = close.index.astype(str)
# close = close.iloc[20:150]
    
# amountEntropy = pd.read_csv('amountEntropy.csv')
# amountEntropy.set_index(['time'], inplace=True)
# amountEntropy = amountEntropy.iloc[20:250, :150]
# amountEntropy.index = amountEntropy.index.astype(str)
# amountEntropy = amountEntropy.replace(np.inf, 10000000)
# amountEntropy = amountEntropy.replace(-np.inf, -10000000)
# amountEntropy = amountEntropy.sub(np.nanmean(amountEntropy, axis=1), axis=0).divide(np.nanstd(amountEntropy), axis=0)

# activeBuyRatio = pd.read_csv('activeBuyRatio.csv')
# activeBuyRatio.set_index(['time'], inplace=True)
# activeBuyRatio = activeBuyRatio.iloc[20:150]
# activeBuyRatio.index = activeBuyRatio.index.astype(str)
# activeBuyRatio = activeBuyRatio.replace(np.inf, 10000000)
# activeBuyRatio = activeBuyRatio.replace(-np.inf, -10000000)
# activeBuyRatio = activeBuyRatio.sub(np.nanmean(activeBuyRatio, axis=1), axis=0).divide(np.nanstd(activeBuyRatio), axis=0)

# #%%
# weightedCloseRatio = pd.read_csv('weightedCloseRatio.csv')
# weightedCloseRatio.set_index(['time'], inplace=True)
# weightedCloseRatio = weightedCloseRatio.iloc[20:]
# weightedCloseRatio.index = weightedCloseRatio.index.astype(str)
# weightedCloseRatio = weightedCloseRatio.sub(weightedCloseRatio.mean(axis=1), axis=0).divide(weightedCloseRatio.std(axis=1), axis=0)

# #%%
# BIAS = pd.read_csv('BIAS.csv')
# BIAS.set_index(['time'], inplace=True)
# BIAS = BIAS.iloc[20:]
# BIAS.index = close.index
# BIAS = BIAS.sub(BIAS.mean(axis=1), axis=0).divide(BIAS.std(axis=1), axis=0)

#%%
# analysis = HFSingleFactorBacktest('activeBuyRatio', activeBuyRatio, close)
# analysis.analyze()

# analysis = HFSingleFactorBacktest('weightedCloseRatio', weightedCloseRatio, close)
# analysis.analyze()

# analysis = HFSingleFactorBacktest('BIAS', BIAS, close)
# analysis.analyze()
    
# #%%
# analysis = HFBacktest(close)
# analysis.addFactor('activeBuyRatio', activeBuyRatio)
# analysis.addFactor('BIAS', BIAS)

# #%%
# analysis.orthogonalizeFactors()

# #%%
# analysis.analyzeSingleFactor()

# #%%
# analysis.analyzeMultiFactor('XGBoost')
# analysis.analyzeMultiFactor('equalWeight')