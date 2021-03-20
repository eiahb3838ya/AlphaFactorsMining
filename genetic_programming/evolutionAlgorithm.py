# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:54:46 2021
定義遺傳算法的進化的算法，對應的是 deap.algorithm

@author: eiahb
"""
from time import time
import numpy.random as random
from deap import tools
from scipy.stats import rankdata
import pandas as pd

#%% easimple
def easimple(toolbox, stats, logbook, evaluate, logger,\
             N_POP = 100, N_GEN = 7, CXPB = 0.6, MUTPB = 0.2):
    # pop = toolbox.population(n = N_POP)

    # tic = time()
    # logger.info('start easimple at {:.2f}'.format(tic))
    # logger.info('evaluating initial pop......start')

    # 對初始種群進行適應度評價 
    # fitnesses = toolbox.map(evaluate, pop)

    times = 2
    POP = list()
    Fitnesses = list()
    for i in range(times):
        pop = toolbox.population(n=N_POP)
        POP = POP + pop

        tic = time()
        logger.info('start easimple at {:.2f} the {}th times' .format(tic, i))
        logger.info('evaluating initial pop......start')

        # 對初始種群進行適應度評價
        fitnesses = toolbox.map(evaluate, pop)
        Fitnesses = Fitnesses + fitnesses
    Fitnesses_for_rank = [-abs(fit[0]) for fit in Fitnesses]  # 都变为负的，因为rankdata是从小到大排序，但是我们希望最大的为1
    rank = rankdata(Fitnesses_for_rank)
    summary = pd.DataFrame([Fitnesses, rank, POP], index=['Fitness', 'rank', 'POP']).T
    pop = list(summary.loc[summary['rank'] <= N_POP, 'POP'].values)
    fitnesses = list(summary.loc[summary['rank'] <= N_POP, 'Fitness'].values)




        
    for i, (ind, fit) in enumerate(zip(pop, fitnesses)):
        # 將適應度給他成績單
        ind.fitness.values = fit
    toc = time()
    logger.info('evaluating initial pop......done with {:.5f} sec'.format(toc-tic))
    record = stats.compile(pop)
    logger.info("The initial record:{}".format(str(record)))
    
    # start evolution
    # 開始進化
    for gen in range(N_GEN):
        # 配种选择
        # 這裡使用簡單 兩倍複製
        offspring = toolbox.select(pop, 2*N_POP)
        offspring = list(map(toolbox.clone, offspring)) # 复制个体，供交叉变异用
        
        # 对选出的育种族群两两进行交叉，对于被改变个体，删除其适应度值
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        # 对选出的育种族群进行变异，对于被改变个体，删除适应度值
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
          
        # 对于被改变的个体，重新评价其适应度        
        tic = time()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.info('start evaluate for {}th Generation new {} individuals......'.format(gen, len(invalid_ind)))
        fitnesses = toolbox.map(evaluate, invalid_ind)   
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
            if(fit[0]>0.4):
                # get something useful
                # 找到合格的因子
                logger.info('got a expr useful in gen:{}, end gp algorithm'.format(gen))
                return(True, ind, logbook)
        toc = time()
        logger.info('evaluate for {}th Generation new individual......done with {:.5f} sec'.format(gen, toc-tic))
        
        # select best 环境选择保留精英
        # 只留下 N_POP 個菁英
        pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
        
        # 记录数据
        record = stats.compile(pop)
        logger.info("The {} th record:{}".format(gen, str(record)))
        logbook.record(gen=gen, **record)
    ind = tools.selBest(offspring, 1, fit_attr='fitness')[0]
    logger.info('none expr useful, terminate easimple')
    logger.info('end easimple {:.2f}'.format(tic))
    
    # 跑完都沒找到合格的因子，回傳最好的那個
    return(False, ind, logbook)