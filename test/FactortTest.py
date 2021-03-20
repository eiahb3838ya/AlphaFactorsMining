import numpy as np
import pandas as pd
import sys
ROOT = "./"
sys.path.append(ROOT)
import pickle
import pandas as pd
import numpy as np
import os
from Tool import Logger, GeneralData, Factor,globalVars
from GeneticPogramming.utils import rowwise_corrcoef
from GetData import load_data, align_all_to
from GeneticPogramming.Backtest.HFBacktest import HFBacktest,HFSingleFactorBacktest

PERIOD_START = "2017-01-01"
PERIOD_END = "2019-01-01"



PROJECT_ROOT = 'F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\factors'
globalVars.initialize()
load_data("materialData",
          "F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\h5\\materialData_newData.h5")

# with open(os.path.join(os.path.join(os.path.join(PROJECT_ROOT, "20210306_18_37_48"),"found_factors"), "log(subtract_(open,high)).pickle"),'rb') as f:
#     factorData = pickle.load(f)
with open('F:\\Keira\\AlphaSignalFromMachineLearning\\Data\\factors\\20210306_18_37_48\\found_factors\\'
          'log(subtract_(open, high)).pickle', 'rb') as f:
    factorData = pickle.load(f)

factor_arr = factorData.generalData
close_ = globalVars.materialData['close'].to_DataFrame().loc[PERIOD_START:PERIOD_END]
factor_df= pd.DataFrame(factor_arr,index=close_.index, columns=close_.columns)
close_.index = close_.index.astype(str)
factor_df.index = factor_df.index.astype(str)


analysis = HFSingleFactorBacktest('facto1',factor_df, close_)
analysis.analyze()
