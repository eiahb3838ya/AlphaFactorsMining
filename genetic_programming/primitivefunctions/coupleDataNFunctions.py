import numpy as np
from Tool.GeneralData import GeneralData
import copy
from GeneticPogramming import utils #get_strided, get_maskAllNaN

# import scipy.stats as stats
from GeneticPogramming.utils import rowwise_corrcoef, rowwise_cov

import warnings
warnings.filterwarnings("ignore")


def ts_corr(this: GeneralData, that: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    assert isinstance(this.generalData, np.ndarray) == True
    assert isinstance(that.generalData, np.ndarray) == True
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)

    toStride2DArrayThis = this.generalData
    toStride2DArrayThat = that.generalData

    # column = toStride2DArrayThis.shape[1]
    stridedThis = utils.get_strided(toStride2DArrayThis, rollingDaysN)
    stridedThat = utils.get_strided(toStride2DArrayThat, rollingDaysN)
    temparray = np.zeros((stridedThis.shape[0],stridedThis.shape[2]))
    for i in range(stridedThis.shape[0]):
        tempThis = stridedThis[i].T
        tempThat = stridedThat[i].T
        # corr = np.corrcoef(tempThis, tempThat)
        # for j in range(stridedThis.shape[2]):
        corr = rowwise_corrcoef(tempThis,tempThat)
        temparray[i] = corr
        # print(corr)
        # print(i)
    print('done')

    outputToReturn.generalData = temparray
            # print(stats.pearsonr(stridedThis[i,j,:], stridedThis[i,j,:]))
            # # print(corr[i,stridedThis.shape[2]+i])
            # outputToReturn.generalData[i,j] = corr[i,stridedThis.shape[2]+i]
            # outputToReturn.generalData[i, j] = 1
    #print('over')
    return outputToReturn

def ts_cov(this: GeneralData, that: GeneralData, rollingDaysN: int = 2) -> GeneralData:
    assert rollingDaysN >= 0
    assert isinstance(this.generalData, np.ndarray) == True
    assert isinstance(that.generalData, np.ndarray) == True
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)

    toStride2DArrayThis = this.generalData
    toStride2DArrayThat = that.generalData

    # column = toStride2DArrayThis.shape[1]
    stridedThis = utils.get_strided(toStride2DArrayThis, rollingDaysN)
    stridedThat = utils.get_strided(toStride2DArrayThat, rollingDaysN)
    temparray = np.zeros((stridedThis.shape[0],stridedThis.shape[2]))
    for i in range(stridedThis.shape[0]):
        tempThis = stridedThis[i].T
        tempThat = stridedThat[i].T
        # corr = np.corrcoef(tempThis, tempThat)
        # for j in range(stridedThis.shape[2]):
        corr = rowwise_cov(tempThis,tempThat)
        temparray[i] = corr
        # print(corr)
        # print(i)
    print('done')

    outputToReturn.generalData = temparray
            # print(stats.pearsonr(stridedThis[i,j,:], stridedThis[i,j,:]))
            # # print(corr[i,stridedThis.shape[2]+i])
            # outputToReturn.generalData[i,j] = corr[i,stridedThis.shape[2]+i]
            # outputToReturn.generalData[i, j] = 1
    #print('over')
    return outputToReturn