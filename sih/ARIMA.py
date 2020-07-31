# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams 
rcParams['figure.figsize']=10 ,6

d=os.path.dirname(os.getcwd())
d=os.path.join(d,"sih")
d=os.path.join(d,"States")
d=os.path.join(d,"gujarat")
d=os.path.join(d,"Amreli")
os.chdir(d)
dataset = pd.read_csv('real.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
indexedDataset = dataset.set_index(['Date'])
indexedDataset = indexedDataset.fillna(method='ffill')


from datetime import datetime
#indexedDataset.tail(12)



rolmean = indexedDataset.rolling(window=12).mean()

rolstd = indexedDataset.rolling(window=12).std()
#print(rolmean,rolstd)



from statsmodels.tsa.stattools import adfuller

#print('Results of DFT: ')
dftest = adfuller(indexedDataset['Prices'],autolag='AIC')

dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-val','lag used','Number of obser'])
 

indexedDataset_logScale=np.log(indexedDataset)

movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingstd=indexedDataset_logScale.rolling(window=12).std()


datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
#datasetLogScaleMinusMovingAverage.head(12)

datasetLogScaleMinusMovingAverage.dropna(inplace=True)
#datasetLogScaleMinusMovingAverage.head(12)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    
    dftest=adfuller(timeseries['Prices'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test stats','pval','lag','No of obser'])
  

test_stationarity(datasetLogScaleMinusMovingAverage)

exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()


datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

datasetLogDiffShifting=indexedDataset_logScale - indexedDataset_logScale.shift()


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)





from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(indexedDataset_logScale,order=(1,1,1))
results_AR=model.fit(disp=-1)


predictions_ARIMA_diff=pd.Series(results_AR.fittedvalues,copy=True)
#print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
#print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log=pd.Series(indexedDataset_logScale['Prices'].iloc[0],index=indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#predictions_ARIMA_log.head()

predictions_ARIMA=np.exp(predictions_ARIMA_log)

#indexedDataset_logScale
#predictions_ARIMA

modell=ARIMA(predictions_ARIMA,order=(1,1,1))
results_ARM=modell.fit(disp=-1)

#results_ARM.plot_predict(1,60)
x=results_ARM.forecast(steps=12)



toplot=x[0][0:12]
#toplot
print(toplot)