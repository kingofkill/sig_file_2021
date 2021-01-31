import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

all_fvm = pd.read_csv('./azure/vm_all_d1.csv')
all_fvm.rename(columns={'vm_id':'vmid','min_cpu':'mincpu','max_cpu':'maxcpu','avg_cpu':'avgcpu'},inplace=True)
vm_full = all_fvm.groupby([all_fvm['vmid'],all_fvm['timestamp']])[['mincpu','maxcpu','avgcpu']].mean()
n=0
for i in vm_full.index.levels[0]:
    df = vm_full.loc[i]

    fd = pd.to_datetime("1th of May, 2020")
    date_list = pd.DatetimeIndex([fd + timedelta(minutes=5*x) for x in range(0, len(df))])
    df.set_index(date_list,inplace=True)

    df_avg = df.resample('30min').mean()['avgcpu'][:864]
    df_min = df.resample('30min').min()['mincpu'][:864]
    df_max = df.resample('30min').max()['maxcpu'][:864]

    max_triple = ExponentialSmoothing(df_max,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues
    avg_triple = ExponentialSmoothing(df_avg,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues
    min_triple = ExponentialSmoothing(df_min,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues

    x1=np.sqrt(mean_squared_error(df_max,max_triple))
    y1=mean_absolute_error(df_max, max_triple)

    x2=np.sqrt(mean_squared_error(df_avg,avg_triple))
    y2=mean_absolute_error(df_avg, avg_triple)

    x3=np.sqrt(mean_squared_error(df_min,min_triple))
    y3=mean_absolute_error(df_min, min_triple)
    
    new = pd.DataFrame({"vm":i,"mae_max":y1,"rmse_max":x1,"mae_avg":y2,"rmse_avg":x2,"mae_min":y3,"rmse_min":x3},index=["0"])
    new.to_csv('./azure/ewma/18forecast_vm_d1.csv',mode='a',header=0)
    n=n+1
    if(n%100==0):
        print('%d VMs have been processed'%(n))


