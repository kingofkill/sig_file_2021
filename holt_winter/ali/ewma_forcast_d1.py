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

cpu=pd.read_csv('./ali1/ali_vm_all_d1.csv')
vm_full = cpu.groupby([cpu['ins_id'],cpu['report_ts']])[['ifnull(cpu_rate, 0)']].mean()
n=0
for i in vm_full.index.levels[0]:
    df = vm_full.loc[i]
    df['date'] = pd.to_datetime(df.index,unit='s')
    df['date'] = pd.DatetimeIndex(df['date']) + timedelta(hours=8)
    df.set_index(df['date'],inplace=True)
    df.drop(['date'],axis=1,inplace=True)

    df_avg = df.resample('30min').mean()["ifnull(cpu_rate, 0)"][:864]*100
    df_min = df.resample('30min').min()["ifnull(cpu_rate, 0)"][:864]*100
    df_max = df.resample('30min').max()["ifnull(cpu_rate, 0)"][:864]*100

    mean_avg = np.mean(df_avg)
    mean_min = np.mean(df_min)
    mean_max = np.mean(df_max)

    df_avg.fillna(mean_avg, inplace=True)
    df_min.fillna(mean_min, inplace=True)
    df_max.fillna(mean_max, inplace=True)

    if len(df_avg)<864:
        continue

    max_triple = ExponentialSmoothing(df_max,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues
    avg_triple = ExponentialSmoothing(df_avg,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues
    min_triple = ExponentialSmoothing(df_min,trend='add',seasonal='add',seasonal_periods=24).fit().fittedvalues

    x1=np.sqrt(mean_squared_error(df_max,max_triple))
    y1=mean_absolute_error(df_max,max_triple)

    x2=np.sqrt(mean_squared_error(df_avg,avg_triple))
    y2=mean_absolute_error(df_avg,avg_triple)

    x3=np.sqrt(mean_squared_error(df_min,min_triple))
    y3=mean_absolute_error(df_min,min_triple)

    new = pd.DataFrame({"vm":i,"mae_max":y1,"rmse_max":x1,"mae_avg":y2,"rmse_avg":x2,"mae_min":y3,"rmse_min":x3},index=["0"])
    new.to_csv('./ali1/ewma_forecast_vm_d1.csv',mode='a',header=0)
    n=n+1
    if(n%100==0):
        print('%d VMs have been processed'%(n))


