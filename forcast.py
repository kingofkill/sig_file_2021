import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)

#holt_winter Ali的结果
name=["vm","mae_max","rmse_max","mae_avg","rmse_avg","mae_min","rmse_min"]
ali1 = pd.read_csv('./holt_winter/ali/ewma_forecast_vm_d1.csv',names=name)
ali2 = pd.read_csv('./holt_winter/ali/ewma_forecast_vm_d2.csv',names=name)
ali3 = pd.read_csv('./holt_winter/ali/ewma_forecast_vm_d3.csv',names=name)
ali4 = pd.read_csv('./holt_winter/ali/ewma_forecast_vm_d4.csv',names=name)
ali_all = pd.concat([ali1,ali2,ali3,ali4])

#holt_winter Azure的结果
v1=pd.read_csv('./holt_winter/azure/18forecast_vm_d1.csv',names=name)
v2=pd.read_csv('./holt_winter/azure/18forecast_vm_d2.csv',names=name)
v3=pd.read_csv('./holt_winter/azure/18forecast_vm_d3.csv',names=name)
v4=pd.read_csv('./holt_winter/azure/18forecast_vm_d4.csv',names=name)
v5=pd.read_csv('./holt_winter/azure/18forecast_vm_d5.csv',names=name)
v6=pd.read_csv('./holt_winter/azure/18forecast_vm_d6.csv',names=name)
v7=pd.read_csv('./holt_winter/azure/18forecast_vm_d7.csv',names=name)
v8=pd.read_csv('./holt_winter/azure/18forecast_vm_d8.csv',names=name)
azure = pd.concat([v1,v2,v3,v4,v5,v6,v7,v8])

#LSTM Ali的结果
lstm_ali_max=pd.read_csv('./lstm_ali_max.csv',names=["vm","mae_max","rmse_max"])
lstm_ali_avg=pd.read_csv('./lstm_ali_avg.csv',names=["vm","mae_avg","rmse_avg"])

#LSTM Azure的结果（上面三个数据集都是完整的不需要替换，每次只需在当前文件夹下替换lstm Azure的结果）
lstm_azure_avg=pd.read_csv('./lstm_azure_avg.csv',names=["vm","mae_avg","rmse_avg"])
lstm_azure_max=pd.read_csv('./lstm_azure_max.csv',names=["vm","mae_max","rmse_max"])

#LSTM2 Azure的结果
lstm2_azure_max=pd.read_csv('./lstm2_azure_max.csv',names=["vm","mape_max","rmse_max"])
lstm2_azure_avg=pd.read_csv('./lstm2_azure_avg.csv',names=["vm","mape_avg","rmse_avg"])
lstm2_ali_max=pd.read_csv('./lstm2_ali_max.csv',names=["vm","mape_max","rmse_max"])
lstm2_ali_avg=pd.read_csv('./lstm2_ali_avg.csv',names=["vm","mape_avg","rmse_avg"])

#rmse的max_cpu图片
ali_rmse_max=ali_all[ali_all['rmse_max']<=100]
lstm_ali_rmse_max=lstm_ali_max[lstm_ali_max['rmse_max']<=100]
lstm_azure_rmse_max=lstm_azure_max[lstm_azure_max['rmse_max']<=100]

lstm2_azure_rmse_max=lstm2_azure_max[lstm2_azure_max['rmse_max']<=100]
lstm2_ali_rmse_max=lstm2_ali_max[lstm2_ali_max['rmse_max']<=100]
plt.figure()

cdf(azure['rmse_max'],label='Azure Holt-Winters',color='b',ls='-')
#cdf(lstm_azure_rmse_max['rmse_max'],label='Azure LSTM',color='b',ls='-.')
cdf(lstm2_azure_rmse_max['rmse_max'],label='Azure LSTM2',color='b',ls='-.')
cdf(ali_rmse_max['rmse_max'],label='Edge Holt-Winters',color='r',ls='-')
cdf(lstm2_ali_rmse_max['rmse_max'],label='Edge LSTM2',color='r',ls='-.')
#cdf(lstm_ali_rmse_max['rmse_max'],label='Edge LSTM',color='r',ls='-.')
plt.title('VMs max_cpu rmse distribution')
plt.legend(fontsize='large')
plt.xlabel('rmse_max')
plt.tick_params(labelsize = 13)
plt.savefig('./VMs max_cpu rmse distribution.png')

#rmse的avg_cpu图片
azure_rmse_avg=azure[azure['rmse_avg']<=100]
ali_rmse_avg=ali_all[ali_all['rmse_avg']<=100]
lstm_ali_rmse_avg=lstm_ali_avg[lstm_ali_avg['rmse_avg']<=100]
lstm_azure_rmse_avg=lstm_azure_avg[lstm_azure_avg['rmse_avg']<=100]

lstm2_azure_rmse_avg=lstm2_azure_avg[lstm2_azure_avg['rmse_avg']<=100]
lstm2_ali_rmse_avg=lstm2_ali_avg[lstm2_ali_avg['rmse_avg']<=100]
plt.figure()
cdf(azure_rmse_avg['rmse_avg'],label='Azure Holt-Winters',color='b',ls='-')
cdf(lstm2_azure_rmse_avg['rmse_avg'],label='Azure LSTM2',color='b',ls='-.')
#cdf(lstm_azure_rmse_avg['rmse_avg'],label='Azure LSTM',color='b',ls='-.')
cdf(ali_rmse_avg['rmse_avg'],label='Edge Holt-Winters',color='r',ls='-')
cdf(lstm2_ali_rmse_avg['rmse_avg'],label='Edge LSTM2',color='r',ls='-.')
#cdf(lstm_ali_rmse_avg['rmse_avg'],label='Edge LSTM',color='r',ls='-.')

plt.title('VMs avg_cpu rmse distribution')
plt.legend(fontsize='large')
plt.xlabel('rmse_avg')
plt.xlim(-1,8)
plt.tick_params(labelsize = 13)
plt.savefig('./VMs avg_cpu rmse distribution.png')