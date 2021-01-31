import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#from statsmodels.tsa.seasonal import seasonal_decompose
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import optimizers
#https://keras-zh.readthedocs.io/

def lstm_forecast(df):
    ar = df.values
    x = []
    y = []
    for i in range(48,1440):
        x.append(ar[i - 48:i])
        y.append(ar[i])
    x, y = np.array(x), np.array(y)
    bx=[]
    for i in range(0,x.shape[0]):
        bx.append(np.isnan(x[i]).any())
    bo = np.isnan(y) | np.array(bx)
                        
    x=x[~bo]
    y=y[~bo]
    x = np.reshape(x, (x.shape[0], 48, 1))
                                        
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
    lstm_model = Sequential()
    lstm_model.add(LSTM(24, input_dim=1, input_length=48,activation='relu',return_sequences=False))
    #lstm_model.add(Dropout(0.3))
    #lstm_model.add(LSTM(120))
    #lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))

    early_stop = EarlyStopping(monitor='val_loss', verbose=0, patience=3)
    history=lstm_model.fit(x=X_train, y=y_train,batch_size=48, epochs=50, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred_test_lstm)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred_test_lstm))                                                                                                                                                                                                                         
    return rmse, mae

col=['date','ifnull(cpu_rate, 0)','ins_id']
all_fvm=pd.read_csv('./azure_d1_avg.csv',names=col)

all_fvm['date']=pd.to_datetime(all_fvm['date'])
vm_full = all_fvm.groupby([all_fvm['ins_id'],all_fvm['date']])[['ifnull(cpu_rate, 0)']].mean()

n=0
for i in vm_full.index.levels[0][14820:]:
    df = vm_full.loc[i]
    
    #df_avg = df['ifnull(cpu_rate, 0)']

    x2,y2 = lstm_forecast(df['ifnull(cpu_rate, 0)'])
    new = pd.DataFrame({"vm":i,"mae_avg":y2,"rmse_avg":x2},index=["0"])
    new.to_csv('./azure/lstm_azure_avg.csv',mode='a',header=0)
    n=n+1
    if(n%100==0):
        print('%d VMs have been processed'%(n))

