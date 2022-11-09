import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

train=pd.read_csv("train3.csv") 
# test=pd.read_csv("train2.csv")
# print(train.head())
# print(test.head())

# train_original=train.copy()
# test_original=train.copy()
# print(train.columns)
# print(train.dtypes)
# print(train.shape)
# train.drop(['ID'], axis=1, inplace=True)
train['Datetime'] = pd.to_datetime(train['Datetime']).dt.date
# train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
train.set_index('Datetime',inplace=True)
# train.index = pd.DatetimeIndex(train.index).to_period('M')

train = train[~train.index.duplicated()]
# train = train.sort_index()
# train = train.resample('M').mean()
# print(train.tail())


test_result=adfuller(train['Count'])

def adfuller_test(counts):
    result=adfuller(counts)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
# print(adfuller_test(train['Count']))


# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(train['Count'],lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(train['Count'],lags=40,ax=ax2)

from statsmodels.tsa.arima.model import ARIMA

# model=ARIMA(train['Count'],order=(1,1,1))
# model_fit=model.fit()

# train['forecast']=model_fit.predict(start=20,end=26,dynamic=True)
# train[['Count','forecast']].plot(figsize=(12,8))

model=sm.tsa.statespace.SARIMAX(train['Count'],order=(1, 1, 1), seasonal_order=(1,1,1,12))
results=model.fit()

# train['forecast']=results.predict(start=20,end=26,dynamic=True)
# train[['Count','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[train.index[-1]+ DateOffset(months=x)for x in range(0,5)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=train.columns)

print(future_datest_df.tail())

future_df=pd.concat([train,future_datest_df])

future_df['forecast'] = results.predict(start = 25, end = 30, dynamic= True)  
# future_df[['Count', 'forecast']].plot(figsize=(12, 8)) 
print(future_df)

# train['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
# train[['Count','forecast']].plot(figsize=(12,8))

# test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M')
# test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M')

# for i in (train, train_original):
#     i['year']=i.Datetime.dt.year 
#     i['month']=i.Datetime.dt.month 
#     i['day']=i.Datetime.dt.day    
#     i['Hour']=i.Datetime.dt.hour

# test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
# test.index = test.Timestamp 
# test = test.resample('D').mean() 

# train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
# train.index = train.Timestamp
# train = train.resample('D').mean()

# Train=train.loc['2012-08-25':'2014-06-24'] 
# valid=train.loc['2014-06-25':'2014-09-25']

# Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 
# valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 
# plt.xlabel("Datetime") 
# plt.ylabel("Passenger count") 
# plt.legend(loc='best') 
# plt.show()
# print(train)
# train.plot()

# dd= np.asarray(Train.Count) 
# y_hat = valid.copy() 
# y_hat['naive'] = dd[len(dd)-1]
# plt.figure(figsize=(12,8)) 
# plt.plot(Train.index, Train['Count'], label='Train') 
# plt.plot(valid.index,valid['Count'], label='Valid') 
# plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
# plt.legend(loc='best') 
# plt.title("Naive Forecast") 
plt.show()
