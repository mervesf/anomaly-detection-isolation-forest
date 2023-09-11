import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('taxi_rides.csv')
df1 = pd.read_csv('taxi_rides_clean.csv')
df.head(100)

plt.scatter(df['timestamp'],df['value'])
plt.xlabel('Time')
plt.ylabel('value')
plt.show

model=IsolationForest(n_estimators=80, max_samples='auto',contamination=float(0.004), max_features=1.0, bootstrap=True, n_jobs=-1, random_state=42, verbose=0)
model.fit(df[['value']])

df['scores']=model.decision_function(df[['value']])
df['anomaly']=model.predict(df[['value']])
print(df['anomaly'].unique())

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(len(anomaly))
print(anomaly)

window_size=1000
df3=pd.DataFrame()
num_rows=len(df['value'])

for i in range(0 ,num_rows,window_size):
    window=df.iloc[i:i+window_size]
    window_mean=window['value'].mean()
    window.loc[window['anomaly']==-1,'value']=window_mean
    df3=pd.concat([df3,window], ignore_index=True)

common_column='value'
predict_column=df3[common_column]
real_column=df1[common_column]
mae=np.mean(np.abs(real_column-predict_column))
print(mae)
