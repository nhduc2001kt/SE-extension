# 1. Các thư viện sử dụng
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import json
import math
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# 2. Đọc dữ liệu
# f = open("/content/drive/MyDrive/SE extend/photo_v3_cpu_load.json")
f = open("photo_v3_cpu_load.json", "r")
data = (json.load(f)['values'])
data = np.array(data).transpose()

data_df = pd.DataFrame(data.transpose())
data_df[0] = data_df[0].astype(int)
data_df[1] = data_df[1].astype(float)
data_df.head()

df = pd.DataFrame(data_df)
df[0] = pd.to_datetime(df[0], unit='s')
df = df.set_index(0)
df = df.asfreq('20T')
df.head()

# 3. Huấn luyện mô hình
result = seasonal_decompose(df[1], model='additive', period = 6*24)

plt.rc('figure',figsize=(12,8))
plt.rc('font',size=15)
result.plot()
plt.show()

# 4. Xử lý kết quả đầu ra
threshold = 0.1
anomalies = [1 if (not math.isnan(x)) and (abs(x) > threshold)  else 0 for x in result.resid.values]
data_df["anomaly"] = anomalies
data_df["residual"] = result.resid.values
data_df[:][400:405]

anomaly_df = pd.DataFrame(data_df)
anomaly_df = anomaly_df.loc[anomaly_df['anomaly'] == 1]
anomaly_df.head()

# 5. Mô phỏng kết quả
fig = px.line(data_df, x=0, y=1, labels={"0": "time", "1": "cpu_load"}, title='Unsupervised anomaly detection in CPU utilization')
fig.add_trace(go.Scatter(x=anomaly_df[0].to_list(), y=anomaly_df[1].to_list(), mode='markers', name='anomalies'))
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# 6. Thay đổi giá trị ngưỡng (Threshold)
def getNomaly(date: str, df: pd.DataFrame):
    return float(df.loc[df[0] == date, 'residual'])

threshold = getNomaly('2021-10-18 08:20:00', data_df)
threshold

anomalies = [1 if (not math.isnan(x)) and (abs(x) > threshold)  else 0 for x in result.resid.values]
data_df["anomaly"] = anomalies
data_df["residual"] = result.resid.values
anomaly_df = pd.DataFrame(data_df)
anomaly_df = anomaly_df.loc[anomaly_df['anomaly'] == 1]
fig = px.line(data_df, x=0, y=1, labels={"0": "time", "1": "cpu_load"}, title='Unsupervised anomaly detection in CPU utilization')
fig.add_trace(go.Scatter(x=anomaly_df[0].to_list(), y=anomaly_df[1].to_list(), mode='markers', name='anomalies'))
fig.update_xaxes(rangeslider_visible=True)
fig.show()
