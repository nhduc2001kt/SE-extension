# 1. Các thư viện sử dụng
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.io as pio
pio.renderers.default = "browser"

# 2. Đọc dữ liệu
f = open("photo_v3_cpu_load.json")
a = (json.load(f)['values'])
a = np.array(a).transpose()

df = pd.DataFrame(a.transpose())
df[0] = df[0].astype(int)
df[1] = df[1].astype(float)
df[0] = pd.to_datetime(df[0], unit='s')
df.head()

fig_0 = px.line(df, x=0, y=1, labels={"0": "time", "1": "cpu_load"}, title='Unsupervised anomaly detection in CPU utilization')
# fig_0.show()

# 3. Huấn luyện mô hình
iso_forest = IsolationForest(n_estimators = 100, max_samples = "auto", contamination = 0.02, random_state = 42)
iso_forest.fit(np.array([a[1]], dtype=float).transpose())
y_pred = iso_forest.predict(np.array([a[1]], dtype=float).transpose())
y_pred = [1 if x == -1 else 0 for x in y_pred]

# 4. Xử lý kết quả đầu ra
iso_anomaly_df = pd.DataFrame(a.transpose())
iso_anomaly_df[0] = iso_anomaly_df[0].astype(int)
iso_anomaly_df[1] = iso_anomaly_df[1].astype(float)
iso_anomaly_df[0] = pd.to_datetime(iso_anomaly_df[0], unit='s')
iso_anomaly_df["anomaly"] = y_pred
iso_anomaly_df.head()

iso_anomaly_df = iso_anomaly_df.loc[iso_anomaly_df['anomaly'] == 1]
print(iso_anomaly_df.head())

# 5. Mô phòng kết quả (Visualization)
fig = px.line(df, x=0, y=1, labels={"0": "time", "1": "cpu_load"}, title='Unsupervised anomaly detection in CPU utilization')
fig.add_trace(go.Scatter(x=iso_anomaly_df[0].to_list(), y=iso_anomaly_df[1].to_list(), mode='markers', name='anomalies'))
fig.update_xaxes(rangeslider_visible=True)
fig.show()