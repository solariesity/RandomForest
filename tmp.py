import pandas as pd
from sklearn.ensemble import IsolationForest

# 读取CSV文件
df = pd.read_csv('../data/rawData/1.csv')

# 处理异常值
# 使用Z-score方法检测异常值
z_scores = df.iloc[:, :-1].select_dtypes(include='number').apply(lambda x: (x - x.mean()) / x.std(), axis=0)
anomaly_mask = (z_scores.abs() > 3).any(axis=1)
anomalies = df.loc[anomaly_mask]
print("异常值:")
for idx, row in anomalies.iterrows():
    print(f"行 {idx}:")
    for col, value in row.items():
        if abs(z_scores.loc[idx, col]) > 3:
            print(f"列 {col}: 值 {value} 超出 3 个标准差")

# 处理重复数据
df_cleaned = df.drop_duplicates()

# 处理离群值
# 使用Isolation Forest算法检测离群值
clf = IsolationForest(contamination=0.01)
anomaly_scores = clf.fit_predict(df_cleaned.iloc[:, :-1])
outliers = df_cleaned.loc[anomaly_scores == -1]
print("\n离群值:")
for idx, row in outliers.iterrows():
    print(f"行 {idx}: {row.to_string()}")