import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据集
file_path = '../data/cleanedData/part-00059-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
df = pd.read_csv(file_path)

# 分离特征和标签
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化模型并设置max_iter和solver
model = LogisticRegression(max_iter=1000, solver='saga')

# 应用RFE
# 选择前n个特征，假设我们想选择前n个重要特征
n_features_to_select = 45
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
rfe.fit(X_scaled, y)

# 结果分析
selected_features = X.columns[rfe.support_]
print("Selected features: ", selected_features)

# 查看训练后的模型
final_model = rfe.estimator_
print("Final model: ", final_model)

# 查看输入维度
print("Input dimensions after RFE: ", n_features_to_select)

# 获取特征排名
ranking = rfe.ranking_

# 获取特征重要性
feature_importance = pd.Series(ranking, index=X.columns)
feature_importance_sorted = feature_importance.sort_values(ascending=False)  # 按降序排序

# 打印特征重要性
print("Feature importance (ranked from high to low):")
print(feature_importance_sorted)

# 绘制特征重要性图表
plt.figure(figsize=(10, 8))
feature_importance_sorted.plot(kind='barh')
plt.title('Feature Importance Ranking')
plt.xlabel('Ranking')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # 排名从1开始，因此应将y轴倒置
plt.show()
