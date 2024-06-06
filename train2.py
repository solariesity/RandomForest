import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

print(1)
# 加载CSV文件
# file_path = 'data/cleanedData/part-00060-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
# file_path = 'data/mergedData/total_sample.csv'
file_path = '../data/mergedData/total_150_SMOTE.csv'
df = pd.read_csv(file_path)


# 分离特征和标签
X = df.iloc[:, :-1]  # 前n-1列
y = df.iloc[:, -1]  # 最后一列

# 打印出各个类别的样本数量
print(y.value_counts())

# # 过采样 (SMOTE)
# print("过采样")
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # 打印出各个类别的样本数量
# print(y_resampled.value_counts())

# # 欠采样 (TomekLinks)
# print("欠采样")
# tl = TomekLinks()
# X_resampled, y_resampled = tl.fit_resample(X_resampled, y_resampled)
#
# # 打印出各个类别的样本数量
# print(y_resampled.value_counts())

# 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.08, test_size=0.02, random_state=42)

# 打印出各个类别的样本数量
print(y_train.value_counts())

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练分类器
rf_classifier.fit(X_train, y_train)

# # 导出训练好的模型
# model_file = '../models/train2_.pkl'
# with open(model_file, 'wb') as file:
#     pickle.dump(rf_classifier, file)

# 进行预测
y_pred = rf_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
