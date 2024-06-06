import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# 加载模型
model_file = '../models/train.pkl'
with open(model_file, 'rb') as file:
    rf_classifier = pickle.load(file)

# file_path = 'data/mergedData/total_10.csv'
# file_path = 'data/mergedData/total_10_20.csv'
# file_path = 'data/cleanedData/part-00059-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
file_path = '../data/mergedData/total_sample.csv'
df = pd.read_csv(file_path)

# # 分离特征和标签
# # features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
# #             'Srate', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
# #             'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Tot size', 'IAT',
# #             'Magnitue', 'Covariance', 'Variance']
# # X = df[features]
# X = df.iloc[:, :-1]  # 前n-1列
# y = df.iloc[:, -1]  # 最后一列
#
# # 拆分数据为训练集和测试集
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#
#
# # 进行预测
# y_pred = rf_classifier.predict(X_test)
#
# # 评估模型
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)
#
# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(classification_rep)


# 获取特征名称
feature_names = df.columns[:-1]  # 排除最后一列(可能是目标变量)

# 进行预测（新的测试数据X_new）
X_new = df.iloc[:, :-1].values
# X_new = X_new.reshape(1, -1)
# X_new = np.array(X_new)
X_new = pd.DataFrame(X_new, columns=feature_names)

# print(X_new)
y_pred_new = rf_classifier.predict(X_new)
# print(y_pred_new)

# 计算预测概率
y_proba_new = rf_classifier.predict_proba(X_new)

# 输出预测结果和确定性
for i, (pred, proba) in enumerate(zip(y_pred_new, y_proba_new)):
    certainty = max(proba)  # 最高的概率表示预测结果的确定性
    print(f"样本 {i+1}: 预测结果 = {pred}, 确定性 = {certainty:.2f}")