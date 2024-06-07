import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# 加载模型
model_file = '../models/train3.pkl'
with open(model_file, 'rb') as file:
    rf_classifier = pickle.load(file)

# 加载数据
file_path = '../data/mergedData/total_10_20.csv'
df = pd.read_csv(file_path)

# 分离特征和标签
with open('model_info.pkl', 'rb') as file:
    model_info = pickle.load(file)
features = model_info['selected_features']
X = df[features]
# X = df.iloc[:, :-1]  # 前n-1列
y = df.iloc[:, -1]  # 最后一列

# 拆分数据为训练集和测试集u
_, X_test, _, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 进行预测
y_pred = rf_classifier.predict(X_test)
# 计算预测概率
y_proba = rf_classifier.predict_proba(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# 初始化计数器
low_certainty_counts = {}
high_certainty_incorrect_counts = {}
total_label_counts = {}

# 统计各个标签的数量
for label in y_test:
    if label not in total_label_counts:
        total_label_counts[label] = 1
    else:
        total_label_counts[label] += 1

# 统计确定性小于0.9的实际label
for i, (pred, proba, actual) in enumerate(zip(y_pred, y_proba, y_test)):
    certainty = max(proba)  # 最高的概率表示预测结果的确定性
    if certainty < 0.9:
        if actual not in low_certainty_counts:
            low_certainty_counts[actual] = 1
        else:
            low_certainty_counts[actual] += 1
    elif pred != actual:
        # 确定性高但预测错误
        if actual not in high_certainty_incorrect_counts:
            high_certainty_incorrect_counts[actual] = 1
        else:
            high_certainty_incorrect_counts[actual] += 1

# 输出确定性小于0.9的实际标签统计结果
print("确定性小于0.9的实际标签统计结果：")
for label, count in low_certainty_counts.items():
    total_count = total_label_counts[label]
    print(f"标签 {label}: {count} 次, 总共 {total_count} 次")

# 输出确定性高但预测错误的实际标签统计结果
print("确定性高但预测错误的实际标签统计结果：")
for label, count in high_certainty_incorrect_counts.items():
    total_count = total_label_counts[label]
    print(f"标签 {label}: {count} 次, 总共 {total_count} 次")
