import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 加载CSV文件
# file_path = 'data/cleanedData/part-00060-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
# file_path = 'data/mergedData/total_sample.csv'
file_path = '../data/mergedData/total_10_SMOTE.csv'
df = pd.read_csv(file_path)

# 分离特征和标签
# 加载选定的特征列表
with open('model_info.pkl', 'rb') as file:
    model_info = pickle.load(file)
features = model_info['selected_features']

X = df[features]
y = df.iloc[:, -1]  # 最后一列

# 打印出各个类别的样本数量
print(y.value_counts())

# 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, train_size=0.08, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练分类器
rf_classifier.fit(X_train, y_train)

# 导出训练好的模型
model_file = '../models/train3_.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(rf_classifier, file)

# 进行预测
y_pred = rf_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
