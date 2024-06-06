import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 加载CSV文件
file_path = '../data/mergedData/total_10_SMOTE.csv'
df = pd.read_csv(file_path)

# 分离特征和标签
features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
            'Srate', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
            'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Tot size', 'IAT',
            'Magnitue', 'Covariance', 'Variance']
X = df[features]
y = df.iloc[:, -1]  # 最后一列

# 打印出各个类别的样本数量
print(y.value_counts())

# 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, train_size=0.08, random_state=42)

# 定义要搜索的超参数范围
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 初始化随机森林分类器和网格搜索对象
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数和模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 打印最佳超参数
print("Best Parameters:")
print(best_params)

# 进行预测
y_pred = best_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# 保存最佳模型
model_file = '../models/train4.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(best_model, file)