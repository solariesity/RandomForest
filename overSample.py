import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# 加载CSV文件
file_path = '../data/mergedData/total_150.csv'
df = pd.read_csv(file_path)

# 分离特征和标签
X = df.iloc[:, :-1]  # 前n-1列
y = df.iloc[:, -1]  # 最后一列

# 打印出各个类别的样本数量
print("原始数据集各个类别的样本数量:")
print(y.value_counts())

# 过采样 (SMOTE)
print("过采样")
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)

# 打印过采样后的类别数量
print("过采样后各个类别的样本数量:")
print(Counter(y_resampled_smote))

# 欠采样
print("欠采样")
under_sampler = RandomUnderSampler(random_state=42)
X_resampled_final, y_resampled_final = under_sampler.fit_resample(X_resampled_smote, y_resampled_smote)

# 合并为一个DataFrame
resampled_data_final = pd.DataFrame(X_resampled_final, columns=X.columns)
resampled_data_final['label'] = y_resampled_final

# 保存为CSV文件
resampled_data_final.to_csv('../data/mergedData/total_10_SMOTE_UnderSampling.csv', index=False)

# 打印最终采样后的类别数量
print("最终采样后各个类别的样本数量:")
print(Counter(y_resampled_final))
