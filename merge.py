import os
import pandas as pd

# 设置CSV文件所在的目录路径
# data_dir = 'data/sampleData'
data_dir = '../data/cleanedData'

# 创建一个空的DataFrame
combined_df = pd.DataFrame()

file_begin = 0
file_end = 2

# 遍历目录中的所有CSV文件
for i, filename in enumerate(os.listdir(data_dir)):
    if i < file_begin:
        continue
    if i >= file_end:
        break
    if filename.endswith('.csv'):
        # 拼接文件路径
        file_path = os.path.join(data_dir, filename)

        # 读取CSV文件并追加到合并的DataFrame中
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 查看合并后的DataFrame
print(combined_df.head())
print(f"合并后的DataFrame有 {len(combined_df)} 行数据")

# 导出合并后的DataFrame到CSV文件
output_file = '../data/mergedData/total_2.csv'
combined_df.to_csv(output_file, index=False)
print(f"合并后的数据已导出到 {output_file}")