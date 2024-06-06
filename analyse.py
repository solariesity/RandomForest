# 2024/6/1 11:10
import pandas as pd


def analyse(file_path):
    # 加载CSV文件
    df = pd.read_csv(file_path)

    # 分离标签列
    y = df.iloc[:, -1]  # 假设最后一列是标签

    # 统计标签类别数量
    label_counts = y.value_counts()
    num_classes = label_counts.shape[0]

    print(f"总共有 {num_classes} 类标签")
    print("每个标签的数量如下：")
    print(label_counts)


if __name__ == "__main__":
    # file_path = 'data/reprocessedData/part-00060-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
    # file_path = 'data/reprocessedData/1.csv'

    file_path = '../data/cleanedData/part-00060-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
    # file_path = 'data/mergedData/total_sample.csv'
    analyse(file_path)
