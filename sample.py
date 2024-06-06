import pandas as pd
import os


def sample(file_name):
    # 读取CSV文件
    csv_file = f'../data/rawData/{file_name}'
    df = pd.read_csv(csv_file)

    # 获取标签列的名称
    label_column = 'label'

    # 统计标签类别数量
    label_counts = df[label_column].value_counts()

    # 找出数量最少的类别及其数量
    min_count = label_counts.min()
    min_label = label_counts.idxmin()

    # 打印结果
    # print(f"The CSV file has {len(label_counts)} different classes.")
    # print(f"The class with the fewest instances is '{min_label}' with {min_count} instances.")

    # 创建一个新的DataFrame,保存每个类别的min_count个随机选取的实例
    new_df = pd.DataFrame()
    for label, count in label_counts.items():
        label_df = df[df[label_column] == label].sample(n=min_count, random_state=42)
        new_df = pd.concat([new_df, label_df], ignore_index=True)

    # 保存新的DataFrame到CSV文件
    new_csv_file = f'../data/sampleData/{file_name}'
    new_df.to_csv(new_csv_file, index=False)

    # print(f"The balanced dataset has been saved to '{new_csv_file}'.")
    print(f"{file_name}: {min_count}")


if __name__ == "__main__":
    # file_name = 'part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'
    # sample(file_name)

    directory = '../data/cleanedData'

    # 获取目录下的所有文件名
    file_names = os.listdir(directory)

    # 打印文件名
    for file_name in file_names:
        if file_name.endswith('.csv'):
            # print('../data/cleanedData/' + file_name)
            sample(file_name)