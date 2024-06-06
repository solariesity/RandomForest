import pandas as pd
import os

index_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM',
              'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU']


def clean_data_type(raw_file_name):
    raw_file_path = f"../data/rawData/{raw_file_name}"
    df = pd.read_csv(raw_file_path)

    # 用于存储需要删除的行索引
    rows_to_drop = []

    for index in range(len(df.iloc[0]) - 1):
        # print(row)
        # 将第一列数据存入列表
        column = df.iloc[:, index].tolist()

        # 判断 column 中的数据是否全部为 float64 类型
        is_all_float = True
        non_float_index = None
        non_float_value = None
        for i, value in enumerate(column):
            try:
                float(value)
            except ValueError:
                is_all_float = False
                non_float_index = i
                non_float_value = value
                break

        if not is_all_float:
            print(
                f"'Column {index_list[index]}' contains a non-float64 value: '{non_float_value}' at index {non_float_index + 2}")
            rows_to_drop.append(non_float_index)

    # 检查最后一列是否全都是字符串类型
    last_column = df.iloc[:, -1].tolist()
    is_all_string = True
    for i, value in enumerate(last_column):
        if not isinstance(value, str):
            is_all_string = False
            print(f"The last column contains a non-string value: '{value}' at row {i + 1}")
            break

    if is_all_string:
        print("The last column contains only string values.")

    # 检查是否有空值,如果有,删除当前行
    rows_with_nan = df[df.isnull().any(axis=1)].index.tolist()
    rows_to_drop.extend(rows_with_nan)
    print(f"Rows with NaN values: {[row + 2 for row in rows_with_nan]}")

    # 删除包含非 float64 或非字符串类型的行
    df = df.drop(df.index[list(set(rows_to_drop))])

    # 将预处理后的数据保存到新的文件
    df.to_csv(f'../data/cleanedData/{raw_file_name}', index=False)


if __name__ == "__main__":
    # file_name = "part-00060-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
    # file_name = "1.csv"
    # file_name = "part-00151-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
    directory = '../data/rawData'

    # 获取目录下的所有文件名
    file_names = os.listdir(directory)

    # 打印文件名
    for file_name in file_names:
        if file_name.endswith('.csv'):
            print('../data/rawData/' + file_name)
            clean_data_type(file_name)
