import pandas as pd

# 1. 读取parquet文件
# 替换为你的parquet文件路径
file_path = "data2.parquet"
df = pd.read_parquet(file_path)
print(df.head())  # 输出前几行数据，检查列名和内容

# 2. 检查"diagnose"列的NaN值，获取包含NaN的行索引（行号）
# 先验证列是否存在，避免报错
if "diagnose" not in df.columns:
    raise ValueError(f"数据中不存在'diagnose'列，当前列名列表：{df.columns.tolist()}")

# 获取diagnose列为NaN的行索引（行号，默认从0开始计数）
nan_row_indices = df[df["diagnose"].isna()].index.tolist()

# 3. 输出包含NaN的行号
if nan_row_indices:
    print(f"diagnose列包含NaN值的行号（从0开始计数）：{nan_row_indices}")
    # 若需要输出从1开始的行号（符合日常阅读习惯）
    nan_row_numbers_1based = [idx + 1 for idx in nan_row_indices]
    print(f"diagnose列包含NaN值的行号（从1开始计数）：{nan_row_numbers_1based}")
else:
    print("diagnose列中没有发现NaN值，无需删除任何行")

# 4. 去除包含NaN的行
# dropna(subset=["diagnose"]) 仅删除diagnose列的NaN行，不影响其他列
df_cleaned = df.dropna(subset=["diagnose"])

# 5. 重新保存为parquet文件
# 替换为你的输出文件路径
output_file_path = "data2.parquet"
df_cleaned.to_parquet(output_file_path, index=False)  # index=False 不保存行索引列
print(f"清理后的数据已保存至：{output_file_path}")
print(f"原始数据行数：{len(df)}，清理后数据行数：{len(df_cleaned)}，删除行数：{len(df) - len(df_cleaned)}")
