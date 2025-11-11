import random
import pandas as pd
from datetime import datetime, timedelta

# 设置时间范围
start_range = datetime(2025, 1, 1, 8, 0)
end_range = datetime(2025, 10, 22, 22, 0)

# 计算总天数和时间间隔
total_days = (end_range - start_range).days
interval_days = total_days / 230  # 平均间隔天数

records = []

current_time = start_range

for i in range(230):
    # 在平均间隔基础上添加一点随机偏移（±0.3天）
    day_offset = i * interval_days + random.uniform(-0.3, 0.3)
    start_date = start_range + timedelta(days=day_offset)

    # 随机确定当天的开始时间（8:00~22:00，整点或半点）
    hour = random.randint(8, 21)
    minute = random.choice([0, 30])
    start_time = start_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 使用时间（0.5~20小时，整数或半小时）
    use_time = random.choice([x * 0.5 for x in range(1, 41)])
    end_time = start_time + timedelta(hours=use_time)

    # 若结束时间超过当日22:30，则顺延到下一天8:00
    if end_time.hour > 22 or (end_time.hour == 22 and end_time.minute > 30):
        next_day = start_time.date() + timedelta(days=1)
        start_time = datetime.combine(next_day, datetime.min.time()).replace(hour=8)
        use_time = random.choice([x * 0.5 for x in range(1, 41)])
        end_time = start_time + timedelta(hours=use_time)

    # 限制时间在范围内
    if end_time > end_range:
        end_time = end_range

    # 确保按时间先后排序
    records.append([start_time, end_time, use_time])

# 按开始时间排序
records.sort(key=lambda x: x[0])

# 转为 DataFrame
df = pd.DataFrame(records, columns=["开始时间", "结束时间", "使用时间"])

# 格式化输出
df["开始时间"] = df["开始时间"].dt.strftime("%Y/%m/%d %H:%M")
df["结束时间"] = df["结束时间"].dt.strftime("%Y/%m/%d %H:%M")

# 保存为 CSV 文件
df.to_csv("时间段数据_均匀分布.csv", index=False, encoding="utf-8-sig")

print("✅ 已生成 230 行时间段数据（均匀分布在 2025/1/1 ~ 2025/10/22 之间）")
print(df.head())

