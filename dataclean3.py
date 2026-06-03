# 统计每个特征有多少种不同取值。
import pandas as pd
# 1. 读取训练集和测试集
train = pd.read_excel("data/data.xlsx")
test = pd.read_excel("data/eval.xlsx")
# 2. 添加来源标记并合并
train["source"] = "train"
test["source"] = "test"
data = pd.concat([train, test], ignore_index=True, sort=False)
# 3. 统计每个字段的不同取值数量
unique_count = data.apply(lambda x: len(x.unique())).sort_values(ascending=False)

print("===== 每个特征的类别数量（前10个）=====")
print(unique_count.head(10))