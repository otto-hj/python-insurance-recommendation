import pandas as pd
from sklearn.utils import resample, shuffle

# 平衡训练数据集
# 目标：缓解“未购买移动房车险”样本远多于“购买移动房车险”样本的问题。

train = pd.read_csv("data/train_preprocess.csv", encoding="utf_8_sig")

target = "移动房车险数量"

print("===== 原始训练集类别分布 =====")
print(train[target].value_counts())

# 少数类：购买移动房车险
train_up = train[train[target] == 1]

# 多数类：未购买移动房车险
train_down = train[train[target] == 0]

# 对少数类上采样，对多数类下采样
train_up = resample(train_up, n_samples=696, random_state=0)
train_down = resample(train_down, n_samples=1095, random_state=0)

# 合并并打乱顺序
train_balance = shuffle(pd.concat([train_up, train_down]), random_state=0)

print("\n===== 平衡后训练集类别分布 =====")
print(train_balance[target].value_counts())

train_balance.to_csv("data/train_balance.csv", index=False, encoding="utf_8_sig")

print("\n已保存：data/train_balance.csv")
