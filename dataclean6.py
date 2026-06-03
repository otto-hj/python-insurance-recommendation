import numpy as np
import pandas as pd

# 数据预处理（5）：特征选择
# 计算每个字段与目标变量的相关系数，保留相关系数绝对值 >= 0.01 的字段。

train = pd.read_excel("data/data.xlsx")
test = pd.read_excel("data/eval.xlsx")

target = "移动房车险数量"

corr_target = train.corr()[target]
important_feature = corr_target[np.abs(corr_target) >= 0.01].index.tolist()

print("筛选后的字段数量：", len(important_feature))
print("筛选后的字段：")
print(important_feature)

train_preprocess = train[important_feature]
test_preprocess = test[important_feature]

train_preprocess.to_csv("data/train_preprocess.csv", index=False, encoding="utf_8_sig")
test_preprocess.to_csv("data/test_preprocess.csv", index=False, encoding="utf_8_sig")

print("已保存：data/train_preprocess.csv")
print("已保存：data/test_preprocess.csv")
