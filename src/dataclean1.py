#数据读取 + 整体情况分析
import pandas as pd
data1 = pd.read_excel('data/data.xlsx')
data2 = pd.read_excel('data/eval.xlsx')
print(data1.shape)
print(data1.describe())
print(data2.shape)
print(data2.describe())