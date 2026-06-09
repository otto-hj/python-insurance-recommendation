# 检测合并之后的缺失值情况
import pandas as pd
# 1. 读取训练集和测试集
train = pd.read_excel("data/data.xlsx")
test = pd.read_excel("data/eval.xlsx")
print("===== 原始数据规模 =====")
print("训练集：", train.shape)
print("测试集：", test.shape)

#2.给训练集和测试集添加来源
#在这两个 内存中的表格对象 里新增一列，列名叫source，训练集的值为train，测试集的值为test
#5822 行，86列变成87列
train["source"]="train"
test["source"]="test"

#合并训练集和测试集
#把多个 DataFrame 上下粘在一起（像 Excel 把两个表粘贴成一个),并且为了防止索引乱掉，设置ignore_index=True
# 训练集原来行号是 0~5821，测试集原来也是 0~3999，合并后重新编号成 0~9821。
data=pd.concat([train,test],ignore_index=True)

# 4.缺失值检查
# 这行是在算每一列的缺失值数量。
nan_count = data.isnull().sum().sort_values(ascending=False)
# 计算缺失比例
nan_ratio = nan_count / len(data)
nan_data = pd.concat([nan_count, nan_ratio], axis=1, keys=["count", "ratio"])

print("\n===== 缺失值最多的前10列 =====")
print(nan_data.head(10))