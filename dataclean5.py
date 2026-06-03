# 用箱线图比较不同人群在某个特征上的差异。
# 买了移动房车险的人，和没买移动房车险的人，在某个特征上有没有明显区别？

# 进行分组
# 移动房车险数量 = 0  没买
# 移动房车险数量 = 1  买了
# 比较这两组客户的购买力水平
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示问题
# 1. 读取训练集
train = pd.read_excel("data/data.xlsx")

# 2. 画箱线图：购买力水平 vs 移动房车险数量
plot_data = train[["购买力水平", "移动房车险数量"]]

plot = plot_data.boxplot(column="购买力水平", by="移动房车险数量")

plot.set_xlabel("移动房车险数量")
plot.set_ylabel("购买力水平")
plot.set_title("购买力水平与移动房车险数量的关系")

plt.suptitle("")
plt.show()