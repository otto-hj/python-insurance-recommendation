# python-insurance-recommendation
基于机器学习的保险产品推荐系统
# Insurance Product Recommendation

## 项目简介

本项目基于保险公司客户历史数据，利用机器学习方法预测客户是否购买移动房车险。

通过对客户属性、历史保险记录等特征进行分析，构建分类模型实现保险产品推荐。

---

## 数据集

- 训练集：5822条记录
- 测试集：4000条记录
- 特征数量：86个

---

## 数据处理流程

1. 缺失值分析
2. 特征类别统计
3. 偏度分析
4. 箱线图分析
5. 特征选择
6. 数据平衡处理

---

## 使用模型

### 决策树

- Gini Index

### 逻辑回归

- Logistic Regression

### ID3 决策树

- Information Gain

### 随机森林

- Random Forest

---

## 评价指标

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

---

## 项目结构

```text
data/
dataclean*.py
model*.py
README.md
```

---
## Author

otto-hj