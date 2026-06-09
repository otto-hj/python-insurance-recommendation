import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_validate


def load_data(path):
    data = pd.read_csv(path, encoding="utf_8_sig")
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y


def build_model(x, y):
    # 让决策树根据 x 和 y 学习规律，并且把学到的规律保存在 classifier 这个对象里。
    classifier = tree.DecisionTreeClassifier(random_state=0)
    classifier.fit(x, y)
    return classifier


def test_model(classifier):
    # 使用测试集对模型进行评价，得到模型的准确率、精确率、召回率、F1值和AUC值。
    test_x, test_y = load_data("data/test_preprocess.csv")
    # 计算模型评价指标，使用交叉验证的方式，cv=5 表示把测试集分成 5 份，每次用其中 4 份来测试模型，剩下的 1 份来计算指标，重复 5 次，最后取平均值。
    scores = cross_validate(
        # 这里有问题把 test_preprocess.csv这个文件里的数据：拆成5份，然后：一部分当训练集一部分当测试集轮流进行。
        # 把数据切成 5 份，轮流让其中1 份当测试集，剩下 4 份当训练集
        classifier,
        test_x,
        test_y,
        cv=5,
        scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
    )
    return scores


if __name__ == "__main__":
    # 读取训练数据
    # train_x 是训练集的特征变量，train_y 是训练集的目标变量
    train_x, train_y = load_data("data/train_preprocess.csv")
    # 训练决策树模型，通过训练数据学习分类规则
    classifier = build_model(train_x, train_y)
    # 使用测试集对模型进行评价
    scores = test_model(classifier)

    print("Accuracy %.4f" % np.mean(scores["test_accuracy"]))
    print("Precision %.4f" % np.mean(scores["test_precision"]))
    print("Recall %.4f" % np.mean(scores["test_recall"]))
    print("F1 %.4f" % np.mean(scores["test_f1"]))
    print("AUC %.4f" % np.mean(scores["test_roc_auc"]))
