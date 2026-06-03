import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def load_data(path):
    data = pd.read_csv(path, encoding="utf_8_sig")
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y


def build_model(x, y):
    # 此处调参penalty="l2"表示使用 L2 正则化，C=1.0 是正则化强度的倒数，默认值是 1.0，表示不进行调整。solver="lbfgs" 是优化算法，适用于小数据集和多分类问题。max_iter=1000 表示最多迭代 1000 次。逻辑回归训练时需要不断迭代寻找合适参数，如果默认次数不够，可能会报收敛警告。所以这里设大一点，保证模型能正常训练。
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x, y)
    return classifier


def test_model(classifier):
    test_x, test_y = load_data("data/test_preprocess.csv")
    pred_y = classifier.predict(test_x)
    prob_y = classifier.predict_proba(test_x)[:, 1]

    print("Accuracy %.4f" % metrics.accuracy_score(test_y, pred_y))
    print("Precision %.4f" % metrics.precision_score(test_y, pred_y, zero_division=0))
    print("Recall %.4f" % metrics.recall_score(test_y, pred_y, zero_division=0))
    print("F1 %.4f" % metrics.f1_score(test_y, pred_y, zero_division=0))
    print("AUC %.4f" % metrics.roc_auc_score(test_y, prob_y))


if __name__ == "__main__":
    # train_x, train_y = load_data("data/train_balance.csv")
    train_x, train_y = load_data("data/train_preprocess.csv")
    classifier = build_model(train_x, train_y)
    test_model(classifier)
