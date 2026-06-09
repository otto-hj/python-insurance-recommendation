import pandas as pd
from sklearn import metrics, tree


def load_data(path):
    data = pd.read_csv(path, encoding="utf_8_sig")
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y


def build_model(x, y):
    # 调参部分在这里，max_depth限制树的最大深度 还有max_features每次切分时最多考虑多少个特征。 还有min_samples_split一个节点至少有多少样本才允许继续分裂。都可以调，criterion="entropy"表示用信息增益来选择特征，默认是 gini 不纯度。
    classifier = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
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
