import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def load_data(path):
    data = pd.read_csv(path, encoding="utf_8_sig")
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y


def build_model(x, y):
    classifier = RandomForestClassifier(
        criterion="entropy",
        max_depth=10,
        n_estimators=100,
        random_state=0,
    )
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
    train_x, train_y = load_data("data/train_balance.csv")
    classifier = build_model(train_x, train_y)
    test_model(classifier)
