import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup

def main():
    #1，加载数据（训练和测试）和预处理数据
    #将NumberTime30-59，60-89，90中标记的96，98替换为NaN
    #将Age中的0替换为NaN
    colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59', 
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90', 
                'NORealEstate', 'NOTime60-89', 'NODependents']
    col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
    col_na_values = creatDictKV(colnames, col_nas)
    dftrain = pd.read_csv("./data/cs-training.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    dftrain.pop("NOCredit")
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()
    #使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new
    #使用Imputer将NaN替换为平均值
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    
    train_errors, test_errors = [], []
    for i in range(1, 50):
        print('max_depth: ', i)
        clf = DecisionTreeClassifier(max_depth=i, class_weight="balanced")
        clf.fit(x_train, y_train)
        y_train_predicted = clf.predict(x_train)
        y_test_predicted = clf.predict(x_test_new)
        train_errors.append(mean_squared_error(y_train_predicted, y_train))
        test_errors.append(mean_squared_error(y_test_predicted, y_test_new))
    plt.plot(np.sqrt(train_errors), "r-x", label="Train Set", linewidth=2)
    plt.plot(np.sqrt(test_errors), "b-o", label="Test Set", linewidth=2)
    plt.legend(loc='heighter right')
    plt.xlabel("Max Depth")
    plt.ylabel("RMSE")
    plt.show()
    
    train_errors, test_errors = [], []
    for i in range(1, 1000, 20):
        print('min_samples_leaf: ', i)
        clf = DecisionTreeClassifier( min_samples_leaf=i, class_weight="balanced")
        clf.fit(x_train, y_train)
        y_train_predicted = clf.predict(x_train)
        y_test_predicted = clf.predict(x_test_new)
        train_errors.append(mean_squared_error(y_train_predicted, y_train))
        test_errors.append(mean_squared_error(y_test_predicted, y_test_new))
    plt.plot(range(1, 1000, 20), np.sqrt(train_errors), "r-x", label="Train Set", linewidth=2)
    plt.plot(range(1, 1000, 20), np.sqrt(test_errors), "b-o", label="Test Set", linewidth=2)
    plt.legend(loc='heighter right')
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("RMSE")
    plt.show()


    
if __name__ == "__main__":
    main()