import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

#计算AUC函数, 做性能度量
# input: y_true =[] and y_score=[]
# output: auc
def computeAUC(y_true,y_score):
    #计算并可视化AUC
    fpr, tpr, threshold = roc_curve(y_true,y_score)
    rocauc = auc(fpr,tpr)
    plt.plot(fpr,tpr,'b',label='AUC=%0.4f'% rocauc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return rocauc


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
    #print(dftrain)
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv("./data/cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()
    #2，使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new
    #3，使用Imputer将NaN替换为平均值
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    x_test = imp.transform(x_test)
    x_train = np.delete(x_train, 5, axis=1)
    x_test_new = np.delete(x_test_new, 5, axis=1)
    if not os.path.isfile("boost_model.m"):
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5,min_samples_leaf = 50,class_weight="balanced"), 
                                                        n_estimators = 10,
                                                        algorithm = 'SAMME.R', 
                                                        learning_rate = 0.4)
        clf.fit(x_train, y_train)
        joblib.dump(clf,"boost_model.m")
        predicted_probs_train =clf.predict_proba(x_train)
        predicted_probs_train = [x[1] for  x in predicted_probs_train]
        computeAUC(y_train, predicted_probs_train)
    else:
        clf = joblib.load("boost_model.m")
        predicted_probs_test_new = clf.predict_proba(x_test_new)
        predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
        computeAUC(y_test_new, predicted_probs_test_new)
    
    
if __name__ == "__main__":
    main()