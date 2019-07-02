import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

#创建字典函数
#input: keys =[]and values=[]
#output: dict{}
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
    dftrain.pop("NOCredit")
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()
    dftest = pd.read_csv("./data/cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    dftest.pop("NOCredit")
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()
    #2，使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
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
    #x_train = np.delete(x_train, 5, axis=1)
    #x_test_new = np.delete(x_test_new, 5, axis=1)
    if not os.path.isfile("rfc_model.m"):
        clf = RandomForestClassifier(n_estimators=100,
                                    oob_score= True,
                                    min_samples_split=2,
                                    min_samples_leaf=50,
                                    n_jobs=-1,
                                    class_weight='balanced_subsample',
                                    bootstrap=True)
        
        #输出特征重要性评估
        clf.fit(x_train, y_train)
        param_grid = {"max_features": [2, 3, 4], "min_samples_leaf":[50]}
        grid_search = GridSearchCV(clf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False, n_jobs = -1)
        #c.输出最佳模型
        grid_search.fit(x_train, y_train)
        joblib.dump(grid_search,"rfc_model.m")
        print("the best parameter:", grid_search.best_params_)
        print("the best score:", grid_search.best_score_)
        predicted_probs_train = grid_search.predict_proba(x_train)
        predicted_probs_train = [x[1] for  x in predicted_probs_train]
        computeAUC(y_train, predicted_probs_train)
        print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), dftrain.columns), reverse=True))
    else:
        clf = joblib.load("rfc_model.m")
        predicted_probs_test_new = clf.predict_proba(x_test_new)
        predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
        computeAUC(y_test_new, predicted_probs_test_new)
        clf.fit(x_test_new, y_test_new)
        joblib.dump(clf, "rfc_model.m")
    
if __name__ == "__main__":
    main()
    
