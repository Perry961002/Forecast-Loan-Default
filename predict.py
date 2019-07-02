import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
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
colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59', 
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90', 
                'NORealEstate', 'NOTime60-89', 'NODependents']
col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
col_na_values = creatDictKV(colnames, col_nas)
dftest = pd.read_csv("./data/cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
dftest.pop("NOCredit")
test_id = [int(x) for x in dftest.pop("ID")]
y_test = np.asarray(dftest.pop("label"))
x_test = dftest.as_matrix()
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(x_test)
x_test = imp.transform(x_test)
ruu = [x[0] for x in x_test]
not9 = [x[5] for x in x_test]
not3 = [x[2] for x in x_test]
clf = joblib.load("rfc_model.m")
predicted_probs_test = clf.predict_proba(x_test)
predicted_probs_test = ["%.4f" % x[1] for x in predicted_probs_test]
submission = pd.DataFrame({'Id':test_id, 'Probability':predicted_probs_test, 'RUUnsecuredL':ruu, 'NOTimes90':not9, 'NOTime30-59':not3})
submission.to_csv("rf_benchmark.csv", index=False)