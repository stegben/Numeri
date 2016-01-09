import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

from sklearn.grid_search import GridSearchCV

from utils import preprocess, split, to_array, write_ans

TUNED_PARAMS = [
             {'loss': ['deviance', 'exponential'], 
              'learning_rate': [0.1, 0.4, 1.0], 
              'n_estimators': [30, 75, 100, 150, 200], 
              'subsample': [0.5, 0.7], 
              'max_depth': [2, 3, 4, 5, 6, 7, 8]}
            ]


def getGBDT(df, random_split=None):
    X, Y = to_array(df.drop("validation", axis=1))
    tr_ind = df[df["validation"]==0].index.values.astype(int)
    val_ind = df[df["validation"]==1].index.values.astype(int)
    custom_CV_iterator = [(tr_ind, val_ind)]
    print("Create a GBDT Classifier")
    # TODOs: cross-validation for best hyper parameter
    clf = GridSearchCV(GradientBoostingClassifier(),
                       param_grid=TUNED_PARAMS,
                       scoring='roc_auc',
                       n_jobs=20, 
                       verbose=5,
                       cv=custom_CV_iterator
                      )
    clf.fit(X, Y)
    print("Best score: {}".format(clf.best_score_))
    print("Best parameters: {}".format(clf.best_params_))
    return clf


if __name__ == "__main__":
    output_fname = sys.argv[1]
    df, test_df = preprocess()
    model = getGBDT(df)
    write_ans(model, test_df, ofname=output_fname)

