import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
#from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.grid_search import GridSearchCV

from utils import preprocess, split, to_array, write_ans


TUNED_PARAMS = [
                {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
                {'kernel': ['rbf'], 'C': [10, 100, 800, 1000, 1200], 'gamma': [1.0, 0.05, 0.03, 'auto']}
               ]


def getSVC(df, random_split=None):
    X, Y = to_array(df.drop("validation", axis=1))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tr_ind = df[df["validation"]==0].index.values.astype(int)
    val_ind = df[df["validation"]==1].index.values.astype(int)
    custom_CV_iterator = [(tr_ind, val_ind)]
    print("Create a Random Forest Classifier")
    print("__Parameter searching...")
    # TODOs: cross-validation for best hyper parameter
    clf = GridSearchCV(SVC(probability=False),
                       param_grid=TUNED_PARAMS,
                       scoring='roc_auc',
                       n_jobs=10, 
                       verbose=5,
                       cv=custom_CV_iterator
                      )
    clf.fit(X, Y)
    print("Best score: {}".format(clf.best_score_))
    print("Best parameters: {}".format(clf.best_params_))
    return clf, scaler


if __name__ == "__main__":
    output_fname = sys.argv[1] 
    df, test_df = preprocess()
    model, scaler = getSVC(df)
    write_ans(model, test_df, ofname=output_fname, scaler=scaler)

