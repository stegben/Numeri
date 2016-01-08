import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

from utils import preprocess, split, to_array, write_ans


def getRF(df, random_split=None):
    df_tr, df_val = split(df, rand_ratio=random_split)
    
    X, Y = to_array(df.drop("validation", axis=1))
    Xtr, Ytr = to_array(df_tr)
    #print(sum(Ytr))
    Xval, Yval = to_array(df_val)
    #print(sum(Yval))
    print("Create a Random Forest Classifier")
    print("__Parameter searching...")
    # TODOs: cross-validation for best hyper parameter
    clf = RandomForestClassifier(n_estimators=10000, 
                                     max_depth=5,
                                     max_features=0.5,
                                     oob_score=True, 
                                     verbose=0,
                                     n_jobs=-1)
    clf.fit(Xtr, Ytr)
    print("OOB score: {}".format(clf.oob_score_))
    print("accuracy on train: {}".format(clf.score(Xtr, Ytr)))
    print("accuracy on validation: {}".format(clf.score(Xtr, Ytr)))
    pred_tr = clf.predict_proba(Xtr)[:, 1]
    pred = clf.predict_proba(Xval)[:, 1]
    print("auc on train: {}".format(roc_auc_score(Ytr, pred_tr)))
    print("auc on validation: {}".format(roc_auc_score(Yval, pred)))
    
    print("use best parameter to train on all data")
    clf = RandomForestClassifier(n_estimators=1000, 
                                     max_depth=5,
                                     max_features=0.2,
                                     oob_score=False, 
                                     verbose=1,
                                     n_jobs=-1)
    clf.fit(X, Y)
    return clf


if __name__ == "__main__":
    output_fname = sys.argv[1]
    
    df, test_df = preprocess()
    #print(test_df.columns)
    #print(df.columns)

    model = getRF(df)
    #print(test_df.columns)
    write_ans(model, test_df, ofname=output_fname)

