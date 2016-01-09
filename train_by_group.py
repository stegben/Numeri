import sys
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import preprocess, split, to_array, write_ans

CLFS = {
    "RF": RandomForestClassifier(n_estimators=200, n_jobs=2),
    "GBDT": GradientBoostingClassifier(),
    # "SVC": SVC(probability=True),
}

TUNED_PARAMS = {
    "RF": [
           {'criterion': ["gini", "entropy"], 
            'max_depth':[7, 9, None], 
            'max_features':[0.05, 0.1, 0.3]}
          ],
    "GBDT": [
             {'loss': ['deviance', 'exponential'], 
              'learning_rate': [0.4, 1.0], 
              'n_estimators': [30, 100, 150], 
              'subsample': [0.3, 0.5, 0.7], 
              'max_depth': [2, 4, 6]}
            ]
}
"""
    "SVC": [
            #{'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
            {'kernel': ['rbf'], 'C': [10, 100, 800, 1000, 1200], 'gamma': [1.0, 0.05, 0.03, 'auto']},
           ],
"""


def main():
    ofname = sys.argv[1]
    
    df = pd.read_csv("numerai_training_data.csv")
    test_df = pd.read_csv("numerai_tournament_data.csv")
    gp = df.groupby("c1", as_index=False)
    
    test_ans = []
    test_id = []

    total_best = []

    for k, gp_df in gp:
        """
        df_tr, df_val = split(gp_df)
        Xtr, Ytr = to_array(df_tr)
        #print(sum(Ytr))
        Xval, Yval = to_array(df_val)
        """
        X, Y = to_array(df.drop(["validation", "c1"], axis=1))

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        tr_ind = gp_df[gp_df["validation"]==0].index.values.astype(int)
        val_ind = gp_df[gp_df["validation"]==1].index.values.astype(int)
        custom_CV_iterator = [(tr_ind, val_ind)]
        
        # start try out every model
        best_score = 0.
        best_clf = None
        for model_name in CLFS.keys():
            clf = GridSearchCV(CLFS[model_name],
                               param_grid=TUNED_PARAMS[model_name],
                               scoring='roc_auc',
                               pre_dispatch=1,
                               n_jobs=10, 
                               verbose=5,
                               cv=custom_CV_iterator
                               )
            clf.fit(X, Y)
            print(model_name)
            print("best score: {}".format(clf.best_score_))
            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_clf = clf.best_estimator_

        total_best.append((best_score, gp_df.shape))

        temp_test_df = test_df[test_df["c1"]==k]
        X_test = temp_test_df.ix[:,1:-1].as_matrix()
        X_test = scaler.transform(X_test)
        ans = best_clf.predict_proba(X_test)[:, 1].tolist()
        t_id = temp_test_df["t_id"].tolist()

        test_ans = test_ans + ans
        test_id = test_id + t_id

    output = pd.DataFrame({'t_id':test_id, 'probability':test_ans}, columns=['t_id', 'probability'])
    print("output: ")
    print(output)
    output.to_csv(ofname, index=False)
    pprint(total_best)


if __name__ == "__main__":
    main()

