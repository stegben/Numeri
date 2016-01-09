
import numpy as np
import pandas as pd


def preprocess():
    df = pd.read_csv("numerai_training_data.csv")
    test_df = pd.read_csv("numerai_tournament_data.csv")
    
    df_dummy = pd.get_dummies(df, columns=["c1"])
    test_df_dummy = pd.get_dummies(test_df, columns=["c1"])
    return df_dummy, test_df_dummy


def split(df, rand_ratio=None):
    if rand_ratio is not None:
        assert type(rand_ratio) is float
        df = df.drop("validation")
        rows = random.sample(df.index, df.as_matrix().shape[0]*rand_ratio)
        train_df = df.ix[rows]
        val_df = df.drop(rows)
    else:
        train_df = df[df["validation"]==0]
        #print(train_df.columns)
        train_df.drop("validation", axis=1, inplace=True)
        val_df = df[df.validation==1]
        val_df.drop("validation", axis=1, inplace=True)
    return train_df, val_df


def to_array(df):
    #print(df.drop("target", axis=1).columns)
    x = df.drop("target", axis=1).as_matrix()
    y = df.ix[:, 'target'].as_matrix()
    return x, y


def write_ans(model, df, ofname, scaler=None):
    X_test = df.ix[:,1:].as_matrix()
    if scaler is not None:
        X_test = scaler.transform(X_test)
    ans = model.predict_proba(X_test)
    ans = ans.flatten()
    print(ans)
    t_id = df["t_id"]
    output = pd.DataFrame({'t_id':t_id, 'probability':ans}, columns=['t_id', 'probability'])
    print("output: ")
    print(output)
    output.to_csv(ofname, index=False, float_format='%2.8f')
    
