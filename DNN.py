import sys
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adadelta
from keras.layers.advanced_activations import PReLU, ELU

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import preprocess, split, to_array, write_ans


def getDNN(df, random_split=None):
    df_tr, df_val = split(df, rand_ratio=random_split)
    
    X, Y = to_array(df.drop("validation", axis=1))
    Xtr, Ytr = to_array(df_tr)
    Xval, Yval = to_array(df_val)

    scaler = MinMaxScaler((0, 1))
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.transform(Xval)

    # Start create model
    print("Create a DNN Classifier")
    model = Sequential()

    model.add(Dense(100, input_dim=Xtr.shape[1], activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='linear'))
    model.add(ELU(alpha=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='linear'))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # trainer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    trainer = Adadelta(lr=0.1, tho=0.98, epsilon=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=trainer)
    
    print(Ytr, Yval)
    model.fit(Xtr, Ytr, nb_epoch=30, batch_size=32, verbose=1, validation_data=(Xval, Yval))


    pred_tr = model.predict_proba(Xtr)
    pred = model.predict_proba(Xval)
    print("auc on train: {}".format(roc_auc_score(Ytr, pred_tr)))
    print("auc on validation: {}".format(roc_auc_score(Yval, pred)))

    X = scaler.fit_transform(X)
    model.fit(X, Y, nb_epoch=30, batch_size=32)
    return model, scaler


if __name__ == "__main__":
    output_fname = sys.argv[1]
    
    df, test_df = preprocess()
    #print(test_df.columns)
    #print(df.columns)

    model, scaler = getDNN(df)
    #print(test_df.columns)
    write_ans(model, test_df, ofname=output_fname, scaler=scaler)

