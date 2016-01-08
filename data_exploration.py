import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sn

df = pd.read_csv("numerai_training_data.csv")
# sn.heatmap(df.ix[:, :14].corr())
# plt.show()

test_df = pd.read_csv("numerai_tournament_data.csv")

train_df = df[df.validation==0]
val_df = df[df.validation==1]
"""
print(train_df)
print(val_df)
print(test_df)
"""
# print(df.describe(percentiles=1.0, include='all'))
print(train_df.c1.value_counts())
print(val_df.c1.value_counts())
print(test_df.c1.value_counts())

print(set(val_df.c1.unique().tolist()) | set(train_df.c1.unique().tolist()) )
print(set(val_df.c1.unique().tolist()) & set(train_df.c1.unique().tolist()) )

# train_df['f14'].hist(bins=20)
# plt.show()
