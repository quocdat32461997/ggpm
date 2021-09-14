#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame as df
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[2]:


train_emb = np.load("tmp/z1.npy")
print(train_emb.shape)
train_card = train_emb.shape[0]
train_emb = train_emb.reshape((train_emb.shape[0], train_emb.shape[1] * train_emb.shape[2]))
train_emb = train_emb.tolist()


# In[3]:


test_emb = np.load("tmp/z2.npy")
print(test_emb.shape)
test_card = test_emb.shape[0]
test_emb = test_emb.reshape((test_emb.shape[0], test_emb.shape[1] * test_emb.shape[2]))
test_emb = test_emb.tolist()


# In[4]:


train_homo = []
with open("chem_data/train_homo.txt", "rb") as f1:
    train_homo = pickle.load(f1)
    
f1.close()


# In[5]:


df_1 = pd.read_csv("tmp/smiles.csv")
test_homo = df_1['HOMO'].to_list()


# In[6]:


train_emb = train_emb + test_emb[:30]
test_emb = test_emb[30: 64]

train_homo = train_homo + test_homo[:30]
test_homo = test_homo[30: 64]


# In[7]:


feature_columns = []
train_xdf = df(train_emb, columns=[str(i) for i in range(32)])
train_ydf = df(train_homo, columns=["homo"])

test_xdf = df(test_emb, columns=[str(i) for i in range(32)])
test_ydf = df(test_homo, columns=["homo"])

train_df = pd.concat([train_xdf, train_ydf], axis = 1)
test_df = pd.concat([test_xdf, test_ydf], axis = 1)

columns = [str(i) for i in range(32)]
for feature_name in columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# In[9]:


regression_classifier = tf.estimator.BoostedTreesRegressor

NUM_EXAMPLES = len(train_ydf)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = (dataset
                   .repeat(n_epochs)
                   .batch(NUM_EXAMPLES))
        return dataset
    print("dataset done.")

    return input_fn


train_input_fn = make_input_fn(train_df, train_homo)

model = regression_classifier(feature_columns=feature_columns,
                              n_trees=50,
                              max_depth=3,
                              n_batches_per_layer=1)
model.train(input_fn=train_input_fn)


# In[11]:


eval_input_fn = make_input_fn(test_df, test_homo, shuffle=False, n_epochs=1)
results = model.evaluate(eval_input_fn)
pd.Series(results).to_frame()


# In[12]:


pred_dicts = list(model.predict(eval_input_fn))
preds = [pred['predictions'].astype(float) for pred in pred_dicts]


# In[14]:


from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(test_homo, preds)
print("mse: ", mse)
print("rmse: ", math.sqrt(mse))


# In[ ]:




