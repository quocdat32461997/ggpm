#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tqdm
import pickle
import random

train_emb = np.load("train_z.npy")
print(train_emb.shape)
train_card = train_emb.shape[0]

train_emb = train_emb.reshape((train_emb.shape[0], train_emb.shape[1] * train_emb.shape[2]))
train_emb = train_emb.tolist()

test_emb = np.load("chem_data/real_z.npy")
print(test_emb.shape)
test_card = test_emb.shape[0]

test_emb = test_emb.reshape((test_emb.shape[0], test_emb.shape[1] * test_emb.shape[2]))
test_emb = test_emb.tolist()


# In[2]:


train_homo = []
with open("chem_data/train_homo.txt", "rb") as f1:
    train_homo = pickle.load(f1)
    
test_homo = []
with open("chem_data/real_homo.txt", "rb") as f2:
    test_homo = pickle.load(f2)
    
f1.close()
f2.close()


# In[3]:


train_lumo = []
with open("chem_data/train_lumo.txt", "rb") as f1:
    train_lumo = pickle.load(f1)
    
test_lumo = []
with open("chem_data/real_lumo.txt", "rb") as f2:
    test_lumo = pickle.load(f2)
    
f1.close()
f2.close()


# In[4]:


train = []
for i in range(train_card):
    train.append([train_emb[i], train_homo[i]])
    
for i in range(30):
    train.append([test_emb[i], test_homo[i]])    

print(len(train))


# In[5]:


test = []
for i in range(30,65):
    test.append([test_emb[i], test_homo[i]])

print(len(test))


# In[6]:


x, y, y_pred, y_test, test_data = [], [], [], [], []

for i in range(len(train)):
    x.append(train[i][0])
    y.append(train[i][1])
    
for i in range(len(test)):
    test_data.append(test[i][0])
    y_test.append(test[i][1])


# In[7]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

regr = RandomForestRegressor(max_depth=12, min_samples_leaf = 1, n_jobs = -1, random_state = 50)
    
model = regr.fit(x, y)

y_pred = model.predict(test_data)

mse = mean_squared_error(y_test, y_pred)
print("MSE of random forest: ", mse)
print("RMSE of random forest: ", math.sqrt(mse))


# In[8]:


from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(random_state=0)
model = reg.fit(x, y)

y_pred = model.predict(test_data)

mse = mean_squared_error(y_test, y_pred)
print("MSE of GradientBoostingRegressor: ", mse)
print("RMSE of GradientBoostingRegressor: ", math.sqrt(mse))


# In[ ]:




