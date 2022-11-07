#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


smiles_1 = [line.strip("\r\n ") for line in open("tmp/smiles.txt")]

df = pd.read_csv("chem_data/clean_data.csv")
smiles_2 = df['SMILES'].to_list()



# In[9]:


for i in range(len(smiles_2)):
    if smiles_2[i] not in smiles_1:
        print(i)


# In[10]:


homo_list = df['HOMO'].to_list()
lumo_list = df['LUMO'].to_list()


# In[11]:


del homo_list[2]
del homo_list[37]
del homo_list[37]
del homo_list[37]
del homo_list[37]
del homo_list[58]
del homo_list[58]

del lumo_list[2]
del lumo_list[37]
del lumo_list[37]
del lumo_list[37]
del lumo_list[37]
del lumo_list[58]
del lumo_list[58]

print(len(homo_list), len(lumo_list))


# In[13]:


l = []
l.append(smiles_1)
l.append(homo_list)
l.append(lumo_list)

df1 = pd.DataFrame(l).transpose()
df1.columns = ["SMILES", "HOMO", "LUMO"]

print(df1)


# In[14]:


df1.to_csv("tmp/smiles.csv", index=False)


# In[7]:


smiles_2 = [line.strip("\r\n ") for line in open("tmp/train.txt")]

df2 = pd.read_csv("chem_data/clean_data.csv")
smiles_2 = df['SMILES'].to_list()


# In[ ]:




