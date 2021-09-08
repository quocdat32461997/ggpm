#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

df = pd.read_csv("result.csv")

lines = df['result'].to_list()
print(len(lines))


# In[4]:


t = line[0].split('[')
print(t[0].strip(' '))
print(t[1].strip(']\''))


# In[6]:


ori = []
rec = []

for line in lines:
    s = line.split('[')
    ori.append(s[0].strip(' '))
    rec.append(s[1].strip(']\''))
    
print(len(ori), len(rec))


# In[7]:


new_df = pd.DataFrame({'original':ori, 'reconstruction':rec})
new_df.to_csv("Rec_result.csv")


# In[18]:


from rdkit import Chem

for i in range(len(rec)):
    try:
        mol = Chem.MolFromSmiles(rec[i])
    except:
        print(i)
        pass
    #img=Chem.Draw.MolToMPL(mol)


# In[ ]:




