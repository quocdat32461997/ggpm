#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd

df = pd.read_csv("generation.csv")

smiles = df['smiles'].to_list()


# In[17]:


from rdkit import Chem

for i in range(len(smiles)):      
    mol = Chem.MolFromSmiles(smiles[i])
    Chem.Draw.MolToFile(mol, "gen_result/mol%d.gen.png"%i)


# In[ ]:




