#!/usr/bin/env python
# coding: utf-8

# In[9]:


frag = []

for line in open("vocab.txt"):
    frag.append(line.strip("\n").split(" ")[0])
    frag.append(line.strip("\n").split(" ")[1])
    
print(len(frag))

frag1 = set(frag)

print(len(frag1))


# In[10]:


from rdkit import Chem

frag2 = list(frag1)

with open("fragments/fragments.txt", "w") as output:
    for i in range(len(frag2)):      
        mol = Chem.MolFromSmiles(frag2[i])
        Chem.Draw.MolToFile(mol, "fragments/frag%d.png"%i)
        output.write(frag2[i] + '\n')


# In[11]:


s = 25

print(frag2[s])
mol = Chem.MolFromSmiles(frag2[s])
Chem.Draw.MolToMPL(mol)


# In[8]:


with open("fragments.txt", "w") as output:
    for i in range(len(frag2)): 
        output.write(frag2[i] + '\n')


# In[ ]:




