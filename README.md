# ggpm
GGPM - GraphNN Generation of Photovoltaic Molecules

## Training Steps
### Step 1: Extracting fragment vocabs
```buildoutcfg
python3 get_vocab.py --data path\to\SMILES.csv --output path\to\save\vocabs.txt
```
**NOTICE**
- Used the fragment convertor from [hg2g](https://github.com/wengong-jin/hgraph2graph)
- Skip molecules not read by rdkit  
- Skip molecules w/ *

### Step 2: Cleaning * Preprocessing training data
- HOMO and LOMO values from different dataset vary by scale
-> *standardizing*
- All SMILES containing * asterisks are removed from training data

### Step 3: Training
