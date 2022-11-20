# ggpm
GGPM - GraphNN Generation of Photovoltaic Molecules

* [Thesis Paper](https://github.com/quocdat32461997/ggpm/blob/main/UTD_Dissertations_and_Theses___Dat__Eric__Ngo-11.pdf)

## Setup environments
For easy, run
```
conda env create _ENV_NAME_ -f chem_env.yml
```

## Training Steps
### Step 1: Extracting motif vocabs from training sets only
```
python3 get_vocab.py --data path/to/training_set.csv --output path/to/save/vocabs.txt
```

### Step 2: Serializing molecules by DFS
```
# Training Set
python3 preprocess.py --train path/to/training_set.csv --vocab path/to/save/vocabs.txt --batch-size 20 --ncpu 1

# Validation Set
python3 preprocess.py --train path/to/validation_set.csv --vocab path/to/save/vocabs.txt --batch-size 20 --ncpu 1
```

### Step 3: Pre-Training
```
python3 vae_train.py --path-to-config path/to/pretrain_configs.json
```

### Step 4: Fine-Tunining
```

# Fine-tuning methods: early-stopping & uncertainty-loss-scaling
python3 vae_fine_tune.py --path-to-config path/to/fine_tune_configs.json

# Fine-tuning: individual optimizers for each subnetwork
python3 vae_fine_tune_indv_opt.py --path-to-config path/to/fine_tune_configs.json
```

## Inference

### Molecule Reconstruction
```
python3 reconstruct.py --model model_type --path-to-config path/to/reconstruction_configs.json
```

### Property-guided Molecule Optimization
```
python3 optimizer.py --model model_type --path-to-config path/to/configs.json --optimize-type type --output output/file/name.csv --optim-step max_decoding_steps --latent-lr LR_to_update_gradients --delta improvement_threshold --threshold mse_gap_threshold --patience no_improvement_patience
```

**NOTICE**
- Used the fragment convertor from [hg2g](https://github.com/wengong-jin/hgraph2graph)
- Skip molecules not read by rdkit  
- Skip molecules w/ *
- All SMILES containing * asterisks are removed from training data
