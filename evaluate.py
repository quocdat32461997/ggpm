import os
import torch
import pandas as pd
import argparse
import pickle
import rdkit

from ggpm import *
from ggpm import OPVNet
from torch_geometric.loader.dataloader import DataLoader
from configs import *

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--test-data', required=True)
parser.add_argument('--model-type', required=True)
args = parser.parse_args()

# load data
args.test_data = pd.read_csv(args.test_data)
# drop row w/ empty HOMO and LUMO
args.test_data = args.test_data.dropna().reset_index(drop=True)
args.test_data = args.test_data['SMILES'].to_numpy()

dataset = QM9Dataset(args.test_data)
data_loader = DataLoader(dataset, batch_size=16)
# test data_laoder
for x in data_loader:
    print(x)

# load model


# define metrics
#metrics = Metrics()
