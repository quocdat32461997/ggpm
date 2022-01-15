import torch
from torch.utils.data import DataLoader
import moses
import pandas as pd
import argparse

import rdkit
from ggpm import *

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)

# get configs
args = Configs(path=parser.parse_args().path_to_config)

if args.data.endswith('.csv'):
    args.data = list(pd.read_csv(args.data)['SMILES'])
    args.data = [line.strip("\r\n ") for line in args.data]
else:
    args.data = [line.strip("\r\n ") for line in open(args.data)]
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

model = to_cuda(PropertyVAE(args))

model.load_state_dict(torch.load(args.model,
                                 map_location=device))
model.eval()

dataset = MoleculeDataset(args.data, args.vocab, args.atom_vocab, args.batch_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

torch.manual_seed(args.seed)
random.seed(args.seed)

total, acc, outputs = 0, 0, {'original': [], 'reconstructed': []}
with torch.no_grad():
    for i,batch in enumerate(loader):
        orig_smiles = args.data[args.batch_size * i : args.batch_size * (i + 1)]
        dec_smiles = model.reconstruct(batch)
        for x, y in zip(orig_smiles, dec_smiles):
            # display results
            print(x, y)

            # add to outputs
            outputs['original'].append(x)
            outputs['reconstructed'].append(y)

# save outputs
outputs = pd.DataFrame.from_dict(outputs)
outputs.to_csv(args.output, index=False)