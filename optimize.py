import torch
from torch.utils.data import DataLoader
import moses
import pandas as pd
import argparse
import pickle

import rdkit
from ggpm import *
from ggpm.property_vae import PropertyVAE, PropOptVAE
from configs import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)

# parse args
args = parser.parse_args()
path_to_config = args.path_to_config

# get configs
args = Configs(path=path_to_config)

if args.test_data.endswith('.csv'):
    args.test_data = pd.read_csv(args.test_data)
    # drop row w/ empty HOMO and LUMO
    if args.pretrained == False:
        args.test_data = args.test_data.dropna().reset_index(drop=True)
    args.test_data = args.test_data.to_numpy()
else:
    args.test_data = [[x, float(x), float(l)] for line in open(args.test_data) for x, h, l in line.strip("\r\n ").split()]

vocab = [x.strip("\r\n ").split() for x in open(args.vocab_)]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

# init and load core model
model = to_cuda(PropOptVAE(args))
state_dict = rename_optimizer_state_keys(torch.load(args.output_model, map_location=device))
model.load_state_dict(state_dict)
del state_dict

# init property-control model
control_model = PropertyVAEOptimizer(model=model, args=args)
control_model.eval()

dataset = MoleculeDataset(args.test_data, args.vocab, args.atom_vocab, args.batch_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

torch.manual_seed(args.seed)
random.seed(args.seed)
print('test size', len(args.test_data))
total, acc, outputs = 0, 0, {'original': [], 'reconstructed': [],
                             'org_homo': [], 'org_lumo': [], 'homo': [], 'lumo': []}
logs = []
#with torch.no_grad():
for i, batch in enumerate(loader):
    orig_smiles = args.test_data[args.batch_size * i: args.batch_size * (i + 1)]
    logs_, preds = control_model(batch, args=args)
    properties, logs_, dec_smiles = (logs_, preds[0], preds[1]) if isinstance(preds, tuple) \
        else (([None]*args.batch_size, [None]*args.batch_size), logs_, preds)
    logs.extend(logs_)
    for x, y, h, l in zip(orig_smiles, dec_smiles, properties[0], properties[1]):
        # extract original labels
        x, h_, l_ = x

        # display results
        print('Org: {}, Dec: {}, HOMO: {}, LUMO: {}'.format(x, y, h, l))

        # add to outputs
        outputs['original'].append(x)
        outputs['reconstructed'].append(y)
        outputs['org_homo'].append(h_)
        outputs['org_lumo'].append(l_)
        outputs['homo'].append(h if h is None else h.item())
        outputs['lumo'].append(l if l is None else l.item())

# save outputs
outputs = pd.DataFrame.from_dict(outputs)
outputs.to_csv(args.output, index=False)

with open('logs.pkl', 'wb') as file:
    pickle.dump(logs, file)
