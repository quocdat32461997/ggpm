import torch
from torch.utils.data import DataLoader
import moses
import pandas as pd
import argparse
import pickle

import rdkit
from ggpm import *
from ggpm.property_vae import *
from configs import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)
parser.add_argument('--optimize-type', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--optim-step', default=20)
parser.add_argument('--latent-lr', default=1.0, type=float)
parser.add_argument('--delta', default=0.01)
parser.add_argument('--threshold', default=0.05)
parser.add_argument('--patience', default=5)

# parse args
args = parser.parse_args()
path_to_config = args.path_to_config

# get configs
configs = Configs(path=path_to_config)

# override optimization configs
configs.optimize_type = args.optimize_type
configs.property_optim_step = args.optim_step
configs.property_delta = args.delta
configs.patience_threshold = args.threshold
configs.patience = args.patience
configs.latent_lr = args.latent_lr

if configs.test_data.endswith('.csv'):
    configs.test_data = pd.read_csv(configs.test_data)
    # drop row w/ empty HOMO and LUMO
    if configs.pretrained == False:
        configs.test_data = configs.test_data.dropna().reset_index(drop=True)
    configs.test_data = configs.test_data.to_numpy()
else:
    configs.test_data = [[x, float(x), float(l)] for line in open(configs.test_data) for x, h, l in line.strip("\r\n ").split()]

vocab = [x.strip("\r\n ").split() for x in open(configs.vocab_)]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
configs.vocab = PairVocab([(x,y) for x,y,_ in vocab])

# init and load core model
model = to_cuda(HierPropOptVAE(configs))
# Loading state_dict
try:
    model.load_state_dict(torch.load(configs.output_model,
                                     map_location=device))
except:
        # Try renaming due to unmatched name keys
        state_dict = rename_optimizer_state_keys(torch.load(configs.output_model,
                                                           map_location=device))
        model.load_state_dict(state_dict)
        del state_dict

# init property-control model
control_model = HierPropertyVAEOptimizer(model=model, args=configs)
control_model.eval()

dataset = MoleculeDataset(configs.test_data, configs.vocab, configs.atom_vocab, configs.batch_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

torch.manual_seed(configs.seed)
random.seed(configs.seed)
print('test size', len(configs.test_data))
total, acc, outputs = 0, 0, {'original': [], 'reconstructed': [],
                             'org_homo': [], 'org_lumo': [], 'homo': [], 'lumo': []}
logs = []
#with torch.no_grad():
for i, batch in enumerate(loader):
    orig_smiles = configs.test_data[configs.batch_size * i: configs.batch_size * (i + 1)]
    logs_, preds = control_model(batch, args=configs)
    properties, logs_, dec_smiles = (logs_, preds[0], preds[1]) if isinstance(preds, tuple) \
        else (([None]*configs.batch_size, [None]*configs.batch_size), logs_, preds)
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
