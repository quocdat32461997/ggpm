from evaluation.metrics import Metrics
from ggpm import *
import argparse
import pandas as pd
import os
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--output-data', required=True)
parser.add_argument('--train-data', required=True)
parser.add_argument('--mode', required=True,
                    help='To prescreen predictions or evaluate results of reconstruction or optimization')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--root', type=str,
                    help='Dir to store pretrained model & processed data for predicting HOMOs & LUMOs')
parser.add_argument('--path', type=str, help='Path to raw data')
parser.add_argument('--use-processed', action='store_true', default=False)


def check_smiles(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    valid = True
    if mol is None:
        valid = False
    else:
        try:
            Chem.SanitizeMol(mol)
        except:
            valid = False
    return valid


def load_smiles(args):
    # load data
    output_data = pd.read_csv(args.output_data)
    train_data = pd.read_csv(args.train_data)

    output_homos, output_lumos = None, None
    # get train, test, and output smiles
    train_smiles = train_data['SMILES'].tolist()
    try: # get smiles from output files
        test_smiles = output_data['original'].tolist()
        output_smiles = output_data['reconstructed'].tolist()
        output_homos = output_data['homo'].tolist()
        output_lumos = output_data['lumo'].tolist()
    except: # get smiles from test sets
        test_smiles = output_data['SMILES'].tolist()
        output_smiles = output_data['SMILES'].tolist()
    target_homos = output_data['org_homo'].tolist()
    target_lumos = output_data['org_lumo'].tolist()
    return output_smiles, test_smiles, train_smiles, target_homos, target_lumos, output_homos, output_lumos


def evaluate_recon(args):

    # compute metrics
    metrics = Metrics(homo_net=None, lumo_net=None, batch_size=args.batch_size)
    recon_metrics = metrics.get_recon_metrics(args.output_smiles, args.test_smiles, args.train_smiles)
    return recon_metrics


def evaluate_optim(args):
    from evaluation.schnet import SchNetwork
    # load ChemBERTa for property prediction
    homo_net = SchNetwork.from_qm9_pretrained(args.root, target=2)
    lumo_net = SchNetwork.from_qm9_pretrained(args.root, target=3)

    # compute metrics
    metrics = Metrics(homo_net=homo_net, lumo_net=lumo_net, batch_size=args.batch_size)
    optim_metrics = metrics.get_optim_metrics(args.root, args.path,
                                              args.output_smiles, args.test_smiles, args.train_smiles,
                                              use_processed=args.use_processed)
    
    # compute mae of homos and lumos
    mae_homos = F.l1_loss(torch.tensor(args.target_homos, dtype=torch.float), torch.tensor(args.output_homos, dtype=torch.float))
    mae_lumos = F.l1_loss(torch.tensor(args.target_lumos, dtype=torch.float), torch.tensor(args.output_lumos, dtype=torch.float))
    optim_metrics['mae_homos'] = mae_homos
    optim_metrics['mae_lumos'] = mae_lumos

    # get rid of valids
    del optim_metrics['prop_valids']
    del optim_metrics['prop_valid_idxs']
    del optim_metrics['mw_valids']
    del optim_metrics['mw_valid_idxs']
    return optim_metrics


def prescreen(args):
    # get valid molecules by property and molecular weight
    metrics = evaluate_optim(args)

    valid_idxs = metrics['mw_valid_idxs']
    #valid_idxs = [i for i in metrics['prop_valid_idxs'] if i in metrics['mw_valid_idxs']]

    # get valid molecules by RDKit
    return [args.output_smiles[idx] for idx in valid_idxs if check_smiles(args.output_smiles[idx])]


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.mode in ['recon', 'optim', 'prescreen']

    # load data
    args.output_smiles, args.test_smiles, args.train_smiles, args.target_homos, args.target_lumos, args.output_homos, args.output_lumos = load_smiles(args)

    if args.mode == 'recon':
        print(evaluate_recon(args))
    elif args.mode == 'optim':
        assert args.root is not None
        print(evaluate_optim(args))
    elif args.mode == 'prescreen':
        assert args.root is not None
        print(prescreen(args))

    else:
        raise Exception("{} is not a valid evaluate mode.".format(args.evaluate_mode))
