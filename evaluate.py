from evaluation.metrics import Metrics
from ggpm import *
import argparse
import pandas as pd
import os
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
    print(smi, valid)
    return valid


def load_smiles(args):
    # load data
    output_data = pd.read_csv(args.output_data)
    train_data = pd.read_csv(args.train_data)

    # drop row w/ empty HOMO and LUMO
    output_data = output_data.dropna().reset_index(drop=True)

    # get train, test, and output smiles
    test_smiles = output_data['original'].tolist()
    output_smiles = output_data['reconstructed'].tolist()
    train_smiles = train_data['SMILES'].tolist()

    return output_smiles, test_smiles, train_smiles


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

    return optim_metrics


def prescreen(args):
    # get valid molecules by property and molecular weight
    metrics = evaluate_optim(args)
    valid_idxs = metrics['valid_idxs']

    # get valid molecules by RDKit
    valid_idxs = torch.tensor([idx for idx in valid_idxs if check_smiles(args.output_smiles[idx])],
                              dtype=torch.int)

    # combine all valids
    valids = metrics['prop_valids'] * metrics['mw_valids']
    valids = valids == 1
    valid_idxs = torch.masked_select(valid_idxs, valids)

    valid_smiles = [args.output_smiles[i] for i in valid_idxs]
    return valid_smiles


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.mode in ['recon', 'optim', 'prescreen']

    # load data
    args.output_smiles, args.test_smiles, args.train_smiles = load_smiles(args)

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
