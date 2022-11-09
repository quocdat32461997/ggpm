import evaluation.metrics as property_metrics
from configs import *
from ggpm import *
import argparse
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--output-data', required=True)
parser.add_argument('--train-data', required=True)
parser.add_argument('--evaluate-mode', required=True, help='To evaluate results of reconstruction or optimization')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--root', required=True, help='Dir to store processed data for predicting HOMOs & LUMOs')


def load_data(args):
    # load data
    output_data = pd.read_csv(args.output_data)
    train_data = pd.read_csv(args.train_data)

    # drop row w/ empty HOMO and LUMO
    output_data = output_data.dropna().reset_index(drop=True)

    # get train, test, and output smiles
    test_data = output_data['original'].tolist()
    output_data = output_data['reconstructed'].tolist()
    train_data = train_data['SMILES'].tolist()

    return output_data, test_data, train_data


def evaluate_recon(args):
    # load data
    output_data, test_data, train_data = load_data(args)

    # compute metrics
    metrics = property_metrics.Metrics(homo_net=None, lumo_net=None, batch_size=args.batch_size)
    recon_metrics = metrics.get_recon_metrics(output_data, test_data, train_data)
    return recon_metrics


def evaluate_optim(args):

    # load data
    output_data, test_data, train_data = load_data(args)

    # load ChemBERTa for property prediction
    homo_net = OPVNet.get_model('homo_net')(args.ckpt, target=2)
    lumo_net = OPVNet.get_model('lumo_net')(args.ckpt, target=3)

    # compute metrics
    metrics = property_metrics.Metrics(homo_net=homo_net, lumo_net=lumo_net, batch_size=args.batch_size)
    optim_metrics = metrics.get_optim_metrics()

    return optim_metrics


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.evaluate_mode in ['recon', 'optim']

    if args.evaluate_mode == 'recon':
        evaluate_recon(args)
    elif args.evaluate_mode == 'optim':
        evaluate_optim(args)
    else:
        raise Exception("{} is not a valid evaluate mode.".format(args.evaluate_mode))
