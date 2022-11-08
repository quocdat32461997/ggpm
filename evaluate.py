from evaluation import *
from configs import *
from ggpm import *
import argparse
import pandas as pd
import torch
from transformers import RobertaTokenizer
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUe'


# get path to config
parser = argparse.ArgumentParser()
parser.add_argument('--output-data', required=True)
parser.add_argument('--train-data', required=True)
parser.add_argument('--path-to-config', required=True)
parser.add_argument('--evaluate-mode', required=True)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--model-type', required=True)

def evaluate_recon(args):
    pass


def evaluate_optim(args):
    # load data
    output_data = pd.read_csv(args.output_data)
    train_data = pd.read_csv(args.train_data)

    # drop row w/ empty HOMO and LUMO
    output_data = output_data.dropna().reset_index(drop=True)
    
    # get train, test, and output smiles
    #test_data = output_data['original'].tolist()
    #output_data = output_data['reconstructed'].tolist()
    test_data = None
    output_data = output_data['SMILES'].tolist()
    train_data = train_data['SMILES'].tolist()

    # load ChemBERTa for property prediction
    configs = Configs(args.path_to_config)
    model_class = OPVNet.get_model(args.model_type)
    chemberta = model_class(hidden_size_list=configs.hidden_size_list)
    chemberta.load_state_dict(torch.load(configs.save_dir + '/model.best', map_location=device))
    chemberta = to_cuda(chemberta)
    chemberta.eval()
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_VERSION)

    # compute metrics
    metrics = Metrics(property_predictor=chemberta, tokenizer=tokenizer, batch_size=args.batch_size)
    metrics = metrics.get_optimization_metrics(output_data, test_data, train_data)

    print(metrics)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.evaluate_mode in ['recon', 'optim']

    if args.evaluate_mode == 'recon':
        evaluate_recon(args)
    elif args.evaluate_mode == 'optim':
        evaluate_optim(args)
    else:
        raise Exception("{} is not a valid evaluate mode.".format(args.evaluate_mode))
