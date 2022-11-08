import sys
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import torch.optim as torch_optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict

from torchtools import EarlyStopping
from configs import *
from ggpm import *
from evaluation import *

# define command line argus
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)
parser.add_argument('--model-type', required=True)
parser.add_argument('--property-type', default=None)
args = parser.parse_args()

assert args.property_type in ['homos', 'lumos']


def fine_tune(args):
    # get configs
    configs = Configs(path=args.path_to_config)

    # get dataset
    train_dataset = PR2Dataset(configs.data, smiles_col='SMILES',
                               homo_col='HOMO', lumo_col='LUMO')
    val_dataset = PR2Dataset(configs.val_data, smiles_col='SMILES',
                             homo_col='HOMO', lumo_col='LUMO')

    # get tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_VERSION)

    # build model
    model_class = OPVNet.get_model(args.model_type)
    model = model_class(hidden_size_list=configs.hidden_size_list, dropout=configs.dropout)
    model = to_cuda(model)

    # saver configs
    configs.to_json(configs.save_dir + '/configs.json')

    # optimizer & scheduler
    optimizer = torch_optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, configs.anneal_rate)

    # earlystopping
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=configs.patience_delta,
                                   path=configs.save_dir + '/model.{}'.format('best'))
    # train
    total_step = 0
    metrics = defaultdict(float)
    for epoch in range(configs.epoch):
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True)
        for smiles, homo_labels, lumo_labels in train_dataloader:
            total_step += 1
            model.zero_grad()
            model.train()

            # encode data
            smiles_tokens = tokenizer(smiles, return_tensors='pt', add_special_tokens=True, padding=True)
            smiles_tokens = smiles_tokens['input_ids']
            loss, metrics_ = model(inputs=to_cuda(smiles_tokens), homos=to_cuda(homo_labels), 
                                   lumos=to_cuda(lumo_labels), property_type=args.property_type)

            # backprop
            loss.backward()
            optimizer.step()

            # accumulate metrics
            for k, v in metrics_.items():
                metrics[k] += v
            metrics['loss'] += loss.item()

            if total_step % configs.print_iter == 0:
                metrics = {k: v / configs.print_iter for k, v in metrics.items()}
                print("[%d] " % total_step, ', '.join([k + ': %.3f' % v for k, v in metrics.items()]))
                sys.stdout.flush()
                metrics = defaultdict(float)

            if total_step % configs.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if configs.save_iter >= 0 and total_step % configs.save_iter == 0:
                n_iter = total_step // configs.save_iter - 1
                torch.save(model.state_dict(), configs.save_dir + "/model." + str(n_iter))

            # evaluate
            if total_step % configs.eval_iter == 0:
                model.eval()
                val_metrics = defaultdict(float)
                with torch.no_grad():
                    val_dataloader = DataLoader(val_dataset, configs.batch_size)
                    for smiles, homo_labels, lumo_labels in val_dataloader:
                        smiles_tokens = tokenizer(smiles, return_tensors='pt', add_special_tokens=True, padding=True)
                        smiles_tokens = smiles_tokens['input_ids']
                        loss, metrics_ = model(inputs=to_cuda(smiles_tokens), homos=to_cuda(homo_labels), lumos=to_cuda(lumo_labels), property_type=args.property_type)

                        # accumulate metrics
                        for k, v in metrics_.items():
                            val_metrics[k] += v
                        val_metrics['loss'] += loss.item()

                # average val loss & metrics
                n = len(val_dataloader)
                val_metrics = {k: v / n for k, v in val_metrics.items()}
                # print metrics
                print("[%d] " % total_step, ', \
                    '.join([k + ': %.3f' % v for k, v in val_metrics.items()]))
                sys.stdout.flush()

                del val_dataloader

                # update early_stopping
                if configs.early_stopping:
                    early_stopping(val_metrics['loss'], model)
                    if early_stopping.early_stop:
                        break
        if configs.early_stopping and early_stopping.early_stop:
            print('Stop: early stopping')
            break

        del train_dataloader
        if configs.save_iter == -1:
            torch.save(model.state_dict(), configs.save_dir + "/model." + str(epoch))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])


if __name__ == '__main__':
    fine_tune(args)
