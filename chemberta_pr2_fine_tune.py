import sys
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict

#import torchtools
from torchtools import EarlyStopping
from configs import *
from ggpm.nnutils import to_cuda

# define command line argus
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-config', required=True)
parser.add_argument('--homo-col', required=True)
parser.add_argument('--lumo-col', required=True)
args = parser.parse_args()

MODEL_VERSION = 'seyonec/PubChem10M_SMILES_BPE_450k'
PROPERTY_MAP = {'homo': 'HOMO', 'lumo': 'LUMO'}


class ChemBertaForPR2(torch.nn.Module):
    def __init__(self, hidden_size_list, dropout=0.1):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(MODEL_VERSION)
        embed_size = self.backbone.config.hidden_size

        self.regressors = torch.nn.ModuleList()
        for hidden_size in hidden_size_list:
            self.regressors.extend([
                torch.nn.Linear(embed_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
            embed_size = hidden_size
        self.regressors.append(torch.nn.Linear(hidden_size_list[-1], 2))

        # loss
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()

    def forward(self, inputs, homo_labels, lumo_labels):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)


        # flatten outputs
        homo_outputs = outputs[:, 0]
        lumo_outputs = outputs[:, 1]

        # homo
        homo_mae_loss = self.mae_loss(homo_outputs, homo_labels)
        homo_mse_loss = self.mse_loss(homo_outputs, homo_labels)

        # lumo
        lumo_mae_loss = self.mae_loss(lumo_outputs, lumo_labels)
        lumo_mse_loss = self.mse_loss(lumo_outputs, lumo_labels)

        # total loss
        loss = homo_mae_loss + lumo_mae_loss
        return loss, {'homo_mae': homo_mae_loss.item(), 'homo_mse': homo_mse_loss.item(),
                'lumo_mae': lumo_mae_loss.item(), 'lumo_mse': lumo_mse_loss.item()}


class PR2Dataset(Dataset):

    def __init__(self, path, smiles_col, homo_col, lumo_col):
        assert homo_col in ['HOMO', 'LUMO']
        assert lumo_col in ['HOMO', 'LUMO']

        data = pd.read_csv(path)
        self.smiles = data[smiles_col]
        self.homo_labels = data[homo_col]
        self.lumo_labels = data[lumo_col]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        homo_label = self.homo_labels[index]
        lumo_label = self.lumo_labels[index]
        return smiles, homo_label, lumo_label


def fine_tune(args):
    # get configs
    configs = Configs(path=args.path_to_config)

    # get dataset
    train_dataset = PR2Dataset(configs.data, smiles_col='SMILES', homo_col=PROPERTY_MAP[args.homo_col], lumo_col=PROPERTY_MAP[args.lumo_col])
    val_dataset = PR2Dataset(configs.val_data, smiles_col='SMILES', homo_col=PROPERTY_MAP[args.homo_col], lumo_col = PROPERTY_MAP[args.lumo_col])

    # get tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_VERSION)

    # build model
    model = ChemBertaForPR2(hidden_size_list=configs.hidden_size_list, dropout=configs.dropout)
    model = to_cuda(model)

    # saver configs
    configs.to_json(configs.save_dir + '/configs.json')

    # optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, configs.anneal_rate)

    # earlystopping
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, delta=configs.patience_delta,
                                   path=configs.save_dir + '/model.{}'.format('best'))
    # train
    total_step = 0
    metrics = defaultdict(float)
    for epoch in range(configs.epoch):
        train_dataloader = DataLoader(train_dataset, configs.batch_size)
        for smiles, homo_labels, lumo_labels in train_dataloader:
            total_step += 1
            model.zero_grad()
            model.train()

            # encode data
            smiles_tokens = tokenizer(smiles, return_tensors='pt', add_special_tokens=True, padding=True)
            smiles_tokens = smiles_tokens['input_ids']
            loss, metrics_ = model(to_cuda(smiles_tokens), to_cuda(homo_labels), to_cuda(lumo_labels))

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

            if configs.save_iter >= 0 and total_step % configs.save_iter == 0:
                n_iter = total_step // configs.save_iter - 1
                torch.save(model.state_dict(), configs.save_dir + "/model." + str(n_iter))
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            # evaluate
            if total_step % configs.eval_iter == 0:
                model.eval()
                val_metrics = defaultdict(float)
                with torch.no_grad():
                    val_dataloader = DataLoader(val_dataset, configs.batch_size)
                    for smiles, homo_labels, lumo_labels in val_dataloader:
                        smiles_tokens = tokenizer(smiles, return_tensors='pt', add_special_tokens=True, padding=True)
                        smiles_tokens = smiles_tokens['input_ids']
                        loss, metrics_ = model(to_cuda(smiles_tokens), to_cuda(homo_labels), to_cuda(lumo_labels))

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
