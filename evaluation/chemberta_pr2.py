import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel

MODEL_VERSION = 'seyonec/PubChem10M_SMILES_BPE_450k'

class ChemBertaForSinglePR2(torch.nn.Module):
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
        self.regressors.append(torch.nn.Linear(hidden_size_list[-1], 1))

        # loss
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()

    def forward(self, inputs, **kwargs):
        # get labels
        labels  = kwargs[kwargs['property_type']]

        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)

        # lumo
        outputs = outputs.view(-1)
        mae_loss = self.mae_loss(outputs, labels)
        mse_loss = self.mse_loss(outputs, labels)

        return mae_loss, {'mae': mae_loss.item(), 'mse': mse_loss.item()}

    def predict(self, inputs):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)

        return  outputs.view(-1)

class ChemBertaForTwoSplitPR2(torch.nn.Module):
    def __init__(self, hidden_size_list, dropout=0.1):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(MODEL_VERSION)
        embed_size = self.backbone.config.hidden_size

        self.homo_regressors = torch.nn.ModuleList()
        self.lumo_regressors = torch.nn.ModuleList()
        for hidden_size in hidden_size_list:
            modules = [torch.nn.Linear(embed_size, hidden_size),
                torch.nn.ReLU(),   
                torch.nn.Dropout(dropout)]
            self.homo_regressors.extend(modules)
            self.lumo_regressor.exend(modules)
            embed_size = hidden_size
        self.homo_regressors.append(torch.nn.Linear(hidden_size_list[-1], 1))
        self.lumo_regressors.append(torch.nn.Linear(hidden_size_list[-1], 1))

        # loss
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()

    def forward(self, inputs, **kwargs):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        homo_outputs = self.homo_regressors[0](outputs)
        lumo_outputs = self.lumo_regressors[0](outputs)
        for homo_layer, lumo_layer in zip(self.homo_regressors[1:], self.lumo_regressors[1:]):
            homo_outputs = self.homo_layer(homo_outputs)
            lumo_outputs = self.lumo_layer(lumo_outputs)

        # homo
        homo_outputs = homo_outputs.view(-1)
        homo_mae_loss = self.mae_loss(homo_outputs, kwargs['homo_labels'])
        homo_mse_loss = self.mse_loss(homo_outputs, kwargs['homo_labels'])

        # lumo
        lumo_outputs = lumo_outputs.view(-1)
        lumo_mae_loss = self.mae_loss(lumo_outputs, kwargs['lumo_labels'])
        lumo_mse_loss = self.mse_loss(lumo_outputs, kwargs['lumo_labels'])

        # total loss
        loss = homo_mae_loss + lumo_mae_loss
        return loss, {'homo_mae': homo_mae_loss.item(), 'homo_mse': homo_mse_loss.item(),
                      'lumo_mae': lumo_mae_loss.item(), 'lumo_mse': lumo_mse_loss.item()}

    def predict(self, inputs):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)

        return outputs[:, 0], outputs[:, 1]  # homos, lumos

class ChemBertaForTwoPR2(torch.nn.Module):
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

    def forward(self, inputs, **kwargs):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)

        # homo
        homo_mae_loss = self.mae_loss(outputs[:, 0], kwargs['homo_labels'])
        homo_mse_loss = self.mse_loss(outputs[:, 0], kwargs['homo_labels'])

        # lumo
        lumo_mae_loss = self.mae_loss(outputs[:, 1], kwargs['lumo_labels'])
        lumo_mse_loss = self.mse_loss(outputs[:, 1], kwargs['lumo_labels'])

        # total loss
        loss = homo_mae_loss + lumo_mae_loss
        return loss, {'homo_mae': homo_mae_loss.item(), 'homo_mse': homo_mse_loss.item(),
                      'lumo_mae': lumo_mae_loss.item(), 'lumo_mse': lumo_mse_loss.item()}

    def predict(self, inputs):
        # molecule embeddings
        outputs = self.backbone(inputs).pooler_output

        # regression
        for layer in self.regressors:
            outputs = layer(outputs)

        return outputs[:, 0], outputs[:, 1]  # homos, lumos


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
