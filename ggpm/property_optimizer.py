import torch
from torch import nn


class PropertyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PropertyOptimizer, self).__init__()
        self.input_size = input_size

        # hidden_size to list
        hidden_size = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        hidden_size = [input_size] + hidden_size

        # define homo and lumo linear head
        self.homo_linear = PropertyRegressor(hidden_size, dropout)
        self.lumo_linear = PropertyRegressor(hidden_size, dropout)

    def compute_loss(self, outputs, targets):
        # get last dim of outputs of loss-computing since each property has only 1 score
        return torch.nn.MSELoss(reduction='mean')(outputs, targets)

    def forward(self, homo_vecs, lumo_vecs, targets):
        # Input:
        #   - homo_vecs: torch.Tensor
        #   - lumo_vecs: torch.Tensor
        #   - targets: tuple of HOMO and LUMO targets

        # make predictions
        homo_outputs, lumo_outputs = self.predict(homo_vecs, lumo_vecs)

        # compute loss
        homo_loss = self.compute_loss(homo_outputs, targets[0])
        lumo_loss = self.compute_loss(lumo_outputs, targets[1])

        return homo_loss, lumo_loss, homo_outputs, lumo_outputs

    def predict(self, homo_vecs, lumo_vecs):
        # function to flatten property prediction
        def _flatten(x):
            if x.shape[-1] == 1:
                if len(x.shape) == 1:
                    x = x[-1]
                elif len(x.shape) > 1:
                    x = x[:, -1]
            return x

        # make predictions
        homo_vecs = self.homo_linear(homo_vecs)
        lumo_vecs = self.lumo_linear(lumo_vecs)

        # reshape to linear if last-dim == 1
        return _flatten(homo_vecs), _flatten(lumo_vecs)


class PropertyRegressor(nn.Module):
    def __init__(self, hidden_size: list, dropout: float):
        super(PropertyRegressor, self).__init__()
        self.linear = nn.ModuleList()
        for idx in range(len(hidden_size) - 1):
            self.linear.extend([nn.Linear(hidden_size[idx], hidden_size[idx + 1]),
                                nn.ReLU(), nn.Dropout(dropout)])
        self.linear.append(nn.Linear(hidden_size[-1], 1))

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
        return x