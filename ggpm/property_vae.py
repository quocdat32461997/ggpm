import torch
import torch.nn as nn

class PropertyVAE(nn.Module):
    def __init__(self, args):
        # Args:
        #   - hidden_size: int or list of int.
        #   - num_prop: int
        #       By default num_prop = 2 that specifies number of properties to be predicted
        #   - dropout: float
        #       Dropout value
        super(PropertyVAE, self).__init__()

        # property regressor
        self.property_regressor = PropertyRegressor(args.num_property, args.property_hiddeen_size, args.dropout)

        # initialize encoder and decoder
        self.encoder = None
        self.decoder = None
    def forward(self, inputs):
        return None

class PropertyRegressor(nn.Module):
    def __init__(self, num_property, hidden_size, dropout:
        # Args
        #   - num_property: int
        #       Number of properties to predict
        #   - hidden_size: int or list
        #       Hidden sizes of hidden linear layers
        #   - dropout: float
        #       Dropout rate
        super(PropertyRegressor, self).__init__()

        # convert hidden_size to list
        assert isinstance(hidden_size, int) or isinstance(hidden_size, int)
        if isinstance(hidden_size, int): hidden_size = [hidden_size]

        # add hidden layers
        self.regressor = nn.ModuleList()
        for idx in range(1, len(hidden_size)):
            self.regressor.extend([
                nn.Linear(hidden_size[idx - 1], hidden_size[idx]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # add the final linear layer for regression
        self.regressor.append(
            nn.Linear(hidden_size[-1], num_property))

    def forward(self, inputs):
        return self.regressor(inputs)