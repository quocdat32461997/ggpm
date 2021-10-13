import torch

from ggpm.encoder import MotifEncoder
from ggpm.decoder import MotifGenerator
from ggpm.nnutils import *


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: to_cuda(x if type(x) is torch.Tensor else torch.tensor(x))
    tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class PropertyVAE(torch.nn.Module):
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
        self.encoder = MotifEncoder(args.embedding_size, args.vocab, args.dropout)
        self.decoder = None

    def forward(self, tensors, orders, beta, perturb_z=True):
        # unzip tensors into tree_tensors
        tree_tensors, _ = tensors = make_cuda(tensors)

        # encode
        root_vecs, tree_veecs, _, _ = self.encoder(tree_tensors)

        return None


class PropertyRegressor(torch.nn.Module):
    def __init__(self, num_property, hidden_size, dropout):
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
        self.regressor = torch.nn.ModuleList()
        for idx in range(1, len(hidden_size)):
            self.regressor.extend([
                torch.nn.Linear(hidden_size[idx - 1], hidden_size[idx]),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])

        # add the final linear layer for regression
        self.regressor.append(
            torch.nn.Linear(hidden_size[-1], num_property))

    def forward(self, inputs):
        return self.regressor(inputs)
