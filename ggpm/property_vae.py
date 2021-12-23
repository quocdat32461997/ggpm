import torch

from ggpm.encoder import MotifEncoder
from ggpm.decoder import MotifDecoder
from ggpm.nnutils import *


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: to_cuda(x if type(x) is torch.Tensor else torch.tensor(x))
    tree_tensors = [to_cuda(make_tensor(x)).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [to_cuda(make_tensor(x)).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
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
        self.latent_size = args.latent_size

        # property regressor
        #self.property_regressor = PropertyRegressor(args.num_property, args.hiddeen_size, args.dropout)

        # initialize encoder and decoder
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.latent_size, args.diterT, args.diterG, args.dropout)

        # tie embedding
        #self.encoder.tie_embedding(self.decoder.)

        # guassian noise
        self.R_mean = torch.nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = torch.nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = self.R_mean(z_vecs)
        z_log_var = -1 * torch.abs(self.R_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = to_cuda(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def reconstruct(self, batch):
        mols, graphs, tensors, _ = batch
        tree_tensors, _ = tensors = make_cuda(tensors)
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)
        return self.decoder.decode(mols, (root_vecs, root_vecs, root_vecs),
                                   greedy=True,
                                   max_decode_step=150)

    def forward(self, mols, graphs, tensors, orders, beta, perturb_z=True):
        # unzip tensors into tree_tensors
        tree_tensors, _ = tensors = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add guassian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # decode
        loss, wacc, iacc, tacc, sacc = self.decoder(mols, (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc


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
