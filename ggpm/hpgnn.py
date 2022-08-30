import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from ggpm.mol_graph import MolGraph
from ggpm.encoder import HierMPNEncoder
from ggpm.decoder import HierMPNDecoder
from ggpm.nnutils import *


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [to_cuda(make_tensor(x)).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [to_cuda(make_tensor(x)).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

class HierVAE(nn.Module):
    def __init__(self, args):
        super(HierVAE, self).__init__()
        self.encoder = HierMPNEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.depthT, args.depthG, args.dropout)
        self.decoder = HierMPNDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.latent_size, args.diterT, args.diterG, args.dropout)
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size

        self.R_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = nn.Linear(args.hidden_size, args.latent_size)

        # self.T_mean = nn.Linear(args.hidden_size, args.latent_size)
        # self.T_var = nn.Linear(args.hidden_size, args.latent_size)

        # self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        # self.G_var = nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = to_cuda(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size):
        root_vecs = to_cuda(torch.randn(batch_size, self.latent_size))
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def reconstruct(self, batch):
        mols, graphs, tensors, _, _, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        # graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        # size = graph_vecs.new_tensor([le for _,le in graph_tensors[-1]])
        # graph_vecs = graph_vecs.sum(dim=1) / size.unsqueeze(-1)

        # tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        # size = tree_vecs.new_tensor([le for _,le in tree_tensors[-1]])
        # tree_vecs = tree_vecs.sum(dim=1) / size.unsqueeze(-1)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        # tree_vecs, tree_kl = self.rsample(tree_vecs, self.T_mean, self.T_var, perturb_z)
        # graph_vecs, graph_kl = self.rsample(graph_vecs, self.G_mean, self.G_var, perturb_z)
        kl_div = root_kl  # + tree_kl + graph_kl

        # modify molecules
        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc