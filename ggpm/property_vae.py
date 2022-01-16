import torch

from ggpm.encoder import MotifEncoder
from ggpm.decoder import MotifDecoder
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
        self.latent_size = args.latent_size

        # property regressor
        #self.property_regressor = PropertyRegressor(args.num_property, args.hiddeen_size, args.dropout)

        # initialize encoder and decoder
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.latent_size, args.diterT, args.diterG, args.dropout)

        # tie embedding
        #self.encoder.tie_embedding(self.decoder.hmpn)

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

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        # unzip tensors into tree_tensors
        tree_tensors, _ = tensors = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add guassian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # decode
        loss, wacc, iacc, tacc, sacc = self.decoder(mols, (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        return {'loss': loss + beta * kl_div, 'kl_div': kl_div.items(),
                'wacc': wacc, 'iacc': iacc, 'tacc': tacc, 'sacc': sacc}


class PropOptVAE(nn.Module):
    def __init__(self, args):
        super(PropOptVAE, self).__init__()
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                      args.latent_size, args.diterT, args.diterG, args.dropout)
        self.property_optim = PropertyOptimizer(input_size=args.latent_size // 2, hidden_size=args.linear_hidden_size,
                                                dropout=args.dropout)
        #self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size

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

    def sample(self, batch_size):
        root_vecs = to_cuda(torch.randn(batch_size, self.latent_size))
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def reconstruct(self, batch):
        mols, graphs, tensors, _, _, _ = batch
        tree_tensors, _ = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        tree_tensors, _ = tensors = make_cuda(tensors)

        root_vecs, tree_vecs = self.encoder(tree_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        # HOMO & LUMO predictors
        homo_loss, lumo_loss = self.property_optim(root_vecs, labels=(homos, lumos))

        # modify molecules
        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        # sum-up loss
        loss += beta * kl_div
        total_loss = loss + homo_loss + lumo_loss
        return {'loss': total_loss, 'recs_loss': loss, 'kl_div': kl_div.items(), 'homo_loss': homo_loss, 'lumo_loss': lumo_loss,
                'wacc': wacc, 'iacc': iacc, 'tacc': tacc, 'sacc': sacc}


class PropertyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(PropertyOptimizer, self).__init__()
        self.input_size = input_size
        # hidden_size to list
        hidden_size = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        hidden_size = [input_size] + hidden_size

        # define homo and lumo linear head
        self.homo_linear, self.lumo_linear = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(hidden_size)-1):
            self.homo_linear.extend([nn.Linear(hidden_size[idx], hidden_size[idx+1]),
                nn.ReLU(), nn.Dropout(dropout)])
            self.lumo_linear.extend([nn.Linear(hidden_size[idx], hidden_size[idx + 1]),
                nn.ReLU(), nn.Dropout(dropout)])
        self.homo_linear.append(nn.Linear(hidden_size[-1], 1))
        self.lumo_linear.append(nn.Linear(hidden_size[-1], 1))

    def compute_loss(self, outputs, labels):
        # get last dim of outputs of loss-computing since each property has only 1 score
        return torch.nn.MSELoss()(outputs[:, -1], torch.tensor(labels.tolist(), dtype=torch.float))

    def forward(self, features, labels):
        # extract labels
        homo_labels, lumo_labels = labels

        # make predictions
        homo_outputs, lumo_outputs = self.predict(features=features)

        # compute loss
        homo_loss = self.compute_loss(homo_outputs, homo_labels)
        lumo_loss = self.compute_loss(lumo_outputs, lumo_labels)

        return homo_loss, lumo_loss

    def predict(self, features):
        # extract features
        homo_outputs = features[:, :self.input_size]
        lumo_outputs = features[:, self.input_size:]

        # make predictions
        for hl, ll in zip(self.homo_linear, self.lumo_linear):
            homo_outputs = hl(homo_outputs)
            lumo_outputs = ll(lumo_outputs)

        return homo_outputs, lumo_outputs