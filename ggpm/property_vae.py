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
        #   - num_prop: intdqn1700
        #       By default num_prop = 2 that specifies number of properties to be predicted
        #   - dropout: float
        #       Dropout value
        super(PropertyVAE, self).__init__()
        self.latent_size = args.latent_size

        # property regressor
        # self.property_regressor = PropertyRegressor(args.num_property, args.hiddeen_size, args.dropout)

        # initialize encoder and decoder
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.latent_size, args.diterT, args.diterG, args.dropout)

        # tie embedding
        # self.encoder.tie_embedding(self.decoder.hmpn)

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

        self.property_hidden_size = args.latent_size // 2
        self.property_optim = PropertyOptimizer(input_size=self.property_hidden_size,
                                                hidden_size=args.linear_hidden_size,
                                                dropout=args.dropout, latent_lr=args.latent_lr)
        # self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size
        self.property_optim_step = args.property_optim_step

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

    def reconstruct(self, batch, **kwargs):
        mols, graphs, tensors, _, homos, lumos = batch
        tree_tensors, _ = tensors = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # sample latent vectors
        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)

        # find new latent vector that optimize HOMO & LUMO properties
        property_outputs, root_vecs = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos),
                                                                   **kwargs)

        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        tree_tensors, _ = tensors = make_cuda(tensors)

        root_vecs, tree_vecs = self.encoder(tree_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # HOMO & LUMO predictors
        homo_loss, lumo_loss, _, _ = self.property_optim(homo_vecs=root_vecs[:, :self.property_hidden_size],
                                                         lumo_vecs=root_vecs[:, self.property_hidden_size:],
                                                         labels=(homos, lumos))

        # modify molecules
        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        # sum-up loss
        loss += beta * kl_div
        total_loss = loss + homo_loss + lumo_loss
        return {'loss': total_loss, 'recs_loss': loss, 'kl_div': kl_div.items(), 'homo_loss': homo_loss,
                'lumo_loss': lumo_loss,
                'wacc': wacc, 'iacc': iacc, 'tacc': tacc, 'sacc': sacc}


class PropertyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, latent_lr):
        super(PropertyOptimizer, self).__init__()
        self.input_size = input_size
        self.latent_r = latent_lr

        # hidden_size to list
        hidden_size = [hidden_size] if isinstance(hidden_size, int) else hidden_size
        hidden_size = [input_size] + hidden_size

        # define homo and lumo linear head
        self.homo_linear, self.lumo_linear = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(hidden_size) - 1):
            self.homo_linear.extend([nn.Linear(hidden_size[idx], hidden_size[idx + 1]),
                                     nn.ReLU(), nn.Dropout(dropout)])
            self.lumo_linear.extend([nn.Linear(hidden_size[idx], hidden_size[idx + 1]),
                                     nn.ReLU(), nn.Dropout(dropout)])
        self.homo_linear.append(nn.Linear(hidden_size[-1], 1))
        self.lumo_linear.append(nn.Linear(hidden_size[-1], 1))

    def compute_loss(self, outputs, labels):
        # get last dim of outputs of loss-computing since each property has only 1 score
        return torch.nn.MSELoss()(outputs[:, -1], torch.tensor(labels.tolist(), dtype=torch.float))

    def forward(self, homo_vecs, lumo_vecs, targets):
        # Input:
        #   - homo_vecs: torch.Tensor
        #   - lumo_vecs: torch.Tensor
        #   - targets: tuple of HOMO and LUMO targets

        # make predictions
        homo_outputs, lumo_outputs = self.predict(homo_vecs, lumo_vecs)

        # compute loss
        homo_loss = self.compute_loss(homo_outputs, targets[0])
        lumo_loss = self.compute_loss(lumo_outputs, targeets[1])

        return homo_loss, lumo_loss, homo_outputs, lumo_outputs

    def predict(self, homo_vecs, lumo_vecs):
        # make predictions
        for hl, ll in zip(self.homo_linear, self.lumo_linear):
            homo_vecs = hl(homo_vecs)
            lumo_vecs = ll(lumo_vecs)

        return homo_vecs, lumo_vecs

    def optimize(self, root_vecs, targets, **kwargs):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets
        #   - type: str, type of optimizer (i.e. fixed, delta)

        # get property-optimizer
        type = PropertyOptimizer.TYPES[kwargs['type']] is 'fixed':
        if type is 'fixed':
            func = self.hard_optimize
        elif type is 'delta':
            func = self.soft_optimize
        else:
            raise "Error: property-optimizing choice \"{}\"is not valid".format(type)

        # optimize HOMOs & LUMOs
        with torch.enable_grad():  # enable gradient calculation in the test mode
            return func(homo_vecs=root_vecs[:, :self.input_size], lumo_vecs=root_vecs[:, self.input_size:],
                        targets=targets, **kwargs)

    def soft_optimize(self, homo_vecs, lumo_vecs, targets, **kwargs):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        # cast delta and patience threshold to torch.float
        delta = torch.tensor(kwargs['property_delta'], dtype=torch.float)
        patience_threshold = torch.tensor(kwargs['patience_threshold'], dtype=torch.float)

        for h_vec, l_vec, h_tar, l_tar, i in zip(homo_vecs, lumo_vecs, targets[0], targets[1],
                                                               range(homo_vecs.shape[0])):
            prev_loss, patience = 0, kwargs['patience']
            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            while True and patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # break if sum of loss <= delta
                total_loss = h_loss + l_loss
                if total_loss <= delta: break

                # update gradients if total loss larger than delta
                h_vec = gradient_update(h_vec, h_out, h_tar)
                l_vec = gradient_update(l_vec, l_out, l_tar)

                # update patience to stop if loss change smaller than patience threshold
                if abs(total_loss - prev_loss) <= patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = kwargs['patience']
                # update prev_loss
                prev_loss = total_loss

            # assign vecs back
            homo_vecs[i] = h_vec
            lumo_vecs[i] = l_vec

        # final prediction
        return self.forward(homo_vecs, lumo_vecs, targets)

    def hard_optimize(self, homo_vecs, lumo_vecs, targets, **kwargs):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

        def _update_latent(vecs, outs, targs):
            # Function to iterate through each sample and gradient
            # descent/ascent latent vectors accordingly

            return torch.stack([gradient_update(v, o, t) for v, o, t in zip(vecs, outs, targs)], dim=0)

        for _ in range(kwargs['property_optim_step']):
            # predict HOMOs and LUMOs
            homo_loss, lumo_loss, homo_outputs, lumo_outputs = self.forward(homo_vecs, lumo_vecs, targets)

            # compute gradients
            homo_loss.backward()
            lumo_loss.backward()

            # update latent vectors
            homo_vecs = _update_latent(homo_vecs, homo_outputs, targets[0])
            lumo_vecs = _update_latent(lumo_vecs, lumo_outputs, targets[1])

        # final prediction
        return self.forward(homo_vecs, lumo_vecs, targets)
