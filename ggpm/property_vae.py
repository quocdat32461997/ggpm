import torch
import numpy as np
from icecream import ic

from ggpm.encoder import MotifEncoder
from ggpm.decoder import MotifDecoder
from ggpm.nnutils import *


def make_tensor(x):
    if not isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        x = torch.tensor(x)

    return to_cuda(x)


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
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
        self.encoder.tie_embedding(self.decoder.hmpn)

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

        # add gaussian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # decode
        loss, wacc, iacc, tacc, sacc = self.decoder(mols, (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        loss += beta * kl_div
        return loss, {'Loss': loss.item(), 'KL:': kl_div.item(),
                'Word': wacc, 'I-Word': iacc, 'Topo': tacc, 'Assm': sacc}


class PropOptVAE(nn.Module):
    def __init__(self, args):
        super(PropOptVAE, self).__init__()
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.latent_size, args.diterT, args.diterG, args.dropout)

        # tie embedding
        self.encoder.tie_embedding(self.decoder.hmpn)

        # define property optimizer
        self.property_hidden_size = args.latent_size // 2
        self.property_optim = PropertyOptimizer(input_size=self.property_hidden_size,
                                                hidden_size=args.linear_hidden_size,
                                                dropout=args.dropout, latent_lr=args.latent_lr)
        # self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size
        self.property_optim_step = args.property_optim_step

        # gaussian noise
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

    def reconstruct(self, batch, args):
        mols, graphs, tensors, _, homos, lumos = batch
        tree_tensors, _ = tensors = make_cuda(tensors)
        homos, lumos = make_tensor(homos), make_tensor(lumos)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add gaussian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)

        # find new latent vector that optimize HOMO & LUMO properties
        root_vecs, _, property_outputs = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos),
                                                                      args=args)

        # extract property outputs
        return property_outputs, self.decoder.decode(mols, (root_vecs, root_vecs, root_vecs),
                                                     greedy=True, max_decode_step=150)

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        tree_tensors, _ = tensors = make_cuda(tensors)
        homos, lumos = make_tensor(homos), make_tensor(lumos)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # sampling latent vectors
        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # predict HOMO & LUMO
        homo_loss, lumo_loss, _, _ = self.property_optim(homo_vecs=root_vecs[:, :self.property_hidden_size],
                                                         lumo_vecs=root_vecs[:, self.property_hidden_size:],
                                                         targets=(homos, lumos))

        # decode
        loss, wacc, iacc, tacc, sacc = self.decoder(mols, (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        # sum-up loss
        loss += beta * kl_div
        total_loss = loss + homo_loss + lumo_loss
        return total_loss, {'Loss': total_loss.item(), 'KL': kl_div.item(), 'Recs_Loss': loss.item(),
                'HOMO_MSE': homo_loss, 'LUMO_MSE': lumo_loss, 'Word': wacc, 'I-Word': iacc, 'Topo': tacc, 'Assm': sacc}


class PropertyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, latent_lr):
        super(PropertyOptimizer, self).__init__()
        self.input_size = input_size
        self.latent_lr = latent_lr

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

    def compute_loss(self, outputs, targets):
        # get last dim of outputs of loss-computing since each property has only 1 score
        return torch.nn.MSELoss()(outputs, targets)

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
        flatten = lambda x: x[:, -1] if len(x.shape) == 2 and x.shape[-1] == 1 else x

        # make predictions
        for hl, ll in zip(self.homo_linear, self.lumo_linear):
            homo_vecs = hl(homo_vecs)
            lumo_vecs = ll(lumo_vecs)

        # reshape to linear if last-dim == 1
        return flatten(homo_vecs), flatten(lumo_vecs)

    def optimize(self, root_vecs, targets, args):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets
        #   - optimize_type: str, type of optimizer (i.e. fixed, delta)

        # get property-optimizer
        if args.optimize_type == 'fixed':
            func = self.hard_optimize
        elif args.optimize_type == 'delta':
            func = self.soft_optimize
        else:
            raise ValueError("Error: property-optimizing choice \"{}\" is not valid".format(optimize_type))

        # optimize HOMOs & LUMOs
        with torch.enable_grad():  # enable gradient calculation in the test mode
            root_vecs, outputs = func(homo_vecs=root_vecs[:, :self.input_size],
                                      lumo_vecs=root_vecs[:, self.input_size:],
                                      targets=targets, args=args)
        return root_vecs, outputs[0:2], outputs[2:4]

    def soft_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        # cast delta and patience threshold to torch.float
        delta = torch.tensor(args.property_delta, dtype=torch.float)
        patience_threshold = torch.tensor(args.patience_threshold, dtype=torch.float)

        for h_vec, l_vec, h_tar, l_tar, i in zip(homo_vecs, lumo_vecs, targets[0],
                                                 targets[1], range(homo_vecs.shape[0])):
            prev_loss, patience = 0, args.patience
            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            while True and patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # break if sum of loss <= delta
                if total_loss <= delta: break

                # update gradients if total loss larger than delta
                h_vec = gradient_update(h_vec, h_loss, h_out, h_tar, self.latent_lr)
                l_vec = gradient_update(l_vec, l_loss, l_out, l_tar, self.latent_lr)

                # update patience to stop if loss change smaller than patience threshold
                if abs(total_loss - prev_loss) <= patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = args.patience
                # update prev_loss
                prev_loss = total_loss

            # assign vecs back
            homo_vecs[i] = h_vec
            lumo_vecs[i] = l_vec

        # final prediction
        return torch.cat([homo_vecs, lumo_vecs], dim=-1), self.forward(homo_vecs, lumo_vecs, targets)

    def patience_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        # cast patience threshold to torch.float
        patience_threshold = torch.tensor(args.patience_threshold, dtype=torch.float)

        for h_vec, l_vec, h_tar, l_tar, i in zip(homo_vecs, lumo_vecs, targets[0], targets[1],
                                                 range(homo_vecs.shape[0])):
            prev_loss, patience = 0, args.patience
            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            while True and patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # update gradients if total loss larger than delta
                h_vec = gradient_update(h_vec, h_loss, h_out, h_tar, self.latent_lr)
                l_vec = gradient_update(l_vec, l_loss, l_out, l_tar, self.latent_lr)

                # update patience to stop if loss change smaller than patience threshold
                if abs(total_loss - prev_loss) <= patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = args.patience

                # update prev_loss
                prev_loss = total_loss

            # assign vecs back
            homo_vecs[i] = h_vec
            lumo_vecs[i] = l_vec

        # final prediction
        return torch.cat([homo_vecs, lumo_vecs], dim=-1), self.forward(homo_vecs, lumo_vecs, targets)

    def hard_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

        # enable gradients
        homo_vecs.requires_grad = True
        lumo_vecs.requires_grad = True

        for _ in range(args.property_optim_step):
            # predict HOMOs and LUMOs
            homo_loss, lumo_loss, homo_outputs, lumo_outputs = self.forward(homo_vecs, lumo_vecs, targets)

            # enable grads for non-leaf tensors
            homo_vecs.retain_grad()
            lumo_vecs.retain_grad()

            # compute gradients
            homo_loss.backward()
            lumo_loss.backward()

            # update latent vectors
            homo_vecs = property_grad_optimize(homo_vecs, homo_outputs, targets[0], self.latent_lr)
            lumo_vecs = property_grad_optimize(lumo_vecs, lumo_outputs, targets[1], self.latent_lr)

        # final prediction
        return torch.cat([homo_vecs, lumo_vecs], dim=-1), self.forward(homo_vecs, lumo_vecs, targets)
