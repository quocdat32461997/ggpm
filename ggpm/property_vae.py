import numpy as np

from ggpm.encoder import MotifEncoder
from ggpm.decoder import MotifDecoder, MotifSchedulingDecoder
from ggpm.property_optimizer import PropertyOptimizer
from ggpm.loss_weigh import *
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
        if args.tie_embedding:
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

    def reconstruct(self, batch, args):
        mols, graphs, tensors, _, homos, lumos = batch
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
                      'Word': wacc, 'I-Word': iacc, 'Topo': tacc, 'Assm': sacc}, False


class PropOptVAE(torch.nn.Module):
    def __init__(self, args):
        super(PropOptVAE, self).__init__()
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.depthT, args.depthG, args.dropout)
        self.decoder = MotifDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.latent_size, args.diterT, args.diterG, args.dropout)

        # tie embedding
        self.encoder.tie_embedding(self.decoder.hmpn)

        self.latent_size = args.latent_size // 2  # define property optimizer
        self.property_optim = PropertyOptimizer(input_size=self.latent_size,
                                                hidden_size=args.linear_hidden_size,
                                                dropout=args.dropout)

        if args.tie_embedding:
            self.encoder.tie_embedding(self.decoder.hmpn)
        self.property_optim_step = args.property_optim_step

        # gaussian noise
        self.R_mean = torch.nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = torch.nn.Linear(args.hidden_size, args.latent_size)

        # setup loss-scaling
        self.loss_scaling = False
        try:
            if args.loss_scaling:
                self.loss_scaling = True
                self.loss_weigh = LossWeigh()
        except:
            pass

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
        tree_tensors, _ = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add gaussian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)

        latent_vecs = root_vecs.clone()  # torch.cat([root_vecs, root_vecs.clone()], dim=-1)# find new latent vector that optimize HOMO & LUMO properties
        # root_vecs, _, property_outputs = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos),
        #                                                              args=args)
        property_outputs = self.property_optim.predict(homo_vecs=latent_vecs[:, :self.latent_size],
                                                       lumo_vecs=latent_vecs[:, self.latent_size:])
        # extract property outputs
        return property_outputs, self.decoder.decode(mols, tuple([root_vecs] * 3),
                                                     greedy=True, max_decode_step=150)

    def optimize_recs(self, batch, args):
        mols, graphs, tensors, _, homos, lumos = batch
        tree_tensors, _ = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add gaussian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)

        # optimize properties
        root_vecs, homo_prop, lumo_prop = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos))

        return [homo_prop, lumo_prop], self.decoder.decode(mols, (root_vecs, root_vecs, root_vecs),
                                                           greedy=True, max_decode_step=150)

    def clip_negative_loss(self, loss):
        if loss > 0:
            return False, loss
        else:
            return True, loss * 0 + torch.normal(mean=0.5, std=0.5, size=loss.size(), dtype=loss.dtype,
                                                 device=loss.device)

    def forward(self, mols, graphs, tensors, orders, homos, lumos, beta, perturb_z=True):
        tree_tensors, _ = tensors = make_cuda(tensors)
        homos, lumos = make_tensor(homos), make_tensor(lumos)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # sampling latent vectors
        root_vecs, root_kl = self.rsample(root_vecs, perturb_z)
        kl_div = root_kl

        # predict HOMO & LUMO
        latent_vecs = root_vecs.clone()  # torch.cat([root_vecs, root_vecs.clone()], dim=-1)
        homo_loss, lumo_loss, _, _ = self.property_optim(homo_vecs=latent_vecs[:, :self.latent_size],
                                                         lumo_vecs=latent_vecs[:, self.latent_size:],
                                                         targets=(homos, lumos))

        # decode
        loss, wacc, iacc, tacc, sacc = self.decoder(mols, (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

        # sum-up loss
        loss += beta * kl_div

        # since loss-scaling may lead to negative loss
        # hence, separate each loss term and clip individually
        if self.loss_scaling:
            loss = self.loss_weigh.compute_recon_loss(loss)
            homo_loss, lumo_loss = self.loss_weigh.compute_prop_loss(homo_loss, lumo_loss)

        total_loss = loss + homo_loss + lumo_loss
        loss_clipped, total_loss = self.clip_negative_loss(total_loss)

class PropOptSchedulingVAE(nn.Module):
    def __init__(self, args):
        super(PropOptSchedulingVAE, self).__init__()
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.depthT, args.depthG, args.dropout)
        self.decoder = MotifSchedulingDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
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
        #root_vecs, _, property_outputs = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos),
        #                                                              args=args)
        property_outputs = self.property_optim.predict(homo_vecs=root_vecs[:, :self.property_hidden_size],
                                                                     lumo_vecs=root_vecs[:, self.property_hidden_size:])

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

        # scale up homo & lumo loss by 20
        total_loss = loss + 20 * homo_loss + 20 * lumo_loss
        return total_loss, {'Loss': total_loss.item(), 'KL': kl_div.item(), 'Recs_Loss': loss.item(),
                'HOMO_MSE': homo_loss, 'LUMO_MSE': lumo_loss, 'Word': wacc, 'I-Word': iacc, 'Topo': tacc, 'Assm': sacc}


        return total_loss, {'Loss': total_loss.item(), 'KL': kl_div.item(), 'Recs_Loss': loss.item(),
                            'HOMO_MSE': homo_loss.item(), 'LUMO_MSE': lumo_loss.item(), 'Word': wacc, 'I-Word': iacc,
                            'Topo': tacc, 'Assm': sacc}, \
               True if loss_clipped else False


class PropOptSchedulingVAE(nn.Module):
    def __init__(self, args):
        super(PropOptSchedulingVAE, self).__init__()
        self.encoder = MotifEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                    args.depthT, args.depthG, args.dropout)
        self.decoder = MotifSchedulingDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size,
                                              args.hidden_size,
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
        # root_vecs, _, property_outputs = self.property_optim.optimize(root_vecs=root_vecs, targets=(homos, lumos),
        #                                                              args=args)
        property_outputs = self.property_optim.predict(homo_vecs=root_vecs[:, :self.property_hidden_size],
                                                       lumo_vecs=root_vecs[:, self.property_hidden_size:])

    def optimize(self, root_vecs, targets, args):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets
        #   - optimize_type: str, type of optimizer (i.e. fixed, delta)

        func_dict = {'fixed': self.hard_optimize,
                     'patience': self.patience_optimize,
                     'soft': self.soft_optimize}

        # get property-optimizer
        if args.optimize_type not in func_dict:
            raise ValueError("Error: property-optimizing choice \"{}\" is not valid".format(args.optimize_type))
        func = func_dict[args.optimize_type]

        # optimize HOMOs & LUMOs
        with torch.enable_grad():  # enable gradient calculation in the test mode
            root_vecs, outputs = func(homo_vecs=root_vecs[:, :self.input_size],
                                      lumo_vecs=root_vecs[:, self.input_size:],
                                      targets=targets, args=args)
        return root_vecs, outputs[0:2], outputs[2:4]

    def soft_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, targets[0], targets[1]):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, args.patience
            while patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # break if sum of loss <= delta
                if total_loss <= args.property_delta:
                    break

                # enable grads for non-leaf tensors
                h_vec.retain_grad()
                l_vec.retain_grad()

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # update gradients if total loss larger than delta
                h_vec = property_grad_optimize(h_vec, h_out, h_tar, self.latent_lr)
                l_vec = property_grad_optimize(l_vec, l_out, l_tar, self.latent_lr)

                # update patience to stop if loss change smaller than patience threshold
                if (abs(total_loss - prev_loss) / prev_loss) <= args.patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = args.patience

                # update prev_loss
                prev_loss = total_loss

            # assign vecs back
            h_vecs.append(h_vec)
            l_vecs.append(l_vec)

            # delete vecs and targets
            del h_vec, l_vec, h_tar, l_tar

        # free memory for vectors
        del homo_vecs, lumo_vecs

        # concat h_vecs and l_vecs
        h_vecs = torch.stack(h_vecs, dim=0)
        l_vecs = torch.stack(l_vecs, dim=0)

        # final prediction
        return torch.cat([h_vecs, l_vecs], dim=-1), self.forward(h_vecs, l_vecs, targets)

    def patience_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, targets[0], targets[1]):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, args.patience
            while patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # enable grads for non-leaf tensors
                h_vec.retain_grad()
                l_vec.retain_grad()

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # update gradients if total loss larger than delta
                h_vec = property_grad_optimize(h_vec, h_out, h_tar, self.latent_lr)
                l_vec = property_grad_optimize(l_vec, l_out, l_tar, self.latent_lr)

                # update patience to stop if loss change smaller than patience threshold
                if (abs(total_loss - prev_loss) / prev_loss) <= args.patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = args.patience

                # update prev_loss
                prev_loss = total_loss

            # append vectors
            h_vecs.append(h_vec)
            l_vecs.append(l_vec)

            # delete vecs and targets
            del h_vec, l_vec, h_tar, l_tar

        # free memory for vectors
        del homo_vecs, lumo_vecs

        # concat h_vecs and l_vecs
        h_vecs = torch.stack(h_vecs, dim=0)
        l_vecs = torch.stack(l_vecs, dim=0)

        # final prediction
        return torch.cat([h_vecs, l_vecs], dim=-1), self.forward(h_vecs, l_vecs, targets)

    def hard_optimize(self, homo_vecs, lumo_vecs, targets, args):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

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

        # scale up homo & lumo loss by 20
        total_loss = loss + 20 * homo_loss + 20 * lumo_loss
        return total_loss, {'Loss': total_loss.item(), 'KL': kl_div.item(), 'Recs_Loss': loss.item(),
                            'HOMO_MSE': homo_loss, 'LUMO_MSE': lumo_loss, 'Word': wacc, 'I-Word': iacc, 'Topo': tacc,
                            'Assm': sacc}