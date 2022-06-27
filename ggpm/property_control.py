import torch

from torch import nn
from property_vae import *


def property_grad_optimize(self, loss, outputs, targets):
    """Performs gradient ascent by negating loss when outputs < targets.
    Otherwise, perform gradient descent"""

    # Negate if outputs < targets
    loss_mask = torch.tensor(outputs < targets, dtype=torch.int) * -1
    loss *= loss_mask

    return loss


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


class PropertyVAEOptimizer(PropOptVAE):
    def __init__(self, args, optimizer):
        super(PropertyOptimizer).__init__(args)

        self.property_optim_step = args.property_optim_step
        self.patience = args.patience
        self.property_delta = args.property_delta
        self.patience_threshold = args.patience_threshold
        self.optimizer = optimizer

        self.func_dict = {'fixed': self.hard_optimize,
                     'patience': self.patience_optimize,
                     'soft': self.soft_optimize}

    def _get_optimize_func(self):
        if self.optimize_type not in self.func_dict:
            raise ValueError("Error: property-optimizing choice \"{}\" is not valid".format(self.optimize_type))
        return self.func_dict[self.optimize_type]

    def forward(self, batch, args):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets

        mols, graphs, tensors, _, homos, lumos = batch
        tree_tensors, _ = make_cuda(tensors)

        # encode
        root_vecs, tree_vecs = self.encoder(tree_tensors)

        # add gaussian noise
        root_vecs, root_kl = self.rsample(root_vecs, perturb=False)

        # optimize HOMOs & LUMOs
        func = self._get_optimize_func()
        root_vecs = func(homo_vecs=root_vecs[:, :self.input_size],
                                  lumo_vecs=root_vecs[:, self.input_size:],
                                  homo_targets=homos,
                                  lumo_targets=lumos)

        # predict properties
        property_outputs = self.property_optim.predict(homo_vecs=root_vecs[:, :self.latent_size],
                                                       lumo_vecs=root_vecs[:, self.latent_size:])

        return property_outputs, self.decoder.decode(mols, tuple([root_vecs] * 3),
                                                     greedy=True, max_decode_step=1500)

    def soft_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, homo_targets, lumo_targets):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            with torch.enable_grad:
                # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
                prev_loss, patience = 0, self.patience
                while patience > 0:
                    # predict HOMOs and LUMOs
                    h_loss, l_loss, h_out, l_out = self.property_optim(h_vec, l_vec, (h_tar, l_tar))
                    total_loss = h_loss + l_loss

                    # break if sum of loss <= delta
                    if total_loss <= self.property_delta:
                        break

                    # enable grads for non-leaf tensors
                    h_vec.retain_grad()
                    l_vec.retain_grad()

                    # update gradients if total loss larger than delta
                    h_loss = property_grad_optimize(h_loss, h_out, h_tar)
                    l_loss = property_grad_optimize(l_loss, l_out, l_tar)

                    # compute gradients
                    h_loss.backward()
                    l_loss.backward()

                    # update patience to stop if loss change smaller than patience threshold
                    if (abs(total_loss - prev_loss) / prev_loss) <= self.patience_threshold:
                        patience -= 1
                    else:  # reset patience
                        patience = self.patience

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

        return torch.cat([h_vecs, l_vecs], dim=-1)

    def patience_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, homo_targets, lumo_targets):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            with torch.enable_grad():
                # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
                prev_loss, patience = 0, self.patience
                while patience > 0:
                    # predict HOMOs and LUMOs
                    h_loss, l_loss, h_out, l_out = self.proprety_optim(h_vec, l_vec, (h_tar, l_tar))
                    total_loss = h_loss + l_loss

                    # enable grads for non-leaf tensors
                    h_vec.retain_grad()
                    l_vec.retain_grad()

                    # update gradients if total loss larger than delta
                    h_loss = property_grad_optimize(h_loss, h_out, h_tar)
                    l_loss = property_grad_optimize(l_loss, l_out, l_tar)

                    # compute gradients
                    h_loss.backward()
                    l_loss.backward()

                    # update patience to stop if loss change smaller than patience threshold
                    if (abs(total_loss - prev_loss) / prev_loss) <= self.patience_threshold:
                        patience -= 1
                    else:  # reset patience
                        patience = self.patience

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

        return torch.cat([h_vecs, l_vecs], dim=-1)

    def hard_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

        # enable gradient tracking
        homo_vecs.requires_grad = True
        lumo_vecs.requires_grad = True

        with torch.enable_grad():
            for _ in range(self.property_optim_step):
                # predict HOMOs and LUMOs
                homo_loss, lumo_loss, homo_outputs, lumo_outputs = self.property_optim(homo_vecs, lumo_vecs,
                                                                                       (homo_targets, lumo_targets))

                # enable grads for non-leaf tensors
                homo_vecs.retain_grad()
                lumo_vecs.retain_grad()

                # update latent vectors
                homo_vecs = property_grad_optimize(homo_loss, homo_outputs, homo_targets)
                lumo_vecs = property_grad_optimize(lumo_loss, lumo_outputs, lumo_targets)

                # compute gradients
                homo_loss.backward()
                lumo_loss.backward()

        return torch.cat([h_vecs, l_vecs], dim=-1)


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
