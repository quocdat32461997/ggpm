import torch

from torch import nn
from ggpm.property_vae import PropOptVAE, make_cuda, make_tensor


def switch_gradients(loss, outputs, targets):
    """Switch to gradient descent/ascent."""

    return (outputs < targets).int() * -1 + 1


class PropertyVAEOptimizer(nn.Module):
    def __init__(self, model, args):
        super(PropertyVAEOptimizer, self).__init__()

        self.model = model
        self.property_optim_step = args.property_optim_step
        self.patience = args.patience
        self.optimize_type = args.optimize_type
        self.property_delta = args.property_delta
        self.patience_threshold = args.patience_threshold
        self.lr = args.latent_lr
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
        homos = make_tensor(homos)
        lumos = make_tensor(lumos)

        # encode
        root_vecs, tree_vecs = self.model.encoder(tree_tensors)

        # add gaussian noise
        root_vecs, root_kl = self.model.rsample(root_vecs, perturb=False)

        # optimize HOMOs & LUMOs
        func = self._get_optimize_func()
        with torch.enable_grad():
            root_vecs = func(homo_vecs=root_vecs[:, :self.model.latent_size],
                             lumo_vecs=root_vecs[:, self.model.latent_size:],
                             homo_targets=homos, lumo_targets=lumos)

        # predict properties
        property_outputs = self.model.property_optim.predict(homo_vecs=root_vecs[:, :self.model.latent_size],
                                                             lumo_vecs=root_vecs[:, self.model.latent_size:])

        # decode
        reconstructions = self.model.decoder.decode(mols, tuple([root_vecs] * 3),
                                                    greedy=True, max_decode_step=150)
        return property_outputs, reconstructions

    def update_params(self, params, preds, targets):
        gradient_sign = (preds < targets).int() * -2 + 1
        if len(params.size()) == 2:
            gradient_sign = gradient_sign.unsqueeze(-1)
        return params.clone() - gradient_sign * self.lr * params.grad

    def soft_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, homo_targets, lumo_targets):
            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, self.patience
            while patience > 0:
                # Recreate tensors to retrieve gradients
                h_vec = h_vec.clone().detach().requires_grad_(True)
                l_vec = l_vec.clone().detach().requires_grad_(True)

                # predict HOMOs and LUMOs
                self.model.zero_grad()
                h_loss, l_loss, h_out, l_out = self.model.property_optim(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # break if sum of loss <= delta
                if total_loss <= self.property_delta:
                    break

                # update patience to stop if loss change smaller than patience threshold
                if total_loss > prev_loss or (abs(total_loss - prev_loss) / prev_loss) <= self.patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = self.patience
                prev_loss = total_loss

                # compute gradients
                total_loss.backward(retain_graph=True)

                # update h_vec and l_vec
                h_vec = self.update_params(h_vec, h_out, h_tar)
                l_vec = self.update_params(l_vec, l_out, l_tar)

            # assign vecs back
            h_vecs.append(h_vec)
            l_vecs.append(l_vec)

        # concat h_vecs and l_vecs
        homo_vecs = torch.stack(h_vecs, dim=0)
        lumo_vecs = torch.stack(l_vecs, dim=0)

        return torch.cat([homo_vecs, lumo_vecs], dim=-1)

    def patience_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs until no patience

        h_vecs, l_vecs = [], []
        idx = 0
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, homo_targets, lumo_targets):
            idx += 1
            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, self.patience
            while patience > 0:
                # Recreate tensors to retrieve gradients
                h_vec = h_vec.clone().detach().requires_grad_(True)
                l_vec = l_vec.clone().detach().requires_grad_(True)

                # predict HOMOs and LUMOs
                self.model.zero_grad()
                h_loss, l_loss, h_out, l_out = self.model.property_optim(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # update patience to stop if loss change smaller than patience threshold
                if total_loss > prev_loss or (abs(total_loss - prev_loss) / prev_loss) <= self.patience_threshold:
                    patience -= 1
                else:  # reset patience
                    patience = self.patience
                prev_loss = total_loss

                # compute gradients
                total_loss.backward(retain_graph=True)

                # update h_vec and l_vec
                h_vec = self.update_params(h_vec, h_out, h_tar)
                l_vec = self.update_params(l_vec, l_out, l_tar)

            # append vectors
            h_vecs.append(h_vec)
            l_vecs.append(l_vec)

        # concat h_vecs and l_vecs
        homo_vecs = torch.stack(h_vecs, dim=0)
        lumo_vecs = torch.stack(l_vecs, dim=0)

        return torch.cat([homo_vecs, lumo_vecs], dim=-1)

    def hard_optimize(self, homo_vecs, lumo_vecs, homo_targets, lumo_targets):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

        for _ in range(self.property_optim_step):
            # Recreate tensors to retrieve gradients
            homo_vecs = homo_vecs.clone().detach().requires_grad_(True)
            lumo_vecs = lumo_vecs.clone().detach().requires_grad_(True)

            # predict HOMOs and LUMOs
            self.model.zero_grad()
            homo_loss, lumo_loss, homo_outputs, lumo_outputs = self.model.property_optim(homo_vecs, lumo_vecs,
                                                                                         (homo_targets, lumo_targets))
            total_loss = homo_loss + lumo_loss

            # compute gradients
            total_loss.backward()

            # update h_vec and l_vec
            homo_vecs = self.update_params(homo_vecs, homo_outputs, homo_targets)
            lumo_vecs = self.update_params(lumo_vecs, lumo_outputs, lumo_targets)

        return torch.cat([homo_vecs, lumo_vecs], dim=-1)

class HierPropertyVAEOptimizer(PropertyVAEOptimizer):
    def __init__(self, model, args):
        super(HierPropertyVAEOptimizer, self).__init__(model, args)

    def forward(self, batch, args):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets

        mols, _, tensors, _, homos, lumos = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        homos = make_tensor(homos)
        lumos = make_tensor(lumos)

        # encode
        root_vecs, _, _, _ = self.model.encoder(tree_tensors, graph_tensors)

        # add gaussian noise
        root_vecs, _ = self.model.rsample(root_vecs, perturb=False)

        # optimize HOMOs & LUMOs
        func = self._get_optimize_func()
        with torch.enable_grad():
            root_vecs = func(homo_vecs=root_vecs[:, :self.model.latent_size],
                             lumo_vecs=root_vecs[:, self.model.latent_size:],
                             homo_targets=homos, lumo_targets=lumos)

        # predict properties
        property_outputs = self.model.property_optim.predict(homo_vecs=root_vecs[:, :self.model.latent_size],
                                                             lumo_vecs=root_vecs[:, self.model.latent_size:])

        # decode
        reconstructions = self.model.decoder.decode(mols, tuple([root_vecs] * 3),
                                                    greedy=True, max_decode_step=150)
        return property_outputs, reconstructions
    