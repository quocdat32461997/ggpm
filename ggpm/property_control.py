import torch

from torch import nn


class PropertyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, latent_lr, optimize_type='fixed'):
        super(PropertyOptimizer, self).__init__()
        self.input_size = input_size
        self.latent_lr = latent_lr
        self.optimize_type = optimize_type

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

    def optimize(self, root_vecs, targets):
        # Input:
        #   - root_vecs: torch.Tensor of root_vecs
        #   - targets: tuples of HOMO and LUMO targets

        func_dict = {'fixed': self.hard_optimize,
                     'patience': self.patience_optimize,
                     'soft': self.soft_optimize}

        # get property-optimizer
        if self.optimize_type not in func_dict:
            raise ValueError("Error: property-optimizing choice \"{}\" is not valid".format(self.optimize_type))
        func = func_dict[self.optimize_type]

        # optimize HOMOs & LUMOs
        with torch.enable_grad():  # enable gradient calculation in the test mode
            root_vecs, outputs = func(homo_vecs=root_vecs[:, :self.input_size],
                                      lumo_vecs=root_vecs[:, self.input_size:],
                                      targets=targets)
        return root_vecs, outputs[0:2], outputs[2:4]

    def soft_optimize(self, homo_vecs, lumo_vecs, targets):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, targets[0], targets[1]):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, self.patience
            while patience > 0:
                # predict HOMOs and LUMOs
                h_loss, l_loss, h_out, l_out = self.forward(h_vec, l_vec, (h_tar, l_tar))
                total_loss = h_loss + l_loss

                # break if sum of loss <= delta
                if total_loss <= self.property_delta:
                    break

                # enable grads for non-leaf tensors
                h_vec.retain_grad()
                l_vec.retain_grad()

                # compute gradients
                h_loss.backward()
                l_loss.backward()

                # update gradients if total loss larger than delta
                h_vec = property_grad_optimize(h_vec, h_out, h_tar)
                l_vec = property_grad_optimize(l_vec, l_out, l_tar)

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

        # final prediction
        return torch.cat([h_vecs, l_vecs], dim=-1), self.forward(h_vecs, l_vecs, targets)

    def patience_optimize(self, homo_vecs, lumo_vecs, targets):
        # Function to optimize homo_vecs and lumo_vecs until the delta difference is hit

        h_vecs, l_vecs = [], []
        for h_vec, l_vec, h_tar, l_tar in zip(homo_vecs, lumo_vecs, targets[0], targets[1]):
            # enable gradient tracking
            h_vec.requires_grad = True
            l_vec.requires_grad = True

            # loop until patience hits 0 or total_loss less than delta to avoid infinite loop
            prev_loss, patience = 0, self.patience
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
                h_vec = property_grad_optimize(h_vec, h_out, h_tar)
                l_vec = property_grad_optimize(l_vec, l_out, l_tar)

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

        # final prediction
        return torch.cat([h_vecs, l_vecs], dim=-1), self.forward(h_vecs, l_vecs, targets)

    def hard_optimize(self, homo_vecs, lumo_vecs, targets):
        # Function to optimize homo_vecs and lumo_vecs in a fixed number of loops

        # enable gradient tracking
        homo_vecs.requires_grad = True
        lumo_vecs.requires_grad = True

        for _ in range(self.property_optim_step):
            # predict HOMOs and LUMOs
            homo_loss, lumo_loss, homo_outputs, lumo_outputs = self.forward(homo_vecs, lumo_vecs, targets)

            # enable grads for non-leaf tensors
            homo_vecs.retain_grad()
            lumo_vecs.retain_grad()

            # compute gradients
            homo_loss.backward()
            lumo_loss.backward()

            # update latent vectors
            homo_vecs = property_grad_optimize(homo_vecs, homo_outputs, targets[0])
            lumo_vecs = property_grad_optimize(lumo_vecs, lumo_outputs, targets[1])

        # final prediction
        return torch.cat([homo_vecs, lumo_vecs], dim=-1), self.forward(homo_vecs, lumo_vecs, targets)

    def property_grad_optiimize(self, vecs, outputs, targets):


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
