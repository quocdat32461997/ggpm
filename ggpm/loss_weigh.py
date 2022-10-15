import torch
from torch import nn


class LossWeigh(torch.nn.Module):
    def __init__(self):
        super(LossWeigh, self).__init__()
        self.homo_log_var = nn.Parameter(torch.zeros((1,), dtype=torch.float64), requires_grad=True)
        self.lumo_log_var = nn.Parameter(torch.zeros((1,), dtype=torch.float64), requires_grad=True)
        self.recon_log_var = nn.Parameter(torch.zeros((1,), dtype=torch.float64), requires_grad=True)

    def compute_recon_loss(self, loss):
        loss = loss * torch.exp(-self.recon_log_var) + self.recon_log_var
        return loss.sum()

    def compute_prop_loss(self, homo_loss, lumo_loss):
        # homo
        homo_loss = homo_loss * torch.exp(-self.homo_log_var) + self.homo_log_var
        homo_loss /= 2

        # lumo
        lumo_loss = lumo_loss * torch.exp(-self.lumo_log_var) + self.lumo_log_var
        lumo_loss /= 2

        return homo_loss, lumo_loss