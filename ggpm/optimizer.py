import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class MultipleOptimizer(object):
    def __init__(self, opts:list, decay_lr=False, **kwargs):
        self.optimizers = opts

        #  by default, use ExponentialLR
        if decay_lr:
            self.schedulers = [lr_scheduler.ExponentialLR(opt, kwargs['anneal_rate'])
                               for opt in self.optimizers]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def decay(self):
        for schl in self.schedulers:
            schl.step()

    def step(self):
        for op in self.optimizers:
            op.step()
