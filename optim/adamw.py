from .register import Optimizers
from torch.optim import AdamW as adamw
import torch


@Optimizers.register_module
class AdamW(object):
    def __init__(self, parameters_group, args, betas=(0.9, 0.999), eps=1e-6):
        self.optimizer = adamw(
            parameters_group,
            args.lr,
            weight_decay=args.weight_decay,
            betas=betas,
            eps=eps
        )

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
