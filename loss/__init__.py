from .register import Losses
from .nce import NCE, DistributedNCE



def create_loss(args):
    loss = Losses.get(args.loss)(args.vl_label_smooth)
    return loss