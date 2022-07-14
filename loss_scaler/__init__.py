from .register import LossScalers
from .nativescaler import NativeScaler



def create_loss_scaler(args):
    loss_Scaler = LossScalers.get(args.loss_scaler)()
    return loss_Scaler
