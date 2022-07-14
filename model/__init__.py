from .vlmodel import VLModels
from .vmodel import VModels



def create_model(args):
    model = VLModels.get(args.vlmodel)(args)
    return model


def create_distill_model(args):
    model = VLModels.get(args.vlmodel)(args)
    return model


def create_distill_quantization_model(args):
    model = VLModels.get(args.vlmodel)(args, quantization=True)
    return model