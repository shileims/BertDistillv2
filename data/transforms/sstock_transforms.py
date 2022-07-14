from .register import Transforms
from timm.data import create_transform

@Transforms.register_module
class SStockTransforms(object):
    def __init__(self, args, is_train=True):
        self.trans = create_transform(
                input_size=args.input_size,
                is_training=is_train,
                hflip=0.5,
                color_jitter=args.color_jitter if args.color_jitter > 0 else None,
                auto_augment=args.aa if args.aa != 'none' else None,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        self.trans.transforms = [lambda image: image.convert("RGB"), *self.trans.transforms]

    def __call__(self, input):
        return self.trans(input)