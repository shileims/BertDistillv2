import torch
from .register import VModels
from torch import nn
from functools import partial
from model.basic_layers import LayerNormWithForceFP32
from model.swin_default import SwinTransformer
from timm.models.vision_transformer import VisionTransformer, _cfg
from utils import smart_partial_load_model_state_dict
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

default_cfgs = {
    # patch models
    'msvit_base_patch4_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

@VModels.register_module
class TinySwin(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.is_distill = args.is_distill
        self.kwargs = {
            'split_version': 'faster',
            'patch_norm': True,
            'no_weight_decay_keys': ["ln"],
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'mlp_ratio': 4,
            'split_size': 7,
            'reduce_channels': [2, 2, 2, None],
            'disturb': True,
            'reduce_norm': True,
            'reduce_act': False,
            'rpe': 'table',
            'dpe': False,
            'qkv_bias': True,
            'norm_layer': partial(LayerNormWithForceFP32, eps=1e-6),
            'drop_path_rate': args.drop_path}
        self.backbone = SwinTransformer.get(args.base_model)(**self.kwargs)
        self.backbone.default_cfg = default_cfgs['msvit_base_patch4_224']

        if not args.vmodel_from_scratch and not args.is_distill:
            print(f'Without distilling and Loading pretrain model from {args.vmodel_pretrain}')
            state_dict = torch.load(args.vmodel_pretrain, map_location='cpu')
            smart_partial_load_model_state_dict(self.backbone, state_dict, rename=False)

        self.projector = nn.Linear(self.backbone.num_features, args.proj_size, bias=False)

        if self.is_distill:
            args.vmodel_fix = True
            # if hasattr(self, 'projector'):
            #     self.projector.eval()
            #     for params in self.projector.parameters():
            #         params.requires_grad = False

            self.apply(self._init_weights)

            assert not args.vmodel_from_scratch, f'Distilling needs a pretrained teacher vision model'
            print(f'Distilling and Loading pretrain vision model from {args.vlmodel_pretrain}')
            state_dict = torch.load(args.vlmodel_pretrain, map_location='cpu')['vision']
            smart_partial_load_model_state_dict(self.backbone,
                                                {k.replace('backbone.', ''): v for k, v in state_dict.items() if
                                                 'backbone.' in k}, rename=False)
            smart_partial_load_model_state_dict(self.projector, {'weight': state_dict['projector.weight']},
                                                rename=False)

        if args.vmodel_fix:
            self.backbone.eval()
            for params in self.backbone.parameters():
                params.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return self.backbone.no_weight_decay()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return self.backbone.no_weight_decay_keywords()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image):
        latents = self.backbone(image)
        if hasattr(self, 'projector'):
            latents = F.linear(input=latents.float(),
                               weight=self.projector.weight.float())
        return latents

    @torch.no_grad()
    def distill_forward(self, image):
        self.backbone.eval()
        if hasattr(self, 'projector'):
            self.projector.eval()
        latents = self.backbone(image)
        if hasattr(self, 'projector'):
            latents = F.linear(input=latents.float(),
                               weight=self.projector.weight.float(),
                               )
        return latents
