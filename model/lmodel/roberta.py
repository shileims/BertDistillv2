import torch
from .register import LModels
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaTokenizer
from .roberta_model import RobertaModel
from timm.models.layers import trunc_normal_
from utils import smart_partial_load_model_state_dict, logger


@LModels.register_module
class Roberta(nn.Module):
    def __init__(self, args):
        super().__init__()
        configuration = RobertaConfig.from_pretrained('roberta-base')
        configuration = configuration.__dict__.copy()
        configuration.update({'return_dict': False})
        configuration.update({'gradient_checkpointing': False})
        configuration.pop('model_type')
        configuration = RobertaConfig(**configuration)

        self.avg_pool = args.avg_pool

        if not args.lmodel_from_scratch:
            self.backbone = RobertaModel.from_pretrained('roberta-base', config=configuration, add_pooling_layer=True)
        else:
            self.backbone = RobertaModel(configuration, add_pooling_layer=self.add_pool)

        hidden_size, proj_size = 768, args.proj_size
        self.projector = nn.Linear(hidden_size, args.proj_size, bias=False)

        if args.is_distill:
            args.lmodel_fix = True
            if hasattr(self, 'projector'):
                self.projector.eval()
            for params in self.projector.parameters():
                params.requires_grad = False

            assert not args.lmodel_from_scratch
            logger.log(f'Distilling and Loading pretrain language model from {args.vlmodel_pretrain}')
            state_dict = torch.load(args.vlmodel_pretrain, map_location='cpu')['text']
            smart_partial_load_model_state_dict(self.backbone, {k.replace('backbone.', ''):v for k, v in state_dict.items() if 'backbone.' in k}, rename=False, remove_prefix=False)
            smart_partial_load_model_state_dict(self.projector, {'weight': state_dict['projector.weight']}, rename=False)

        if args.lmodel_fix:
            self.backbone.eval()
            for params in self.backbone.parameters():
                params.requires_grad = False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _output_avg_pool(self, sequence_output, attention_mask):
        '''
        # This version will take padding part into calculation
        # [bs, h]
        # sequence_output_txt = F.adaptive_avg_pool1d(sequence_output_txt.transpose(1,2), 1).transpose(1,2)
        # sequence_output_img = F.adaptive_avg_pool1d(sequence_output_img.transpose(1,2), 1).transpose(1,2)
        # mask format: [1: attend / 0: ignore]
        '''
        # [bs, 1, 1]
        seq_len = attention_mask.squeeze().sum(-1, keepdim=True).unsqueeze(-1)
        # [bs, sq_len, 1]
        attention_mask = attention_mask.squeeze().unsqueeze(-1)
        # [bs, 1, h]
        pooled_output = (sequence_output * attention_mask).sum(1, keepdim=True) / seq_len
        return pooled_output.squeeze()

    def forward(self, sentence):
        if self.avg_pool:
            latents = self.backbone(**sentence, return_dict=False)[0]
            latents = self._output_avg_pool(latents, sentence['attention_mask'])
        else:
            latents = self.backbone(**sentence, return_dict=False)[1]


        latents = F.linear(input=latents.float(), weight=self.projector.weight.float())
        return latents
