import torch
from torch.nn import functional as F
from torch import nn
from timm.models.layers import trunc_normal_
from .register import VLModels
from model.vmodel import VModels
from model.lmodel import LModels
from model.swin_default import build_mini_model
from utils import SyncFunction, ArgMax

@VLModels.register_module
class BertDistill(nn.Module):
    def __init__(self, args, quantization=False):
        super().__init__()

        self.quantization = quantization
        self.is_dist = args.is_dist
        if args.fix_tau == 0:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / 0.36788)), requires_grad=True)
        else:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / args.fix_tau)), requires_grad=False)
        self.language_model_hidden_size = 768

        self.vision_model   = VModels.get(args.vmodel)(args)
        # self.vision_model_projector = nn.Linear(self.vision_model.backbone.num_features, args.proj_size, bias=False)
        self.language_model = LModels.get(args.lmodel)(args)
        # self.language_model_projector = nn.Linear(self.language_model_hidden_size, args.proj_size, bias=False)
        self.distill_model  = build_mini_model(args.distill_model, num_classes=0, use_checkpoint=False, is_dist=args.is_dist)
        self.distill_projector = nn.Linear(int(self.distill_model.num_features * self.distill_model.pastmlp_ratio), args.proj_size, bias=False)

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

    def stu_forward(self, x):
        stu_image_latents = self.distill_model(x)
        stu_image_latents = F.linear(input=stu_image_latents.float(), weight=self.distill_projector.weight.float(),)
        return stu_image_latents

    def forward(self, tea_imgs, stu_imgs, sentence, return_latent_only=False):

        text_latents = self.language_model(sentence)
        with torch.no_grad():
            tea_image_latents = self.vision_model.distill_forward(tea_imgs)
            tea_image_latents /= tea_image_latents.norm(p=2, dim=-1, keepdim=True)

        stu_image_latents = self.stu_forward(stu_imgs)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()

        text_latents, stu_image_latents = \
            map(lambda t: F.normalize(t, p = 2, dim = -1) if t is not None else t, (text_latents, stu_image_latents))

        if return_latent_only:
            if self.is_dist:
                tea_image_latents, stu_image_latents, text_latents = SyncFunction.apply(tea_image_latents), SyncFunction.apply(stu_image_latents), SyncFunction.apply(text_latents)
            return tea_image_latents, stu_image_latents, text_latents

        if self.is_dist:
            tea_image_latents_gathered = SyncFunction.apply(tea_image_latents)
            stu_image_latents_gathered = SyncFunction.apply(stu_image_latents)
            text_latents_gathered = SyncFunction.apply(text_latents)
        else:
            tea_image_latents_gathered = tea_image_latents
            stu_image_latents_gathered = stu_image_latents
            text_latents_gathered = text_latents

        sim_i_2_t = logit_scale * stu_image_latents @ text_latents_gathered.t()
        sim_t_2_i = logit_scale * text_latents @ stu_image_latents_gathered.t()

        sim_i_2_t_bigmodel = tea_image_latents @ text_latents_gathered.t()
        sim_t_2_i_bigmodel = text_latents @ tea_image_latents_gathered.t()
        sim_i_2_t_bigmodel_labels = torch.argmax(sim_i_2_t_bigmodel, dim=-1)
        sim_t_2_i_bigmodel_labels = torch.argmax(sim_t_2_i_bigmodel, dim=-1)
        # sim_i_2_t_bigmodel_labels = ArgMax(sim_i_2_t_bigmodel)
        # sim_t_2_i_bigmodel_labels = ArgMax(sim_t_2_i_bigmodel)
        return sim_i_2_t, sim_t_2_i, sim_i_2_t_bigmodel, sim_t_2_i_bigmodel, sim_i_2_t_bigmodel_labels, sim_t_2_i_bigmodel_labels
