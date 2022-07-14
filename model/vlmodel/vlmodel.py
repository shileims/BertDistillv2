import torch
from torch.nn import functional as F
from torch import nn
from utils import SyncFunction, logger
from .register import VLModels
from model.vmodel import VModels
from model.lmodel import LModels

@VLModels.register_module
class Bert(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.fix_tau == 0:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / 0.36788)), requires_grad=True)
        else:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1. / args.fix_tau)), requires_grad=False)

        self.vision_model   = VModels.get(args.vmodel)(args)
        self.language_model = LModels.get(args.lmodel)(args)
        logger.log(f'Bert model is initialized')

    @torch.no_grad()
    def encode_image(self, image):
        image_latents = self.vision_model(image)
        return image_latents

    @torch.no_grad()
    def encode_text(self, sentence):
        text_latents = self.language_model(sentence)
        return text_latents

    def forward(self, image, sentence, return_latent_only=False):

        text_latents = self.language_model(sentence)
        image_latents = self.vision_model(image)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()

        text_latents, image_latents = \
            map(lambda t: F.normalize(t, p = 2, dim = -1) if t is not None else t, (text_latents, image_latents))

        if return_latent_only:
            return image_latents, text_latents

        image_latents_gathered = image_latents
        text_latents_gathered = text_latents

        sim_i_2_t = logit_scale * image_latents @ text_latents_gathered.t()
        sim_t_2_i = logit_scale * text_latents @ image_latents_gathered.t()

        return sim_i_2_t, sim_t_2_i