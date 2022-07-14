from .register import Optimizers
from .adamw import AdamW
from utils import LayerDecayValueAssigner
import json
from utils import logger
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def  get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, get_num_layer_l=None, get_layer_scale_l=None, skip_keywords=[]):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if name.startswith('vision_model'):
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "vision_backbone_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None
        if name.startswith('language_model'):
            if get_num_layer_l is not None:
                layer_id = get_num_layer_l(name)
                group_name = "language_backbone_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None and 'vision_backbone' in group_name:
                scale = get_layer_scale(layer_id)
            else:
                if get_layer_scale_l is not None and 'language_backbone' in group_name:
                    scale = get_layer_scale_l(layer_id)
                else:
                    scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                # "lr": scale * base_lr
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                # "lr": scale * base_lr
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    logger.log("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def build_optimizer_parameters_group(args, model):
    n_parameters = sum(p.numel() for p in model.vision_model.parameters() if p.requires_grad)
    logger.info(f'Vision model # of trainable params: {n_parameters/10**6}M')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Vision-language model # of trainable params: {n_parameters/10**6}M')
    vision_model_flops = FlopCountAnalysis(model.vision_model, torch.zeros(1, 3, args.input_size, args.input_size).to(args.device))
    logger.info(f'Vision model flops: {flop_count_table(vision_model_flops)}')

    no_decay = {'vision_model.backbone.pos_embed', 'vision_model.backbone.cls_token',
                'vision_model.backbone.dist_token', 'logit_scale'}

    depth = model.vision_model.backbone.depths
    num_layers = sum(depth)
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            depth=depth, is_sentence=False)
    elif args.backbone_lr_mult != 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.backbone_lr_mult for i in range(num_layers + 1)) + [1.0],
            depth=depth, is_sentence=False)
    else:
        assigner = None

    num_layers = 12  # language backbone layers
    if args.layer_decay_l < 1.0:
        assigner_l = LayerDecayValueAssigner(
            list(args.layer_decay_l ** (num_layers + 1 - i) for i in range(num_layers + 2)), is_sentence=True)
    elif args.backbone_lr_mult_l != 1.0:
        assigner_l = LayerDecayValueAssigner(
            list(args.backbone_lr_mult_l for i in range(num_layers + 1)) + [1.0], is_sentence=True)
    else:
        assigner_l = None

    if assigner is not None:
        logger.log("Assigned values = %s" % (str(assigner.values)))
    if assigner_l is not None:
        logger.log("Assigned values(language) = %s" % (str(assigner_l.values)))

    optimizer_grouped_parameters = get_parameter_groups(model, args.weight_decay, no_decay,
                                                            get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                                            get_layer_scale=assigner.get_scale if assigner is not None else None,
                                                            get_num_layer_l=assigner_l.get_layer_id if assigner_l is not None else None,
                                                            get_layer_scale_l=assigner_l.get_scale if assigner_l is not None else None,
                                                            skip_keywords=model.vision_model.no_weight_decay_keywords()
                                                            )
    return optimizer_grouped_parameters

def build_optimizer(args, model, data_loader):
    args.step_num = len(data_loader) * args.epochs // args.accumulate_step
    optimizer_grouped_parameters = build_optimizer_parameters_group(args, model)
    optimizer = Optimizers.get(args.optimizer)(optimizer_grouped_parameters, args)
    return optimizer

def build_distill_optimizer_parameters_group(args, model):
    if hasattr(model, 'module'):
        n_parameters = sum(p.numel() for p in model.module.distill_model.parameters() if p.requires_grad)
        logger.info(f'Distill model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.module.distill_projector.parameters() if p.requires_grad)
        logger.info(f'Distill projector # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.module.vision_model.parameters() if p.requires_grad)
        logger.info(f'Vision model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.module.language_model.parameters() if p.requires_grad)
        logger.info(f'Language model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        logger.info(f'Vision-language model # of trainable params: {n_parameters / 10 ** 6}M')
        distill_model_flops = FlopCountAnalysis(model.module.distill_model,
                                                torch.zeros(2, 3, args.stu_size, args.stu_size).to(args.device))
        logger.info(f'Distill model flops: {flop_count_table(distill_model_flops)}')
    else:
        n_parameters = sum(p.numel() for p in model.distill_model.parameters() if p.requires_grad)
        logger.info(f'Distill model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.vision_model.parameters() if p.requires_grad)
        logger.info(f'Vision model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        logger.info(f'Language model # of trainable params: {n_parameters / 10 ** 6}M')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Vision-language model # of trainable params: {n_parameters / 10 ** 6}M')
        distill_model_flops = FlopCountAnalysis(model.distill_model,
                                                torch.zeros(2, 3, args.stu_size, args.stu_size).to(args.device))
        logger.info(f'Distill model flops: {flop_count_table(distill_model_flops)}')

    optimizer_grouped_parameters = model.parameters()
    return optimizer_grouped_parameters

def build_distill_optimizer(args, model, data_loader):
    args.step_num = len(data_loader) * args.epochs // args.accumulate_step
    optimizer_grouped_parameters = build_distill_optimizer_parameters_group(args, model)
    optimizer = Optimizers.get(args.optimizer)(optimizer_grouped_parameters, args)
    return optimizer
