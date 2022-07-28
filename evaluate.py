import copy

import torch.cuda
import argparse
import os.path
import json
import numpy as np
import torch.cuda.amp as amp

from data import create_train_val_data_loader, create_train_data_loader, create_distill_train_val_data_loader, create_distill_eval_loader
from utils import device_initialize
from model import create_model, create_distill_model, create_distill_quantization_model
from optim import build_optimizer, build_distill_optimizer
from scheduler import create_scheduler
from loss_scaler import create_loss_scaler
from loss import create_loss
from pathlib import Path
from utils import logger, load_checkpoint
from tqdm import tqdm
from trainer import Trainer, TrainerDistill, TrainerDistributedDistill
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils import init_distributed_mode, get_rank, get_world_size, is_main_process, create_output_dir

def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    # gerneral arguments
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--dataset', default='RetrievalDistill', type=str)
    parser.add_argument('--sampler', default='SequentialSampler', type=str)
    parser.add_argument('--collator', default='RetrievalCollatorDistill', type=str)
    parser.add_argument('--split', type=str, default='coco')
    parser.add_argument('--transforms', default='SStockTransformsDistill')
    parser.add_argument('--dataset_path', default='/home/shilei/datasets/SS200M_split_0')

    # distribution arguments
    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # position embedding dropout; dropout in FFN
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    # FFN & self-Att output dropout
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    # Attention map dropout
    parser.add_argument('--drop-att', type=float, default=0.0, metavar='PCT', help='Drop att rate (default: 0.)')

    parser.add_argument('--amp', action='store_true', default=False, help='Mixed precision training by using torch amp')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--auto_resume', action='store_false', default=True, help='Automatically loading the latest checkpoint')
    parser.add_argument('--resume', type=str, default='', help='Specify the path of checkpoint the model needs to load')
    parser.add_argument('--resume-weight-only', action='store_true', default=False, help='Auto resume from the latest checkpoint')

    # Model parameters
    parser.add_argument('--avg-pool', action='store_true', default=True, help='avg pool for sentence model output instead of cls token+fc')

    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=str, default='CosineScheduler')
    parser.add_argument('--loss_scaler', type=str, default='NativeScaler')
    parser.add_argument('--loss', type=str, default='NCE')
    parser.add_argument('--vl_label_smooth', type=float, default=0.1)

    parser.add_argument('--vmodel-from-scratch', action='store_true', default=False)
    parser.add_argument('--lmodel-from-scratch', action='store_true', default=False)
    parser.add_argument('--vmodel-pretrain', default='/home/shilei/Documents/BertPretrain/pretrain/image_model.pth', type=str)
    parser.add_argument('--vlmodel-pretrain', default='/home/shilei/Documents/BertPretrain/pretrain/ss200m_vl_for_distill.pth', type=str)
    parser.add_argument('--lmodel-pretrain', default='', type=str)
    parser.add_argument('--vmodel-fix', action='store_true', default=False)
    parser.add_argument('--lmodel-fix', action='store_true', default=False)
    parser.add_argument('--vlmodel', default='BertDistill', type=str)
    parser.add_argument('--base_model', default='SwinUnsup', type=str)
    parser.add_argument('--vmodel', default='BaseSwin', type=str)
    parser.add_argument('--lmodel', default='Roberta', type=str)

    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--fix-tau', default=0.05, type=float, help='if want to fix tau, set non zero value')
    parser.add_argument('--proj-size', type=int, default=256, help='the projector dimension')

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default=f'/home/shilei/results/BertDistill_0710/outputs/tmp')

    parser.add_argument('--train_rank', type=list, default=[1])
    parser.add_argument('--train_metric_names', type=list, default=['Loss', 'V2LAcc_R1', 'L2VAcc_R1'])
    parser.add_argument('--val_rank', type=list, default=[1,5,10])
    parser.add_argument('--val_metric_names', type=list, default=['Loss', 'V2LAcc_R1', 'L2VAcc_R1', 'V2LAcc_R5', 'L2VAcc_R5', 'V2LAcc_R10', 'L2VAcc_R10', 'V2LAcc_R128', 'L2VAcc_R128'])

    # distill arguments
    parser.add_argument('--is_distill', action='store_false', default=True)
    parser.add_argument('--tea-img-size', type=int, default=224)
    parser.add_argument('--stu-img-size', type=int, default=192)
    parser.add_argument('--distill-model', type=str, default='swin_mini4')
    parser.add_argument('--distill_train_rank', type=list, default=[1])
    parser.add_argument('--distill_train_metric_names', type=list, default=['Loss', 'TeaV2LAcc_R1', 'TeaL2VAcc_R1', 'StuV2LAcc_R1', 'StuL2VAcc_R1'])
    parser.add_argument('--distill_val_rank', type=list, default=[1, 5, 10])
    parser.add_argument('--distill_val_metric_names', type=list,
                        default=['Loss', 'TeaV2LAcc_R1', 'TeaL2VAcc_R1', 'TeaV2LAcc_R5', 'TeaL2VAcc_R5', 'TeaV2LAcc_R10', 'TeaL2VAcc_R10',
                                 'TeaV2LAcc_R128', 'TeaL2VAcc_R128', 'StuV2LAcc_R1', 'StuL2VAcc_R1', 'StuV2LAcc_R5', 'StuL2VAcc_R5', 'StuV2LAcc_R10', 'StuL2VAcc_R10',
                                 'StuV2LAcc_R128', 'StuL2VAcc_R128'])
    parser.add_argument('--eval_model_dir', type=str, default='')
    parser.add_argument('--eval_model_path', type=str, default='')

    # augmentation arguments

    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # debug arguments
    parser.add_argument('--debug', action='store_false', default=True)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--debug_batch_size', type=int, default=32)
    return parser

def load_weights(model, model_path=None):
    weight_path = model_path
    assert os.path.isfile(weight_path), f'{weight_path} is not a valid file'

    strict = True
    model_load_func = lambda m: getattr(m, 'load_state_dict')
    checkpoint = torch.load(weight_path, map_location='cpu')
    stat = model_load_func(model.module if hasattr(model, 'module') else model)(checkpoint['model'], strict=strict)
    logger.log(stat)
    logger.log(" --------------> Loaded pretrain-weight from {}; epochs {}".format(weight_path, checkpoint['epoch'] + 1))


def device_settings(gpu_ids):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids}')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device('cuda:{}'.format(gpu_ids))
    return device


def create_labels(similarities, acc_pos_num, isT2I=False, begin_idx=0):
    # hack for speedup, real bottleneck
    labels = torch.zeros_like(similarities)
    for idx, (b, e) in enumerate(zip(acc_pos_num[:-1], acc_pos_num[1:])):
        if isT2I:
            labels[idx, b:e] = 1
        else:
            labels[b:e, idx+begin_idx] = 1
    return labels

def accumulate_list(x):
    acc_x = [0]
    for idx, num in enumerate(x):
        acc_x.append(acc_x[-1] + num)
    return acc_x


def compute_rank(similarities, labels, keep_top_k=128):
    def _compute_rank(_labels, _similarities, _total_captions, _keep_top_k):
        ranks = []
        for lab, sim in zip(labels, similarities):
            sims, inds = torch.topk(sim, keep_top_k)
            rank = total_captions
            for r, ind in enumerate(inds):
                if r >= keep_top_k:
                    break
                if lab[ind] == 1:
                    rank = r
                    break
            ranks.append(rank)
        return ranks

    total_captions = 1e8
    ranks = _compute_rank(labels, similarities, total_captions, keep_top_k)

    return ranks


def compute_sims(predict_model, full_text_latents, full_tea_image_latents, num_captions_per_img=5):
    rank = [1, 5, 10]

    similarities_T2I = torch.clamp(
        predict_model.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01))).exp() * full_text_latents @ full_tea_image_latents.t()

    t2i_ranks = []
    labels = create_labels(similarities_T2I, accumulate_list([num_captions_per_img]*full_tea_image_latents.size(0)), isT2I=False, begin_idx=0)
    t2i_ranks.extend(compute_rank(similarities_T2I, labels, keep_top_k=max(rank)))
    del similarities_T2I
    torch.cuda.empty_cache()

    i2t_ranks = []
    pos_label_loc = accumulate_list([num_captions_per_img] * full_tea_image_latents.size(0))

    similarities_I2T = torch.clamp(predict_model.logit_scale, max=torch.log(
        torch.tensor(1. / 0.01))).exp() * full_tea_image_latents @ full_text_latents.t()
    labels = create_labels(similarities_I2T, pos_label_loc, isT2I=True, begin_idx=0)
    i2t_ranks.extend(compute_rank(similarities_I2T, labels, keep_top_k=max(rank)))

    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) * 100. for r in rank]
    print("Eval -- I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) * 100. for r in rank]
    print("Eval -- T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                t2i_accs[0], t2i_accs[1], t2i_accs[2]))

def main_distill(args):
    # device setting
    device = device_settings(args.gpu_ids)
    args.device = device

    # build dataloader
    dataloader = create_distill_eval_loader(args)

    # build model
    model = create_distill_model(args)
    model = model.to(device)
    model.eval()

    if args.eval_model_dir:
        for file in os.listdir(args.eval_model_dir):
            if file.endswith('.pth'):
                args.eval_model_path = os.path.join(args.eval_model_dir, file)
                load_weights(model, args.eval_model_path)

                num_captions_per_img = 5
                full_tea_image_latents = []
                full_stu_image_latents = []
                full_text_latents = []

                predict_model = model.module if hasattr(model, 'module') else model
                for batch_id, batch in enumerate(tqdm(dataloader)):
                    input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
                    input_tea_img = input_tea_img.to(args.device, non_blocking=True)
                    input_stu_img = input_stu_img.to(args.device, non_blocking=True)
                    input_text = {k: v.to(args.device, non_blocking=True) for k, v in input_text.items()}

                    # predict
                    tea_image_latents = predict_model.encode_tea_image(input_tea_img)
                    tea_image_latents /= tea_image_latents.norm(dim=-1, keepdim=True)
                    stu_image_latents = predict_model.encode_stu_image(input_stu_img)
                    stu_image_latents /= stu_image_latents.norm(dim=-1, keepdim=True)
                    text_latents = predict_model.encode_tea_text(input_text)
                    text_latents /= text_latents.norm(dim=-1, keepdim=True)
                    full_tea_image_latents.append(tea_image_latents.detach())
                    full_stu_image_latents.append(stu_image_latents.detach())
                    full_text_latents.append(text_latents.detach())

                full_tea_image_latents = torch.cat(full_tea_image_latents, dim=0)
                full_stu_image_latents = torch.cat(full_stu_image_latents, dim=0)
                full_text_latents = torch.cat(full_text_latents, dim=0)

                print(f'Teacher')
                compute_sims(predict_model, full_text_latents, full_tea_image_latents)
                print(f'Student')
                compute_sims(predict_model, full_text_latents, full_stu_image_latents)
    exit(0)
    if not args.eval_model_dir:
        assert args.eval_model_path, f'model dir and model path cannot be empty at the same time'
        load_weights(model, args.eval_model_path)

    num_captions_per_img = 5
    full_tea_image_latents = []
    full_stu_image_latents = []
    full_text_latents = []

    predict_model = model.module if hasattr(model, 'module') else model
    for batch_id, batch in enumerate(tqdm(dataloader)):
        input_tea_img, input_stu_img, input_text = batch[0], batch[1], batch[2]
        input_tea_img = input_tea_img.to(args.device, non_blocking=True)
        input_stu_img = input_stu_img.to(args.device, non_blocking=True)
        input_text = {k: v.to(args.device, non_blocking=True) for k, v in input_text.items()}

        # predict
        tea_image_latents = predict_model.encode_tea_image(input_tea_img)
        tea_image_latents /= tea_image_latents.norm(dim=-1, keepdim=True)
        stu_image_latents = predict_model.encode_stu_image(input_stu_img)
        stu_image_latents /= stu_image_latents.norm(dim=-1, keepdim=True)
        text_latents = predict_model.encode_tea_text(input_text)
        text_latents /= text_latents.norm(dim=-1, keepdim=True)
        full_tea_image_latents.append(tea_image_latents.detach())
        full_stu_image_latents.append(stu_image_latents.detach())
        full_text_latents.append(text_latents.detach())

    full_tea_image_latents = torch.cat(full_tea_image_latents, dim=0)
    full_stu_image_latents = torch.cat(full_stu_image_latents, dim=0)
    full_text_latents = torch.cat(full_text_latents, dim=0)

    print(f'Teacher')
    compute_sims(predict_model, full_text_latents, full_tea_image_latents)
    print(f'Student')
    compute_sims(predict_model, full_text_latents, full_stu_image_latents)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main_distill(args)
