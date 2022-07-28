import copy

import torch.cuda
import argparse
import os.path
import json
import numpy as np
import torch.cuda.amp as amp

from data import create_train_val_data_loader, create_train_data_loader, create_distill_train_val_data_loader
from utils import device_initialize
from model import create_model, create_distill_model, create_distill_quantization_model
from optim import build_optimizer, build_distill_optimizer
from scheduler import create_scheduler
from loss_scaler import create_loss_scaler
from loss import create_loss
from pathlib import Path
from utils import logger, load_checkpoint, load_weights
from trainer import Trainer, TrainerDistill, TrainerDistributedDistill
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils import init_distributed_mode, get_rank, get_world_size, is_main_process, create_output_dir


def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    # gerneral arguments
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--train_fix_length', type=int, default=2000000, help='The default is 1483257')
    parser.add_argument('--val_fix_length', type=int, default=2048, help='The default is 1483257')
    parser.add_argument('--dataset', default='SStock', type=str)
    parser.add_argument('--train_sampler', default='RandomSampler', type=str)
    parser.add_argument('--val_sampler', default='SequentialSampler', type=str)
    parser.add_argument('--collator', default='SStockCollator', type=str)
    parser.add_argument('--dataset_path', default='/home/shilei/datasets/SS200M_split_0')
    parser.add_argument('--transforms', default='SStockTransformsDistill')

    # distribution arguments
    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

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
    parser.add_argument('--layer-decay', type=float, default=1.0, metavar='LR',
                        help='backbone learning rate decay layer by layer on vision backbone')
    parser.add_argument('--layer-decay-l', type=float, default=1.0, metavar='LR',
                        help='backbone learning rate decay layer by layer on language backbone')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1, metavar='LR',
                        help='backbone learning rate decay [vision branch only]')
    parser.add_argument('--backbone-lr-mult-l', type=float, default=0.1, metavar='LR',
                        help='backbone learning rate decay [language branch only]')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-steps', type=int, default=100, metavar='N',
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, help='epochs to cooldown LR, if scheduler supports')
    parser.add_argument('--accumulate_step', type=int, default=1, metavar='LR',
                        help='accumulate step for train model with smaller bs')

    # position embedding dropout; dropout in FFN
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    # FFN & self-Att output dropout
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    # Attention map dropout
    parser.add_argument('--drop-att', type=float, default=0.0, metavar='PCT', help='Drop att rate (default: 0.)')

    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--fix-tau', default=0.05, type=float, help='if want to fix tau, set non zero value')
    parser.add_argument('--proj-size', type=int, default=256, help='the projector dimension')

    parser.add_argument('--epochs', default=16, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--start_iteration', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default=f'/home/shilei/results/BertDistill_0710/outputs/tmp')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--save_ckpt_freq', type=int, default=1, help='Frequency of saving ckpt')
    parser.add_argument('--log_freq', type=int, default=1, help='Frequency of saving ckpt')
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

    # quantization aware training arguments
    parser.add_argument('--quantization_aware_training', action='store_true', default=False)
    parser.add_argument('--quantization_ckpt_path', type=str, default='ckpts/checkpoint_0.pth')

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
    parser.add_argument('--train_num_samples', type=int, default=512)
    parser.add_argument('--val_num_samples', type=int, default=256)
    parser.add_argument('--debug_batch_size', type=int, default=32)
    return parser


def main(args):
    # device setting
    device, gpu_ids = device_initialize(args.gpu_ids)
    args.device = device
    args.gpu_ids = gpu_ids
    args.num_workers = args.num_workers*len(gpu_ids)
    # build dataloader
    train_data, val_data, _ = create_train_val_data_loader(args)
    train_dataloader, val_dataloader = train_data[-1], val_data[-1]
    train_sampler = train_data[1]
    # build model
    model = create_model(args)
    model = model.to(device)
    # build optimizer
    optimizer = build_optimizer(args, model, train_dataloader).optimizer
    # print(optmizer)
    # build scheduler
    lr_scheduler = create_scheduler(args, train_dataloader)
    # print(lr_scheduler)
    # build loss scaler
    # loss_scaler = create_loss_scaler(args)
    # print(loss_scaler)
    # build loss
    criterion       = create_loss(args)
    # print(loss)

    # Training process
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(json.dumps(str(args)))

    train_engine = Trainer(args=args,
                           model=model,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           val_dataloader=val_dataloader,
                           train_dataloader=train_dataloader,
                           criterion=criterion)
    train_engine.run(train_sampler=train_sampler)

def main_distill(args):
    # device setting
    device, gpu_ids = device_initialize(args.gpu_ids)
    args.device = device
    # build dataloader
    train_data, val_data, _ = create_distill_train_val_data_loader(args)
    train_dataloader, val_dataloader = train_data[-1], val_data[-1]
    train_sampler = train_data[1]
    # build model
    model = create_distill_model(args)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f'Model is training in {torch.cuda.device_count()} gpus')
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    # build optimizer
    optimizer = build_distill_optimizer(args, model, train_dataloader).optimizer

    # build scheduler
    lr_scheduler = create_scheduler(args, train_dataloader)

    # build loss scaler
    loss_scaler = None
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()

    # build loss
    criterion       = create_loss(args)
    # print(loss)

    if args.auto_resume or args.resume:
        load_checkpoint(args, args.auto_resume, args.resume, model, optimizer)


    # Training process
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(json.dumps(str(args)))

    train_engine = TrainerDistill(args=args,
                           model=model,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           val_dataloader=val_dataloader,
                           train_dataloader=train_dataloader,
                           criterion=criterion,
                           loss_scaler=loss_scaler)
    train_engine.run(train_sampler=train_sampler)

def main_distill_quantization(args):
    # device setting
    device = torch.device('cpu')
    args.device = device
    args.amp = False
    # build dataloader
    train_data, val_data, _ = create_distill_train_val_data_loader(args)
    train_dataloader, val_dataloader = train_data[-1], val_data[-1]
    train_sampler = train_data[1]

    # build model
    model = create_distill_model(args)
    load_weights(args, model)
    model = model.to(device)
    fused_model = copy.deepcopy(model)
    model.train()
    fused_model.train()

    # Fusing modules for reducing accuracy of quantization and dequantization
    # Official support operators are: ConV + BN、ConV + BN + ReLU、Conv + ReLU、Linear + ReLU、BN + ReLU
    # So for transformer, this step is omitted

    model.eval()
    fused_model.eval()
    model.distill_quantization()

    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    model.qconfig = quantization_config
    torch.quantization.prepare_qat(model.distill_quantization_model, inplace=True)

    # build optimizer
    optimizer = build_distill_optimizer(args, model, train_dataloader).optimizer

    # build scheduler
    lr_scheduler = create_scheduler(args, train_dataloader)
    # 
    # build loss
    criterion = create_loss(args)
    # 
    # Training process
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(json.dumps(str(args)))

    train_engine = TrainerDistill(args=args,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=lr_scheduler,
                                  val_dataloader=val_dataloader,
                                  train_dataloader=train_dataloader,
                                  criterion=criterion)
    train_engine.run(train_sampler=train_sampler)

def main_distill_dist(args):
    assert args.is_dist, f'args.is_dist should be True'
    assert args.local_rank >= 0, f'For distributed training, local_rank should be set'
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # build dataloader
    if args.train_fix_length > 2e6:
        args.train_sampler = 'RandomSampler'
        args.val_sampler = 'SequentialSampler'
        # args.train_sampler = 'DistributedSampler'
        # args.val_sampler = 'DistributedSampler'
        args.dataset = 'DistribSStockDistill'
    else:
        args.dataset = 'SStockDistill'
        args.train_sampler = 'DistributedSampler'
        args.val_sampler = 'DistributedSampler'

    args.num_tasks = get_world_size()
    args.global_rank = get_rank()
    train_data, val_data, _ = create_distill_train_val_data_loader(args)
    train_dataloader, val_dataloader = train_data[-1], val_data[-1]
    train_sampler = train_data[1]

    # build model
    model = create_distill_model(args)
    model = model.to(device)

    # DDP the model
    model = DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True, broadcast_buffers=False)

    # build optimizer
    optimizer = build_distill_optimizer(args, model, train_dataloader).optimizer

    # build scheduler
    lr_scheduler = create_scheduler(args, train_dataloader)

    # build loss
    args.loss = 'DistributedNCE'
    criterion       = create_loss(args)
    
    # build loss_scaler
    loss_scaler = None
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()

    if args.auto_resume or args.resume:
        load_checkpoint(args, args.auto_resume, args.resume, model, optimizer)


    # Training process
    if is_main_process():
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            f.write(json.dumps(str(args)))

    train_engine = TrainerDistributedDistill(args=args,
                           model=model,
                           optimizer=optimizer,
                           scheduler=lr_scheduler,
                           val_dataloader=val_dataloader,
                           train_dataloader=train_dataloader,
                           criterion=criterion,
                           is_mater=is_main_process(),
                           loss_scaler=loss_scaler)
    train_engine.run(train_sampler=train_sampler)

def main_distill_dist_quantization(args):
    """
    Not Finished!
    :param args:
    :return:
    """
    assert args.is_dist, f'args.is_dist should be True'
    assert args.local_rank >= 0, f'For distributed training, local_rank should be set'
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    init_distributed_mode(args)
    device = torch.device('cpu')
    args.device = device

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # build dataloader
    args.train_sampler = 'DistributedSampler'
    args.val_sampler = 'DistributedSampler'
    args.num_tasks = get_world_size()
    args.global_rank = get_rank()
    train_data, val_data, _ = create_distill_train_val_data_loader(args)
    train_dataloader, val_dataloader = train_data[-1], val_data[-1]
    train_sampler = train_data[1]

    # build model
    model = create_distill_model(args)
    load_weights(args, model)
    model = model.to(device)
    fused_model = copy.deepcopy(model)
    model.train()
    fused_model.train()

    # DDP the model
    model = DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True,
                                    broadcast_buffers=False)

    # build optimizer
    optimizer = build_distill_optimizer(args, model, train_dataloader).optimizer

    # build scheduler
    lr_scheduler = create_scheduler(args, train_dataloader)

    # build loss
    args.loss = 'DistributedNCE'
    criterion = create_loss(args)

    # build loss_scaler
    loss_scaler = None
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()

    if args.auto_resume or args.resume:
        load_checkpoint(args, args.auto_resume, args.resume, model, optimizer)

    # Training process
    if is_main_process():
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            f.write(json.dumps(str(args)))

    train_engine = TrainerDistributedDistill(args=args,
                                             model=model,
                                             optimizer=optimizer,
                                             scheduler=lr_scheduler,
                                             val_dataloader=val_dataloader,
                                             train_dataloader=train_dataloader,
                                             criterion=criterion,
                                             is_mater=is_main_process(),
                                             loss_scaler=loss_scaler)
    train_engine.run(train_sampler=train_sampler)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    create_output_dir(args)
    # args.is_distill = True
    # args.is_dist = False
    # args.quantization_aware_training = True

    if args.is_distill:
        if args.is_dist:
            if args.quantization_aware_training:
                main_distill_dist_quantization(args)
            else:
                main_distill_dist(args)
        else:
            if args.quantization_aware_training:
                main_distill_quantization(args)
            else:
                main_distill(args)
    else:
        main(args)
