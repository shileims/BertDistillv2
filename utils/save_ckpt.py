import torch
import copy
import os
from pathlib import Path
from .logging import logger

CHECKPOINT_EXTN = ".pth"

def save_checkpoint(train_iterations, model, optimizer, epoch, args, checkpoint_name=None, loss_scaler=None, stat=None):
    if not args.quantization_aware_training:
        checkpoint_folder = args.output_dir + '/' + 'ckpts'
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        if checkpoint_name is not None:
            checkpoint_path = checkpoint_folder + '/' + checkpoint_name
        else:
            checkpoint_path = checkpoint_folder + '/' + f'checkpoint_{epoch}.pth'

        stat_dict = {
            'train_iters': train_iterations,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'args': args,
            'stat': stat,
        }
        torch.save(stat_dict, checkpoint_path)
        return checkpoint_path
    else:
        checkpoint_folder = args.output_dir + '/' + 'ckpts'
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        if checkpoint_name is not None:
            checkpoint_path = checkpoint_folder + '/' + checkpoint_name
        else:
            checkpoint_path = checkpoint_folder + '/' + f'quantization_checkpoint_{epoch}.pth'
        int8_model = torch.quantization.convert(model)
        int8_model.eval()
        # torch.jit.save(torch.jit.script(int8_model), checkpoint_path)
        torch.save(int8_model.state_dict(), checkpoint_path)
        return checkpoint_path


def load_checkpoint(args, auto_resume, resume, model, optimizer):
    res = ''
    if resume:
        res = resume
    elif auto_resume:
        ckpt_loc = '{}/{}/checkpoint_'.format(args.output_dir, 'ckpts')
        for epoch in range(args.epochs)[::-1]:
            res = Path(ckpt_loc + str(epoch) + CHECKPOINT_EXTN)
            if res.exists():
                break
        if not res.exists():
            res = ''

    if not res:
        logger.log('No checkpoint is found for resuming training!')
        return

    strict = False if args.resume_weight_only else True
    model_load_func = lambda m: getattr(m, 'load_state_dict')
    checkpoint = torch.load(res, map_location='cpu')
    stat = model_load_func(model.module if hasattr(model, 'module') else model)(checkpoint['model'], strict=strict)
    logger.log(stat)
    if resume:
        logger.log(" --------------> loading pretrained weights from {}; start epoch {}".format(args.resume, args.start_epoch))
    elif auto_resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        logger.log(" --------------> resume from {}; start epoch {}".format(args.resume, args.start_epoch))
    if args.is_dist:
        torch.distributed.barrier()


def load_weights(args, model):
    weight_path = os.path.join(args.output_dir, args.quantization_ckpt_path)
    assert os.path.isfile(weight_path), f'{weight_path} is not a valid file'

    strict = True
    model_load_func = lambda m: getattr(m, 'load_state_dict')
    checkpoint = torch.load(weight_path, map_location='cpu')
    stat = model_load_func(model.module if hasattr(model, 'module') else model)(checkpoint['model'], strict=strict)
    logger.log(stat)
    logger.log(" --------------> Loaded pretrain-weight from {}; epochs {}".format(weight_path, checkpoint['epoch'] + 1))
