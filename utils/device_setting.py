import os
import torch
from .logging import logger

def device_initialize(gpu_ids):
    gpu_ids = gpu_ids.split(',')
    if torch.cuda.device_count() > len(gpu_ids):
        gpu_ids = [i for i in range(torch.cuda.device_count())]
        str_gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    if len(gpu_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str_gpu_ids)
    logger.log(f'{len(gpu_ids)} GPUs are used for training and validation!')

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device('cuda:{}'.format(gpu_ids[0]))
    return device, gpu_ids