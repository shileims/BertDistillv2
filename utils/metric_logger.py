# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import json
import math
import numpy as np
from collections import defaultdict, deque
import datetime
from PIL import Image
from scipy import interpolate

import torch
from torch._six import inf
import torch.distributed as dist

""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp

    has_apex = True
except ImportError:
    amp = None
    has_apex = False


# ================= Scaler ===============

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    # : https://github.com/zeliu98/Swin/blob/840169d6dbd49f0769529b04873e0d7d753dbc94/utils.py
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, is_step=True):
        # is step is for accumulate step
        grad_norm = 0
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if is_step and clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)
        if is_step:
            if clip_grad is None:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
            optimizer.step()
        return grad_norm

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, is_step=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if is_step and clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        if is_step:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# ================= Params ===============
def get_num_layer_for_swin(var_name, num_max_layer, depth):
    if var_name.startswith("visual_model.backbone.patch_embed"):
        return 0
    elif var_name.startswith("visual_model.backbone.layers"):
        var_name = var_name.replace("visual_model.backbone.layers", "layers")
        layer_id = int(var_name.split('.')[1])
        block_id = var_name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm' or block_id == 'channel_reduction':
            return sum(depth[:layer_id + 1])
        layer_id = sum(depth[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_max_layer - 1


def get_num_layer_for_unilm(var_name, num_max_layer):
    if var_name.startswith("sentence_model.backbone.embeddings") or var_name.startswith(
            "sentence_model.backbone.encoder.rel_pos_bias"):
        return 0
    elif var_name.startswith("sentence_model.backbone.encoder.layer"):
        var_name = var_name.replace("sentence_model.backbone.encoder.layer", "layer")
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values, depth=None, is_sentence=False):
        self.values = values
        self.depth = depth
        self.is_sentence = is_sentence

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if self.is_sentence:
            return get_num_layer_for_unilm(var_name, len(self.values))
        else:
            return get_num_layer_for_swin(var_name, len(self.values), self.depth)


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None,
                         get_num_layer_l=None, get_layer_scale_l=None, skip_keywords=[]):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name,
                                                                                                            skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if name.startswith('visual_model'):
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "visual_backbone_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None
        if name.startswith('sentence_model'):
            if get_num_layer_l is not None:
                layer_id = get_num_layer_l(name)
                group_name = "sentence_backbone_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None and 'visual_backbone' in group_name:
                scale = get_layer_scale(layer_id)
            else:
                if get_layer_scale_l is not None and 'sentence_backbone' in group_name:
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
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    # copied from https://github.com/zdaxie/beit-ms/blob/main/utils.py
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep, f"scheduler: {len(schedule)}; total step: {epochs * niter_per_ep}"
    return schedule


# ================= Logger ===============
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for idx, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield idx, obj
            iter_time.update(time.time() - end)
            if (i != 0) and (i % print_freq == 0 or i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# ================= Checkpoint ===============
def save_checkpoint(model, optimizer, epoch, loss_scaler, checkpoint_path, checkpoint_name, args, stat=None):
    checkpoint_path = checkpoint_path + '/' + checkpoint_name
    stat_dict = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'stat': stat,
    }
    save_on_master(stat_dict, checkpoint_path)
    print(f"rank[{dist.get_rank()}]: {checkpoint_path} saved !!!")


def smart_partial_load_model_state_dict(model, state_dict, rename=True, remove_prefix=True):
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if remove_prefix and any([True if 'encoder.' in k else False for k in state_dict.keys()]):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    parsed_state_dict = {}
    non_match_keys = []
    pretrained_keys = []
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = state_dict[key]
            relative_position_bias_table_current = model.state_dict()[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if L1 != L2:
                print(f"{key}: Interpolate relative_position_bias_table using geo.")
                src_size = int(L1 ** 0.5)
                dst_size = int(L2 ** 0.5)

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(nH1):
                    z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                    f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                        relative_position_bias_table_pretrained.device))

                new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                state_dict[key] = new_rel_pos_bias

    if not rename:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete relative_coords_table since we always re-init it
        relative_coords_table_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
        for k in relative_coords_table_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

    for k, v in state_dict.items():
        if rename and 'relative_position_bias_table' in k:
            k = k.replace('relative_position_bias_table', 'rel_pos_embed_table')
        if rename and 'relative_position_index' in k:
            k = k.replace('relative_position_index', 'relative_coords')
        if rename and 'downsample.reduction' in k:
            k = k.replace('downsample.reduction', 'downsample.channel_reduction')
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v
            pretrained_keys.append(k)
        else:
            non_match_keys.append(k)
            # raise ValueError('failed to match key of state dict smartly!')

    non_pretrain_keys = [k for k in model.state_dict().keys() if
                         k not in pretrained_keys and 'num_batches_tracked' not in k]

    # print("[Partial Load] partial load state dict of keys: {}".format(parsed_state_dict.keys()))
    print("[Partial Load] non matched keys: {}".format(non_match_keys))
    print("WARNING! [Partial Load] non pretrain keys: {}".format(non_pretrain_keys))
    new_state_dict = model.state_dict()
    # covert the key-value in src model state dict to a new value from input state_dict (pretrained one)
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)


# ================= DDP ===============
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def safe_setenv(var, value):
    if var in os.environ:
        if os.environ[var] != value:
            print("WARNING: Environment variable '%s' already set to '%s', not changing to '%s'" % (
                var, os.environ[var], value))
        # return
    os.environ[var] = value


def get_global_rank():
    return int(os.environ['OMPI_COMM_WORLD_RANK'])


def get_local_rank():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def get_global_size():
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def get_local_size():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])


def get_world_size_itp():
    return int(os.environ['WORLD_SIZE'])


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    with torch.no_grad():
        out_list = [torch.ones_like(x)
                    for _ in range(dist.get_world_size())]
        dist.all_gather(out_list, x, async_op=False)
    out_list[int(os.environ['RANK'])] = x
    return torch.cat(out_list, dim=0)


class SyncFunction(torch.autograd.Function):
    # from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    # support to do global sim matrix or local sim matrix
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def dist_collect_list(x, device):
    x = torch.tensor(x).to(device)
    x = dist_collect(x)
    return x.cpu().tolist()


def varsize_dist_collect(tensor: torch.Tensor):
    # ref: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
    # ref: https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
    tensor = tensor.contiguous()

    size_tens = dist_collect_list([tensor.shape[0]], tensor.device)
    max_size = max(size_tens)

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=tensor.device)
    padded[:tensor.shape[0]] = tensor

    ag = dist_collect(padded)

    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)


def varsize_dist_collect_list(x, device):
    x = torch.tensor(x).to(device)
    x = varsize_dist_collect(x)
    return x.cpu().tolist()


def accumulate_list(x):
    acc_x = [0]
    for idx, num in enumerate(x):
        acc_x.append(acc_x[-1] + num)
    return acc_x


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, world_size, rank):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )