import os
from pathlib import Path
from .registry import Registry
from .device_setting import device_initialize
from .smart_loading import smart_partial_load_model_state_dict
from .syncfunc import SyncFunction
from .functional_tensor import resize
from .layerdecay import LayerDecayValueAssigner
from .metric_logger import MetricLogger, SmoothedValue
from .save_ckpt import save_checkpoint, load_checkpoint, load_weights
from .logging import logger
from .stats import Statistics
from .metricmonitor import metric_monitor, distill_metric_monitor
from .tensor_utils import tensor_to_python_float
from .distributed_training import init_distributed_mode, get_rank, get_world_size, is_main_process, dist_collect
from .argmax_func import ArgMax


def create_output_dir(args):
    assert args.output_dir != '', f'Please specify output dir!'
    assert os.path.isdir(args.dataset_path), f'{args.dataset_path} does not exist!'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.log(f'Create output dir! ： {args.output_dir}')
