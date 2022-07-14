import numpy as np
import torch
from torch import distributed as dist
from typing import Union




def reduce_tensor(inp_tensor: torch.Tensor) -> torch.Tensor:
    size = float(dist.get_world_size())
    inp_tensor_clone = inp_tensor.clone()
    dist.barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def tensor_to_python_float(inp_tensor: Union[int, float, torch.Tensor],
                           is_distributed: bool) -> Union[int, float, np.ndarray]:
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        inp_tensor = reduce_tensor(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, 'item'):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError("The data type is not supported yet in tensor_to_python_float function")



def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np