#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple, List
from torch import Tensor

from .tensor_utils import tensor_to_python_float



def metric_monitor( loss: Tensor or float, metric_names: list, v2lacc_r1: Optional[float]=None, l2vacc_r1: Optional[float]=None,
                   v2lacc_r5: Optional[float]=None, l2vacc_r5: Optional[float]=None,
                   v2lacc_r10: Optional[float]=None, l2vacc_r10: Optional[float]=None,
                   v2lacc_r128: Optional[float]=None, l2vacc_r128: Optional[float]=None,
                   use_distributed: Optional[bool] = False):

    metric_vals = dict()
    if "Loss" in metric_names:
        loss = tensor_to_python_float(loss, is_distributed=use_distributed)
        metric_vals['Loss'] = loss

    if 'V2LAcc_R1' in metric_names and v2lacc_r1 is not None:
        v2lacc_r1 = tensor_to_python_float(v2lacc_r1, is_distributed=use_distributed)
        metric_vals['V2LAcc_R1'] = v2lacc_r1

    if 'L2VAcc_R1' in metric_names and l2vacc_r1 is not None:
        l2vacc_r1 = tensor_to_python_float(l2vacc_r1, is_distributed=use_distributed)
        metric_vals['L2VAcc_R1'] = l2vacc_r1

    if 'V2LAcc_R5' in metric_names and v2lacc_r5 is not None:
        v2lacc_r5 = tensor_to_python_float(v2lacc_r5, is_distributed=use_distributed)
        metric_vals['V2LAcc_R5'] = v2lacc_r5

    if 'L2VAcc_R5' in metric_names and l2vacc_r5 is not None:
        l2vacc_r5 = tensor_to_python_float(l2vacc_r5, is_distributed=use_distributed)
        metric_vals['L2VAcc_R5'] = l2vacc_r5

    if 'V2LAcc_R10' in metric_names and v2lacc_r10 is not None:
        v2lacc_r10 = tensor_to_python_float(v2lacc_r10, is_distributed=use_distributed)
        metric_vals['V2LAcc_R10'] = v2lacc_r10

    if 'L2VAcc_R10' in metric_names and l2vacc_r10 is not None:
        l2vacc_r10 = tensor_to_python_float(l2vacc_r10, is_distributed=use_distributed)
        metric_vals['L2VAcc_R10'] = l2vacc_r10

    if 'V2LAcc_R128' in metric_names and v2lacc_r128 is not None:
        v2lacc_128 = tensor_to_python_float(v2lacc_r128, is_distributed=use_distributed)
        metric_vals['V2LAcc_R128'] = v2lacc_128

    if 'L2VAcc_R128' in metric_names and l2vacc_r128 is not None:
        l2vacc_128 = tensor_to_python_float(l2vacc_r128, is_distributed=use_distributed)
        metric_vals['L2VAcc_R128'] = l2vacc_128

    return metric_vals

def distill_metric_monitor( loss: Tensor or float, metric_names: list, tea_v2lacc_r1: Optional[float]=None, tea_l2vacc_r1: Optional[float]=None,
                   stu_v2lacc_r1: Optional[float]=None, stu_l2vacc_r1: Optional[float]=None,
                   tea_v2lacc_r5: Optional[float]=None, tea_l2vacc_r5: Optional[float]=None,
                   stu_v2lacc_r5: Optional[float]=None, stu_l2vacc_r5: Optional[float]=None,
                   tea_v2lacc_r10: Optional[float]=None, tea_l2vacc_r10: Optional[float]=None,
                   stu_v2lacc_r10: Optional[float]=None, stu_l2vacc_r10: Optional[float]=None,
                   tea_v2lacc_r128: Optional[float]=None, tea_l2vacc_r128: Optional[float]=None,
                   stu_v2lacc_r128: Optional[float]=None, stu_l2vacc_r128: Optional[float]=None,
                   use_distributed: Optional[bool] = False):

    metric_vals = dict()
    if "Loss" in metric_names:
        loss = tensor_to_python_float(loss, is_distributed=use_distributed)
        metric_vals['Loss'] = loss

    if 'TeaV2LAcc_R1' in metric_names and tea_v2lacc_r1 is not None:
        tea_v2lacc_r1 = tensor_to_python_float(tea_v2lacc_r1, is_distributed=use_distributed)
        metric_vals['TeaV2LAcc_R1'] = tea_v2lacc_r1

    if 'StuV2LAcc_R1' in metric_names and stu_v2lacc_r1 is not None:
        stu_v2lacc_r1 = tensor_to_python_float(stu_v2lacc_r1, is_distributed=use_distributed)
        metric_vals['StuV2LAcc_R1'] = stu_v2lacc_r1

    if 'TeaL2VAcc_R1' in metric_names and tea_l2vacc_r1 is not None:
        tea_l2vacc_r1 = tensor_to_python_float(tea_l2vacc_r1, is_distributed=use_distributed)
        metric_vals['TeaL2VAcc_R1'] = tea_l2vacc_r1

    if 'StuL2VAcc_R1' in metric_names and stu_l2vacc_r1 is not None:
        stu_l2vacc_r1 = tensor_to_python_float(stu_l2vacc_r1, is_distributed=use_distributed)
        metric_vals['StuL2VAcc_R1'] = stu_l2vacc_r1

    if 'TeaV2LAcc_R5' in metric_names and tea_v2lacc_r5 is not None:
        tea_v2lacc_r5 = tensor_to_python_float(tea_v2lacc_r5, is_distributed=use_distributed)
        metric_vals['TeaV2LAcc_R5'] = tea_v2lacc_r5

    if 'StuV2LAcc_R5' in metric_names and stu_v2lacc_r5 is not None:
        stu_v2lacc_r5 = tensor_to_python_float(stu_v2lacc_r5, is_distributed=use_distributed)
        metric_vals['StuV2LAcc_R5'] = stu_v2lacc_r5

    if 'TeaL2VAcc_R5' in metric_names and tea_l2vacc_r5 is not None:
        tea_l2vacc_r5 = tensor_to_python_float(tea_l2vacc_r5, is_distributed=use_distributed)
        metric_vals['TeaL2VAcc_R5'] = tea_l2vacc_r5

    if 'StuL2VAcc_R5' in metric_names and stu_l2vacc_r5 is not None:
        stu_l2vacc_r5 = tensor_to_python_float(stu_l2vacc_r5, is_distributed=use_distributed)
        metric_vals['StuL2VAcc_R5'] = stu_l2vacc_r5

    if 'TeaV2LAcc_R10' in metric_names and tea_v2lacc_r10 is not None:
        tea_v2lacc_r10 = tensor_to_python_float(tea_v2lacc_r10, is_distributed=use_distributed)
        metric_vals['TeaV2LAcc_R10'] = tea_v2lacc_r10

    if 'StuV2LAcc_R10' in metric_names and stu_v2lacc_r10 is not None:
        stu_v2lacc_r10 = tensor_to_python_float(stu_v2lacc_r10, is_distributed=use_distributed)
        metric_vals['StuV2LAcc_R10'] = stu_v2lacc_r10

    if 'TeaL2VAcc_R10' in metric_names and tea_l2vacc_r10 is not None:
        tea_l2vacc_r10 = tensor_to_python_float(tea_l2vacc_r10, is_distributed=use_distributed)
        metric_vals['TeaL2VAcc_R10'] = tea_l2vacc_r10

    if 'StuL2VAcc_R10' in metric_names and stu_l2vacc_r10 is not None:
        stu_l2vacc_r10 = tensor_to_python_float(stu_l2vacc_r10, is_distributed=use_distributed)
        metric_vals['StuL2VAcc_R10'] = stu_l2vacc_r10

    if 'TeaV2LAcc_R128' in metric_names and tea_v2lacc_r128 is not None:
        tea_v2lacc_128 = tensor_to_python_float(tea_v2lacc_r128, is_distributed=use_distributed)
        metric_vals['TeaV2LAcc_R128'] = tea_v2lacc_128

    if 'StuV2LAcc_R128' in metric_names and stu_v2lacc_r128 is not None:
        stu_v2lacc_128 = tensor_to_python_float(stu_v2lacc_r128, is_distributed=use_distributed)
        metric_vals['StuV2LAcc_R128'] = stu_v2lacc_128

    if 'TeaL2VAcc_R128' in metric_names and tea_l2vacc_r128 is not None:
        tea_l2vacc_128 = tensor_to_python_float(tea_l2vacc_r128, is_distributed=use_distributed)
        metric_vals['TeaL2VAcc_R128'] = tea_l2vacc_128

    if 'StuL2VAcc_R128' in metric_names and stu_l2vacc_r128 is not None:
        stu_l2vacc_128 = tensor_to_python_float(stu_l2vacc_r128, is_distributed=use_distributed)
        metric_vals['StuL2VAcc_R128'] = stu_l2vacc_128

    return metric_vals
