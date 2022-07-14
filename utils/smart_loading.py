import numpy as np
import torch
from scipy import interpolate

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

    non_pretrain_keys = [k for k in model.state_dict().keys() if k not in pretrained_keys and 'num_batches_tracked' not in k]

    # print("[Partial Load] partial load state dict of keys: {}".format(parsed_state_dict.keys()))
    print("[Partial Load] non matched keys: {}".format(non_match_keys))
    print("WARNING! [Partial Load] non pretrain keys: {}".format(non_pretrain_keys))
    new_state_dict = model.state_dict()
    # covert the key-value in src model state dict to a new value from input state_dict (pretrained one)
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)