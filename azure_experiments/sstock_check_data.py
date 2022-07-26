import os
import argparse
import json
import re
import time

import PIL.Image
from PIL import Image
import io
import base64
from pathlib import Path

"""
------------------------------------json file: split_0.json
------------------------------------length: 1532660
------------------------------------json file: split_1.json
------------------------------------length: 1530192
------------------------------------json file: split_10.json
------------------------------------length: 1536753
------------------------------------json file: split_11.json
------------------------------------length: 1491425
------------------------------------json file: split_12.json
------------------------------------length: 1490977
------------------------------------json file: split_13.json
------------------------------------length: 1483257
------------------------------------json file: split_2.json
------------------------------------length: 1537870
------------------------------------json file: split_3.json
------------------------------------length: 1512045
********************************************In total: 12115179
"""

parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--modeldir', type=str, default='')
parser.add_argument('--experiment_name', type=str, default='distill_refactor_1.4M')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dataset-path', type=str, default='SS200M')
parser.add_argument('--debug', action='store_false', default=False)
parser.add_argument('--train_num_samples', type=int, default=512)
parser.add_argument('--val_num_samples', type=int, default=256)
parser.add_argument('--debug_batch_size', type=int, default=64)
parser.add_argument('--save_folder', type=str, default='SS12M')
args = parser.parse_args()


def deconfusing(string):
    confuse_head = r'this is a head'.encode('utf-8')
    if string.startswith(confuse_head):
        confuse_code = b'\xff\xdb\x00C\x00\x02\x01'
        string = string[len(confuse_head):]
        result = re.search(b'\xff\xda', string)
        startofscan = result.span()[0]
        return string[:startofscan - len(confuse_code)] + string[startofscan:]
    else:
        # no confusing for Laion dataset
        return string

def _load_image(path):
    assert '.tsv/' in path, f'tsv not in {path}'
    tsv_name, lineidx = path.split('.tsv/')
    _fp = open(tsv_name + '.tsv', 'r')
    _fp.seek(int(lineidx))
    _, img = [s.strip() for s in _fp.readline().split('\t')]
    # if len(xx) == 2:
    #     _, img = xx
    # elif len(xx) == 1:
    #     img = xx[0]
    # else:
    #     raise NotImplementedError
    img = base64.b64decode(img)
    img = deconfusing(img)
    img = Image.open(io.BytesIO(img))
    _fp.close()
    img = img.convert("RGB")
    last_img = img
    return img, tsv_name, lineidx

    # try:
    #     tsv_name, lineidx = path.split('.tsv/')
    #     _fp = open(tsv_name + '.tsv', 'r')
    #     _fp.seek(int(lineidx))
    #     _, img = [s.strip() for s in _fp.readline().split('\t')]
    #     img = base64.b64decode(img)
    #     img = deconfusing(img)
    #     img = Image.open(io.BytesIO(img))
    #     _fp.close()
    #     img = img.convert("RGB")
    #     last_img = img
    #     return img, tsv_name, lineidx
    # except Exception as e:
    #     print("ERROR IMG (.tsv) LOADED: ", path, e)
    #     return None, None, None

def getitem(item, database):
    idb = database[item]
    # images
    raw_img, tsv_name, lineidx = _load_image(idb[1])
    return raw_img, tsv_name, lineidx

if __name__ == '__main__':
    print(f"AML cmd printout: {args}")
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    database_dir = args.basedir
    distill_model = args.tag
    output_dir = args.experiment_name
    debug = args.debug
    if not args.modeldir:
        vlmodel_pretrain = os.path.join(args.basedir, 'ss200m_vl_for_distill.pth')
    else:
        vlmodel_pretrain = os.path.join(args.modeldir, 'ss200m_vl_for_distill.pth')

    if not args.modeldir:
        if args.debug:
            dataset_name = args.dataset_path + '_split_0'
        else:
            dataset_name = args.dataset_path
        dataset_path = os.path.join(args.basedir, dataset_name)
    else:
        dataset_path = args.basedir

    database = []
    full_info = []
    total = 0
    for file in os.listdir(dataset_path):
        if file.endswith('.json'):
            print(f'------------------------------------json file: {file}')
            _full_info = json.load(
                open(os.path.join(dataset_path, file)))
            vs = list(_full_info.values())
            print(f'------------------------------------length: {len(vs)}')
            # full_info.extend(vs)
            total += len(vs)
            full_info.append(vs)
    print(f'********************************************In total: {total}')
    print(f'{len(full_info)}')
    full_info = full_info[1]
    for info in full_info:
        database.append([info['img_caption'], os.path.join(dataset_path,
                                                                f'split_{info["img_location"]}.tsv/{info["lineidx_ptr"]}')])

    num = 79370
    for i in range(len(database)):
        if i < num:
            continue
        print(f'Coping with {i}')
        img, tsv_name, lineidx = getitem(i, database)
        if img is None or tsv_name is None or lineidx is None:
            continue
        tsv_folder = tsv_name.strip().split('/')[-1]
        img_name = tsv_folder + '_' + lineidx + '.jpg'
        save_path = Path(args.basedir + '/' + args.save_folder + '/' + tsv_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        img_path = save_path / img_name
        img.save(str(img_path))
        print(f'Saving image to {img_path}')


