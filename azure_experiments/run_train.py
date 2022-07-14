import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--common.config-file', type=str, default='config/classification/swinmini7.yaml')
parser.add_argument('--common.results-loc', type=str, default='results_swinmini7')
parser.add_argument('--train_dir', type=str, default='ILSVRC2012_img_train')
parser.add_argument('--val_dir', type=str, default='ILSVRC2012_img_val')
parser.add_argument('--root_path', type=str, default='.')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()


if __name__ == '__main__':
    print(f"AML cmd printout: {args}")
    os.environ['PYTHONPATH'] = args.root_path
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    train_dir = Path(args.basedir) / args.train_dir
    val_dir   = Path(args.basedir) / args.val_dir
    results_loc = Path(args.basedir) / getattr(args, 'common.results_loc')
    cmd3 = '''python "azure_main_train.py" --common.config-file {} --common.results-loc {} --dataset.extra_root_train {} --dataset.extra_root_val {}'''.format(getattr(args, 'common.config_file'), results_loc, train_dir, val_dir)

    os.system(cmd3)
