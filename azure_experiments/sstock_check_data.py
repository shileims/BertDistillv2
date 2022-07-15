import os
import argparse
import json
from pathlib import Path

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
args = parser.parse_args()


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

    full_info = []
    for file in os.listdir(dataset_path):
        if file.endswith('.json'):
            print(f'------------------------------------json file: {file}')
            _full_info = json.load(
                open(os.path.join(dataset_path, file)))
            vs = list(_full_info.values())
            print(f'------------------------------------length: {len(vs)}')
            full_info.extend(vs)
    print(f'********************************************In total: {len(full_info)}')


