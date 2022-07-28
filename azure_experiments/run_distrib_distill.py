import os
import argparse
from pathlib import Path

debug_flag = {'True': True, 'False': False}

parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--modeldir', type=str, default='')
parser.add_argument('--experiment_name', type=str, default='distill_refactor_2M')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--node_num', type=int, default=8)
parser.add_argument('--total_data', type=int, default=2000000)
parser.add_argument('--epochs', type=int, default=16)
parser.add_argument('--debug', type=str, default='False')
parser.add_argument('--train_num_samples', type=int, default=2048)
parser.add_argument('--val_num_samples', type=int, default=1024)
parser.add_argument('--debug_batch_size', type=int, default=64)
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--node_index', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    print(f"AML cmd printout: {args}")
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    database_dir = args.basedir
    if not args.modeldir:
        modeldir = args.basedir
    else:
        modeldir = args.modeldir

    distill_model = args.tag
    output_dir = args.experiment_name
    debug = args.debug

    vlmodel_pretrain = os.path.join(modeldir, 'ss200m_vl_for_distill.pth')

    if args.total_data <= 2e6:
        dataset_path = 'SS2M'
    else:
        dataset_path = 'SS200M/version1'

    dataset_path = os.path.join(args.basedir, dataset_path)

    if args.tag == 'swin_mini4':
        stu_size = 192
        distill_model = 'swin_mini4'
    elif args.tag == 'swin_mini7':
        stu_size = 160
        distill_model = 'swin_mini7'
    elif args.tag == 'swin_mini1':
        stu_size = 224
        distill_model = 'swin_mini1'
    else:
        raise NotImplementedError

    output_dir = Path(args.basedir) / output_dir / distill_model
    output_dir = str(output_dir)

    if not debug_flag[args.debug]:
        os.system("python distributed_training.py \
                    --vlmodel-pretrain {} \
                    --batch-size {} \
                    --dataset_path {} \
                    --debug \
                    --debug_batch_size {} \
                    --val_num_samples {} \
                    --train_num_samples {} \
                    --stu-img-size {} \
                    --distill-model {} \
                    --output_dir {} \
                    --node_num {} \
                    --total_data {} \
                    --epochs {} \
                    --nodes {} \
                    --node_index {}".format(vlmodel_pretrain, args.batch_size, dataset_path, args.debug_batch_size, args.val_num_samples, args.train_num_samples, stu_size, distill_model, output_dir, args.node_num, args.total_data, args.epochs, args.nodes, args.node_index))
    else:
        os.system("python distributed_training.py \
                    --vlmodel-pretrain {} \
                    --batch-size {} \
                    --dataset_path {} \
                    --debug_batch_size {} \
                    --val_num_samples {} \
                    --train_num_samples {} \
                    --stu-img-size {} \
                    --distill-model {} \
                    --output_dir {} \
                    --node_num {} \
                    --total_data {} \
                    --epochs {} \
                    --nodes {} \
                    --node_index {}".format(vlmodel_pretrain, args.batch_size, dataset_path, args.debug_batch_size, args.val_num_samples, args.train_num_samples, stu_size, distill_model, output_dir, args.node_num, args.total_data, args.epochs, args.nodes, args.node_index))
