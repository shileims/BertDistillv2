import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--modeldir', type=str, default='')
parser.add_argument('--experiment_name', type=str, default='distill_refactor_2M_0724_eval')
parser.add_argument('--eval_model_dir', type=str, default='distill_refactor_dist_2M_32e_0726')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dataset-path', type=str, default='cocoir')
parser.add_argument('--debug', action='store_false', default=False)

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

    dataset_path = os.path.join(args.basedir, args.dataset_path)

    # dataset_path = args.basedir

    if args.tag == 'swin_mini4':
        stu_size = 192
        distill_model = 'swin_mini4'
        eval_model_dir = os.path.join(args.eval_model_dir, 'swin_mini4', 'ckpts')
    elif args.tag == 'swin_mini7':
        stu_size = 160
        distill_model = 'swin_mini7'
        eval_model_dir = os.path.join(args.eval_model_dir, 'swin_mini7', 'ckpts')
    elif args.tag == 'swin_mini1':
        stu_size = 224
        distill_model = 'swin_mini1'
        eval_model_dir = os.path.join(args.eval_model_dir, 'swin_mini1', 'ckpts')
    else:
        raise NotImplementedError

    if not args.modeldir:
        output_dir = Path(args.basedir) / args.experiment_name / distill_model
    else:
        output_dir = Path(args.modeldir) / args.experiment_name / distill_model
    output_dir = str(output_dir)

    eval_model_dir = os.path.join(args.basedir, eval_model_dir)
#    os.system("bash mount_data.sh")
    if not args.debug:
        os.system("python evaluate.py \
                    --vlmodel-pretrain {} \
                    --batch-size {} \
                    --dataset_path {} \
                    --debug \
                    --debug_batch_size {} \
                    --stu-img-size {} \
                    --distill-model {} \
                    --eval_model_dir {} ".format(vlmodel_pretrain, args.batch_size, dataset_path, args.debug_batch_size,  stu_size, distill_model, eval_model_dir))
    else:
        os.system("python evaluate.py \
                            --vlmodel-pretrain {} \
                            --batch-size {} \
                            --dataset_path {} \
                            --debug_batch_size {} \
                            --stu-img-size {} \
                            --distill-model {} \
                            --eval_model_dir {}".format(vlmodel_pretrain, args.batch_size, dataset_path,
                                                    args.debug_batch_size,
                                                    stu_size, distill_model, eval_model_dir))
