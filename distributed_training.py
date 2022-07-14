import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser('Distribute training script', add_help=False)
parser.add_argument('--base_dir', type=str, default='.')
parser.add_argument('--vlmodel-pretrain', type=str, default='')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--debug', action='store_false', default=True)
parser.add_argument('--debug_batch_size', type=int, default=64)
parser.add_argument('--val_num_samples', type=int, default=256)
parser.add_argument('--train_num_samples', type=int, default=256)
parser.add_argument('--stu-size', type=int, default=160)
parser.add_argument('--distill-model', type=str, default='swin_mini4')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--node_num', type=int, default=4)
args = parser.parse_args()

node_num = args.node_num
num_workers = int(4 * node_num)
num_workers = 0
lr = 5e-4
lr_scale = 1
lr_scale *= args.batch_size // 64

if args.debug:
    os.system("python ./launch.py \
                --nproc_per_node {} --master_port 2999 main.py \
                --dataset SStockDistill \
                --batch-size {} \
                --dataset_path {} \
                --accumulate_step 1 \
                --epochs 160 \
                --vlmodel-pretrain {} \
                --distill-model {} \
                --proj-size 256 \
                --backbone-lr-mult 1.0 \
                --backbone-lr-mult-l 1.0 \
                --layer-decay 1.0 \
                --layer-decay-l 1.0 \
                --fix-tau 0.05 \
                --stu-size {} \
                --mixup 0.0 \
                --cutmix 0.0 \
                --drop 0. \
                --drop-path 0.1 \
                --drop-att 0. \
                --weight-decay 0.05 \
                --lr {} \
                --warmup-lr 1e-6 \
                --min-lr 1e-6 \
                --warmup-epochs 0 \
                --warmup-steps 100 \
                --cooldown-epochs 0 \
                --output_dir {} \
                --num_workers {} \
                --debug_batch_size {} \
                --train_num_samples {} \
                --val_num_samples {} \
                --is_dist \
                ".format(node_num, args.batch_size, args.dataset_path, args.vlmodel_pretrain, args.distill_model, args.stu_size, lr*lr_scale, args.output_dir,  num_workers, args.debug_batch_size, args.train_num_samples, args.val_num_samples))
else:
    os.system("python ./launch.py \
                    --nproc_per_node {} --master_port 2999 main.py \
                    --dataset SStockDistill \
                    --batch-size {} \
                    --dataset_path {} \
                    --accumulate_step 1 \
                    --epochs 160 \
                    --vlmodel-pretrain {} \
                    --distill-model {} \
                    --proj-size 256 \
                    --backbone-lr-mult 1.0 \
                    --backbone-lr-mult-l 1.0 \
                    --layer-decay 1.0 \
                    --debug \
                    --layer-decay-l 1.0 \
                    --fix-tau 0.05 \
                    --stu-size {} \
                    --mixup 0.0 \
                    --cutmix 0.0 \
                    --drop 0. \
                    --drop-path 0.1 \
                    --drop-att 0. \
                    --weight-decay 0.05 \
                    --lr {} \
                    --warmup-lr 1e-6 \
                    --min-lr 1e-6 \
                    --warmup-epochs 0 \
                    --warmup-steps 100 \
                    --cooldown-epochs 0 \
                    --output_dir {} \
                    --num_workers {} \
                    --debug_batch_size {} \
                    --train_num_samples {} \
                    --val_num_samples {} \
                    --is_dist \
                    ".format(node_num, args.batch_size, args.dataset_path, args.vlmodel_pretrain, args.distill_model,
                             args.stu_size, lr * lr_scale, args.output_dir, num_workers, args.debug_batch_size,
                             args.train_num_samples, args.val_num_samples))
