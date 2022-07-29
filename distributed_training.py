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
parser.add_argument('--stu-img-size', type=int, default=160)
parser.add_argument('--distill-model', type=str, default='swin_mini4')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--node_num', type=int, default=8)
parser.add_argument('--total_data', type=int, default=2e6)
parser.add_argument('--epochs', type=int, default=16)
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--node_index', type=int, default=0)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--master_addr', type=str, default="10.0.0.4")
parser.add_argument('--master_port', type=str, default='50000')
args = parser.parse_args()

node_num = args.node_num
num_workers = int(node_num)

lr = 5e-4
lr_scale = 1
lr_scale *= args.batch_size * node_num / 8 / 256

if 'SS2M' in args.dataset_path:
    os.system("./azcopy copy 'https://lei2021.blob.core.windows.net/swintransori/SS2M?sp=racwdli&st=2022-07-22T03:00:10Z&se=2022-08-01T11:00:10Z&sv=2021-06-08&sr=c&sig=lkCl%2BT4%2FLQBNRr4V5%2B9ogMhCZ4bbXCqFeThJqDqVKiY%3D' './' --recursive")
    args.dataset_path = 'SS2M'

if args.nodes == 1:
    if args.debug:
        if args.resume:
            os.system("python ./launch.py \
                        --nproc_per_node {} --master_port 2999 main.py \
                        --dataset SStockDistill \
                        --batch-size {} \
                        --dataset_path {} \
                        --accumulate_step 1 \
                        --epochs {} \
                        --vlmodel-pretrain {} \
                        --distill-model {} \
                        --proj-size 256 \
                        --backbone-lr-mult 1.0 \
                        --backbone-lr-mult-l 1.0 \
                        --layer-decay 1.0 \
                        --layer-decay-l 1.0 \
                        --fix-tau 0.05 \
                        --stu-img-size {} \
                        --mixup 0.0 \
                        --cutmix 0.0 \
                        --drop 0. \
                        --drop-path 0.1 \
                        --drop-att 0. \
                        --weight-decay 0.05 \
                        --lr {} \
                        --warmup-lr 1e-5 \
                        --min-lr 1e-6 \
                        --warmup-epochs 0 \
                        --warmup-steps 100 \
                        --cooldown-epochs 0 \
                        --output_dir {} \
                        --num_workers {} \
                        --debug_batch_size {} \
                        --train_num_samples {} \
                        --val_num_samples {} \
                        --train_fix_length {} \
                        --resume {} \
                        --is_dist \
                        ".format(node_num, args.batch_size, args.dataset_path, args.epochs,
                                 args.vlmodel_pretrain, args.distill_model, args.stu_img_size,
                                 lr*lr_scale, args.output_dir,  num_workers, args.debug_batch_size,
                                 args.train_num_samples, args.val_num_samples, args.total_data, args.resume))
        else:
            os.system("python ./launch.py \
                                --nproc_per_node {} --master_port 2999 main.py \
                                --dataset SStockDistill \
                                --batch-size {} \
                                --dataset_path {} \
                                --accumulate_step 1 \
                                --epochs {} \
                                --vlmodel-pretrain {} \
                                --distill-model {} \
                                --proj-size 256 \
                                --backbone-lr-mult 1.0 \
                                --backbone-lr-mult-l 1.0 \
                                --layer-decay 1.0 \
                                --layer-decay-l 1.0 \
                                --fix-tau 0.05 \
                                --stu-img-size {} \
                                --mixup 0.0 \
                                --cutmix 0.0 \
                                --drop 0. \
                                --drop-path 0.1 \
                                --drop-att 0. \
                                --weight-decay 0.05 \
                                --lr {} \
                                --warmup-lr 1e-5 \
                                --min-lr 1e-6 \
                                --warmup-epochs 0 \
                                --warmup-steps 100 \
                                --cooldown-epochs 0 \
                                --output_dir {} \
                                --num_workers {} \
                                --debug_batch_size {} \
                                --train_num_samples {} \
                                --val_num_samples {} \
                                --train_fix_length {} \
                                --is_dist \
                                ".format(node_num, args.batch_size, args.dataset_path, args.epochs,
                                         args.vlmodel_pretrain, args.distill_model, args.stu_img_size, lr * lr_scale,
                                         args.output_dir, num_workers, args.debug_batch_size, args.train_num_samples,
                                         args.val_num_samples, args.total_data))
    else:
        if args.resume:
            os.system("python ./launch.py \
                            --nproc_per_node {} --master_port 2999 main.py \
                            --dataset SStockDistill \
                            --batch-size {} \
                            --dataset_path {} \
                            --accumulate_step 1 \
                            --epochs {} \
                            --vlmodel-pretrain {} \
                            --distill-model {} \
                            --proj-size 256 \
                            --backbone-lr-mult 1.0 \
                            --backbone-lr-mult-l 1.0 \
                            --layer-decay 1.0 \
                            --debug \
                            --layer-decay-l 1.0 \
                            --fix-tau 0.05 \
                            --stu-img-size {} \
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
                            --train_fix_length {} \
                            --resume {} \
                            --is_dist \
                            ".format(node_num, args.batch_size, args.dataset_path, args.epochs, args.vlmodel_pretrain, args.distill_model,
                                     args.stu_img_size, lr * lr_scale, args.output_dir, num_workers, args.debug_batch_size,
                                     args.train_num_samples, args.val_num_samples, args.total_data, args.resume))
        else:
            os.system("python ./launch.py \
                                        --nproc_per_node {} --master_port 2999 main.py \
                                        --dataset SStockDistill \
                                        --batch-size {} \
                                        --dataset_path {} \
                                        --accumulate_step 1 \
                                        --epochs {} \
                                        --vlmodel-pretrain {} \
                                        --distill-model {} \
                                        --proj-size 256 \
                                        --backbone-lr-mult 1.0 \
                                        --backbone-lr-mult-l 1.0 \
                                        --layer-decay 1.0 \
                                        --debug \
                                        --layer-decay-l 1.0 \
                                        --fix-tau 0.05 \
                                        --stu-img-size {} \
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
                                        --train_fix_length {} \
                                        --is_dist \
                                        ".format(node_num, args.batch_size, args.dataset_path, args.epochs,
                                                 args.vlmodel_pretrain, args.distill_model,
                                                 args.stu_img_size, lr * lr_scale, args.output_dir, num_workers,
                                                 args.debug_batch_size,
                                                 args.train_num_samples, args.val_num_samples, args.total_data))
else:
    # muti-node distributed training
    if args.debug:
        if args.resume:
            os.system("python ./launch.py \
                        --nproc_per_node {} --nnodes {} --node_rank {} --master_addr {} --master_port {} main.py \
                        --dataset SStockDistill \
                        --batch-size {} \
                        --dataset_path {} \
                        --accumulate_step 1 \
                        --epochs {} \
                        --vlmodel-pretrain {} \
                        --distill-model {} \
                        --proj-size 256 \
                        --backbone-lr-mult 1.0 \
                        --backbone-lr-mult-l 1.0 \
                        --layer-decay 1.0 \
                        --layer-decay-l 1.0 \
                        --fix-tau 0.05 \
                        --stu-img-size {} \
                        --mixup 0.0 \
                        --cutmix 0.0 \
                        --drop 0. \
                        --drop-path 0.1 \
                        --drop-att 0. \
                        --weight-decay 0.05 \
                        --lr {} \
                        --warmup-lr 1e-5 \
                        --min-lr 1e-6 \
                        --warmup-epochs 0 \
                        --warmup-steps 100 \
                        --cooldown-epochs 0 \
                        --output_dir {} \
                        --num_workers {} \
                        --debug_batch_size {} \
                        --train_num_samples {} \
                        --val_num_samples {} \
                        --train_fix_length {} \
                        --resume {} \
                        --is_dist \
                        ".format(node_num, args.nodes, args.node_index, args.master_addr, args.master_port, args.batch_size, args.dataset_path, args.epochs, args.vlmodel_pretrain, args.distill_model, args.stu_img_size, lr*lr_scale, args.output_dir,  num_workers, args.debug_batch_size, args.train_num_samples, args.val_num_samples, args.total_data, args.resume))
        else:
            os.system("python ./launch.py \
                                --nproc_per_node {} --nnodes {} --node_rank {} --master_addr {} --master_port {} main.py \
                                --dataset SStockDistill \
                                --batch-size {} \
                                --dataset_path {} \
                                --accumulate_step 1 \
                                --epochs {} \
                                --vlmodel-pretrain {} \
                                --distill-model {} \
                                --proj-size 256 \
                                --backbone-lr-mult 1.0 \
                                --backbone-lr-mult-l 1.0 \
                                --layer-decay 1.0 \
                                --layer-decay-l 1.0 \
                                --fix-tau 0.05 \
                                --stu-img-size {} \
                                --mixup 0.0 \
                                --cutmix 0.0 \
                                --drop 0. \
                                --drop-path 0.1 \
                                --drop-att 0. \
                                --weight-decay 0.05 \
                                --lr {} \
                                --warmup-lr 1e-5 \
                                --min-lr 1e-6 \
                                --warmup-epochs 0 \
                                --warmup-steps 100 \
                                --cooldown-epochs 0 \
                                --output_dir {} \
                                --num_workers {} \
                                --debug_batch_size {} \
                                --train_num_samples {} \
                                --val_num_samples {} \
                                --train_fix_length {} \
                                --is_dist \
                                ".format(node_num, args.nodes, args.node_index, args.master_addr, args.master_port, args.batch_size, args.dataset_path,
                                         args.epochs, args.vlmodel_pretrain, args.distill_model, args.stu_img_size,
                                         lr * lr_scale, args.output_dir, num_workers, args.debug_batch_size,
                                         args.train_num_samples, args.val_num_samples, args.total_data))
    else:
        if args.resume:
            os.system("python ./launch.py \
                            --nproc_per_node {} --nnodes {} --node_rank {} --master_addr {} --master_port {} main.py \
                            --dataset SStockDistill \
                            --batch-size {} \
                            --dataset_path {} \
                            --accumulate_step 1 \
                            --epochs {} \
                            --vlmodel-pretrain {} \
                            --distill-model {} \
                            --proj-size 256 \
                            --backbone-lr-mult 1.0 \
                            --backbone-lr-mult-l 1.0 \
                            --layer-decay 1.0 \
                            --debug \
                            --layer-decay-l 1.0 \
                            --fix-tau 0.05 \
                            --stu-img-size {} \
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
                            --train_fix_length {} \
                            --resume {} \
                            --is_dist \
                            ".format(node_num, args.nodes, args.node_index, args.master_addr, args.master_port, args.batch_size, args.dataset_path, args.epochs, args.vlmodel_pretrain, args.distill_model,
                                     args.stu_img_size, lr * lr_scale, args.output_dir, num_workers, args.debug_batch_size,
                                     args.train_num_samples, args.val_num_samples, args.total_data, args.resume))
        else:
            os.system("python ./launch.py \
                                        --nproc_per_node {} --nnodes {} --node_rank {} --master_addr {} --master_port {} main.py \
                                        --dataset SStockDistill \
                                        --batch-size {} \
                                        --dataset_path {} \
                                        --accumulate_step 1 \
                                        --epochs {} \
                                        --vlmodel-pretrain {} \
                                        --distill-model {} \
                                        --proj-size 256 \
                                        --backbone-lr-mult 1.0 \
                                        --backbone-lr-mult-l 1.0 \
                                        --layer-decay 1.0 \
                                        --debug \
                                        --layer-decay-l 1.0 \
                                        --fix-tau 0.05 \
                                        --stu-img-size {} \
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
                                        --train_fix_length {} \
                                        --is_dist \
                                        ".format(node_num, args.nodes, args.node_index, args.master_addr, args.master_port, args.batch_size,
                                                 args.dataset_path, args.epochs, args.vlmodel_pretrain,
                                                 args.distill_model,
                                                 args.stu_img_size, lr * lr_scale, args.output_dir, num_workers,
                                                 args.debug_batch_size,
                                                 args.train_num_samples, args.val_num_samples, args.total_data
                                                 ))
