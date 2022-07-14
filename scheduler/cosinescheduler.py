from .register import Schedulers
import numpy as np
import math

@Schedulers.register_module
class CosineScheduler(object):
    def __init__(self, args, data_loader, warmup_epochs=0, warmup_steps=-1):
        base_value = args.lr
        final_value = args.min_lr
        epochs = args.epochs
        niter_per_ep = len(data_loader) // args.accumulate_step
        start_warmup_value = args.warmup_lr
        warmup_steps = args.warmup_steps

        warmup_schedule = np.array([])
        warmup_iters = args.warmup_epochs * niter_per_ep
        if warmup_steps > 0:
            warmup_iters = warmup_steps
        print("Set warmup steps = %d" % warmup_iters)
        if warmup_iters > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array(
            [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in
             iters])

        schedule = np.concatenate((warmup_schedule, schedule))

        assert len(
            schedule) == epochs * niter_per_ep, f"scheduler: {len(schedule)}; total step: {epochs * niter_per_ep}"
        self.scheduler = schedule

    def get_lr(self, curr_iter: int) -> float:
        return self.scheduler[curr_iter]


    def update_lr(self, optimizer, dataloader_length, epoch: int, curr_iter: int):
        global_idx = epoch * dataloader_length + curr_iter
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.scheduler[global_idx] * \
                                param_group["lr_scale"]
        return optimizer

    def distill_update_lr(self, optimizer, dataloader_length, epoch: int, curr_iter: int):
        global_idx = epoch * dataloader_length + curr_iter
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.scheduler[global_idx]
        return optimizer

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list
