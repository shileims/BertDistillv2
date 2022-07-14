from .register import Schedulers
from .cosinescheduler import CosineScheduler


def create_scheduler(args, data_loader):
    scheduler = Schedulers.get(args.scheduler)(args, data_loader, args.warmup_epochs)
    return scheduler

