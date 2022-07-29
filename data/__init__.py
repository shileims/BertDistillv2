from .datasets import Datasets
from .samplers import Samplers
from .collators import Collators
from .transforms import Transforms
from torch.utils.data import DataLoader

def create_train_data_loader(args, dataset, sampler, collator):
    is_train = True
    train_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    train_dataset = dataset(args.dataset_path, transforms=train_transforms, is_train=is_train, debug=args.debug)
    if args.debug:
        train_sampler = sampler(train_dataset, num_samples=args.train_num_samples)
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=collator(), batch_size=args.debug_batch_size, drop_last=True)
    else:
        train_sampler = sampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=collator(), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    return train_dataset, train_sampler, train_transforms, train_dataloader

def create_val_data_loader(args, dataset, sampler, collator):
    is_train = False
    val_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    val_dataset = dataset(args.dataset_path, transforms=val_transforms, is_train=is_train, debug=args.debug)

    if args.debug:
        val_sampler = sampler(val_dataset, num_samples=args.val_num_samples)
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_sampler, collate_fn=collator(), batch_size=args.debug_batch_size, drop_last=True)
    else:
        val_sampler = sampler(val_dataset)
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_sampler, collate_fn=collator(), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    return val_dataset, val_sampler, val_transforms, val_dataloader

def create_train_val_data_loader(args):
    dataset = Datasets.get(args.dataset)
    train_sampler = Samplers.get(args.train_sampler)
    val_sampler = Samplers.get(args.val_sampler)
    collator = Collators.get(args.collator)

    train_loader = create_train_data_loader(args, dataset, train_sampler, collator)
    val_loader   = create_val_data_loader(args, dataset, val_sampler, collator)
    return train_loader, val_loader, collator()

def create_distill_train_data_loader(args, dataset, sampler, collator):
    is_train = True
    train_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    train_dataset = dataset(args.dataset_path, transforms=train_transforms, tea_img_size=args.tea_img_size, stu_img_size=args.stu_img_size, is_train=is_train, fix_length=args.train_fix_length, debug=args.debug)
    if args.debug:
        if args.is_dist:
            try:
                train_sampler = sampler(train_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=True, num_samples=args.train_num_samples)
            except:
                train_sampler = sampler(train_dataset, num_samples=args.train_num_samples)
        else:
            train_sampler = sampler(train_dataset, num_samples=args.train_num_samples)
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=collator(), batch_size=args.debug_batch_size, drop_last=False)
    else:
        if args.is_dist:
            try:
                train_sampler = sampler(train_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=False)
            except:
                train_sampler = sampler(train_dataset)
        else:
            train_sampler = sampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, collate_fn=collator(), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    return train_dataset, train_sampler, train_transforms, train_dataloader

def create_distill_val_data_loader(args, dataset, sampler, collator):
    is_train = False
    val_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    val_dataset = dataset(args.dataset_path, transforms=val_transforms, tea_img_size=args.tea_img_size, stu_img_size=args.stu_img_size, is_train=is_train, fix_length=args.val_fix_length, debug=args.debug)

    if args.debug:
        if args.is_dist:
            try:
                val_sampler = sampler(val_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=True, num_samples=args.val_num_samples)
            except:
                val_sampler = sampler(val_dataset, num_samples=args.val_num_samples)
        else:
            val_sampler = sampler(val_dataset, num_samples=args.val_num_samples)
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_sampler, collate_fn=collator(), batch_size=args.debug_batch_size, drop_last=False)
    else:
        if args.is_dist:
            try:
                val_sampler = sampler(val_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=False)
            except:
                val_sampler = sampler(val_dataset)
        else:
            val_sampler = sampler(val_dataset)
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_sampler, collate_fn=collator(), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    return val_dataset, val_sampler, val_transforms, val_dataloader

def create_distill_train_val_data_loader(args):
    dataset = Datasets.get(args.dataset)
    train_sampler = Samplers.get(args.train_sampler)
    val_sampler = Samplers.get(args.val_sampler)
    collator = Collators.get(args.collator)

    train_loader = create_distill_train_data_loader(args, dataset, train_sampler, collator)
    val_loader   = create_distill_val_data_loader(args, dataset, val_sampler, collator)
    return train_loader, val_loader, collator()

def _create_distill_train_data_loader(args):
    dataset = Datasets.get(args.dataset)
    train_sampler = Samplers.get(args.train_sampler)
    collator = Collators.get(args.collator)

    train_loader = create_distill_train_data_loader(args, dataset, train_sampler, collator)
    return train_loader


def create_distill_val_dataset(args, dataset):
    is_train = False
    val_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    val_dataset = dataset(args.dataset_path, transforms=val_transforms, tea_img_size=args.tea_size,
                          stu_img_size=args.stu_size, is_train=is_train, debug=args.debug)
    return val_dataset


def create_distill_train_dataset(args, dataset):
    is_train = True
    train_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    train_dataset = dataset(args.dataset_path, transforms=train_transforms, tea_img_size=args.tea_size,
                            stu_img_size=args.stu_size, is_train=is_train, debug=args.debug)
    return train_dataset

def create_distill_train_val_datasets(args):
    dataset = Datasets.get(args.dataset)
    train_dataset = create_distill_train_dataset(args, dataset)
    val_dataset   = create_distill_val_dataset(args, dataset)
    return train_dataset, val_dataset


def create_distill_eval_loader(args):
    dataset = Datasets.get(args.dataset)
    sampler = Samplers.get(args.sampler)
    collator = Collators.get(args.collator)
    is_train = False
    val_transforms = Transforms.get(args.transforms)(args, is_train=is_train)
    print(dataset)
    dataset = dataset(args.dataset_path, transforms=val_transforms, tea_img_size=args.tea_img_size, stu_img_size=args.stu_img_size)
    if args.debug:
        sampler = sampler(dataset, num_samples=args.num_samples)
        dataloader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=collator(), batch_size=args.debug_batch_size, drop_last=True)
    else:
        sampler = sampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=collator(), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    return dataloader
