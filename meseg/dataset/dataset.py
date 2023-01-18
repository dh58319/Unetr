import json
import os

from torch.utils.data import DataLoader
from .data_utils.transforms import TrainTransform, ValTransform


_dataset_dict = {}

def register_dataset(fn):
    dataset_name = fn.__name__
    if dataset_name not in _dataset_dict:
        _dataset_dict[fn.__name__] = fn
    else:
        raise ValueError(f"{dataset_name} already exists in dataset_dict")

    return fn


def get_dataset(args, mode):
    if mode == 'train':
        # define datasets
        dataset_class = _dataset_dict[args.dataset_type]
        if args.dataset_type in _dataset_dict.keys():
            train_dataset = dataset_class(
                root=args.data_dir,
                mode='train',
            )
            val_dataset = dataset_class(
                root=args.data_dir,
                mode='valid',
            )
            # args.num_classes = 1 if train_dataset.task == 'binary' else len(train_dataset.classes)
            # args.task = train_dataset.task
            # args.num_labels = train_dataset.num_labels
            args.num_classes = train_dataset.classes
        else:
            assert f"{args.dataset_type} is not supported yet. Just make your own code for it"
        return train_dataset, val_dataset

    else:
        # define datasets
        dataset_class = _dataset_dict[args.dataset_type]
        if args.dataset_type in _dataset_dict.keys():
            val_dataset = dataset_class(
                root=args.data_dir,
                mode='valid',
            )
            # args.num_classes = 1 if val_dataset.task == 'binary' else len(val_dataset.classes)
            # args.task = val_dataset.task
            # args.num_labels = val_dataset.num_labels
            args.num_classes = train_dataset.classes
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=None, pin_memory=False)
        else:
            assert f"{args.dataset_type} is not supported yet. Just make your own code for it"
        return val_dataset, val_dataloader

