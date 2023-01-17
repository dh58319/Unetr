import os

import torch
from torch import distributed as dist
import torchmetrics.functional as TMF


class Metric:
    def __init__(self, reduce_every_n_step=50, reduce_on_compute=True, header='', fmt='{val:.4f} ({avg:.4f})'):
        """Base Metric Class supporting ddp setup
        :arg
            reduce_ever_n_step(int): call all_reduce every n step in ddp mode
            reduce_on_compute(bool): call all_reduce in compute() method
            fmt(str): format representing metric in string
        """
        self.dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.dist:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.reduce_every_n_step = reduce_every_n_step
            self.reduce_on_compute = reduce_on_compute
        else:
            self.world_size = None
            self.reduce_every_n_step = self.reduce_on_compute = False

        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        elif self.reduce_every_n_step and not isinstance(val, torch.Tensor):
            raise ValueError('reduce operation is allowed for only tensor')

        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

        if self.reduce_every_n_step and self.n % self.reduce_every_n_step == 0:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

    def compute(self):
        if self.reduce_on_compute:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)


def Accuracy(y_hat, y, top_k=(1,)):
    """Compute top-k accuracy
    :arg
        y_hat(tensor): prediction shaped as (B, C)
        y(tensor): label shaped as (B)
        top_k(tuple): how exactly model should predict in each metric
    :return
        list of metric scores
    """
    prediction = torch.argsort(y_hat, dim=-1, descending=True)
    accuracy = [(prediction[:, :min(k, y_hat.size(1))] == y.unsqueeze(-1)).float().sum(dim=-1).mean() * 100 for k in top_k]
    return accuracy


def compute_metrics(preds, labels, args):
    if isinstance(preds, (list, tuple)):
        preds = torch.concat(preds, dim=0).detach().clone()
        labels = torch.concat(labels, dim=0).detach().clone()

    preds = all_gather_with_different_size(preds)
    labels = all_gather_with_different_size(labels)

    if args.task == 'binary':
        preds = preds.squeeze(1)

    metrics = list(TMF.__dict__[m](
        preds, labels, args.task,
        num_classes=args.num_classes, num_labels=args.num_labels
    ) for m in args.metric_names)

    return metrics


def all_reduce_mean(val, world_size):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_reduce_sum(val):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    return val


def reduce_mean(val, world_size):
    """Collect value to local zero gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.reduce(val, 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_gather(x):
    """Collect value to local rank zero gpu
    :arg
        x(tensor): target
    """
    if dist.is_initialized():
        dest = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(dest, x)
        return torch.cat(dest, dim=0)
    else:
        return x


def all_gather_with_different_size(x):
    """all gather operation with different sized tensor
    :arg
        x(tensor): target
    (reference) https://stackoverflow.com/a/71433508/17670380
    """
    print("===================== {} ==================="%{dist.is_initialized})
    if dist.is_initialized():
        local_size = torch.tensor([x.size(0)], device=x.device)
        all_sizes = all_gather(local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=x.device, dtype=x.dtype)
            x = torch.cat((x, padding))

        all_gathered_with_pad = all_gather(x)
        all_gathered = []
        ws = dist.get_world_size()
        for vector, size in zip(all_gathered_with_pad.chunk(ws), all_sizes.chunk(ws)):
            all_gathered.append(vector[:size])

        return torch.cat(all_gathered, dim=0)
    else:
        return x
