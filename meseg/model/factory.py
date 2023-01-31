import timm
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel
import monai.networks.nets as monai

from .unetr import UNETR
from .my_unetr import UNETR2
from . import model_config

def get_model(args):
    if args.model_type == 'torchvision':
        model = torchvision.models.__dict__[args.model_name](
            num_classes=args.num_classes,
            pretrained=args.pretrained
        ).cuda(args.device)

    elif args.model_type == 'timm':
        model = timm.create_model(
            args.model_name,
            in_chans=args.in_channels,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
            pretrained=args.pretrained
        ).cuda(args.device)

    elif args.model_name == "unetr":
        model = UNETR(**model_config.UNETR()).cuda(args.device)
        
    elif args.model_name == "dh_unet":
        model = UNETR2((96,96,96),1,14,mul_head=args.mul_head).cuda(args.device)

    else:
        raise Exception(f"{args.model_type} is not supported yet")

    return model



def get_ddp_model(model, args):
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])

    else:
        ddp_model = None

    return model, ddp_model