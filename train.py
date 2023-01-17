import os
import argparse
import torch

from meseg.dataset import get_dataset, get_dataloader
from meseg.model import get_model, get_ddp_model
from meseg.engine import train_one_epoch_with_valid, validate
from meseg.utils.setup import print_batch_run_settings
from meseg.utils import setup, clear, get_optimizer_and_scheduler, get_criterion_scaler, \
    print_meta_data, load_model_list_from_config, get_args_with_setting

import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='pytorch jaejung\'s framework',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1.setup
    setup = parser.add_argument_group('setup')
    setup.add_argument(
        '--config', type=str, default=os.path.join('config', 'train.json'),
        help='paths for each dataset and pretrained-weight. (json)'
    )
    setup.add_argument(
        '-s', '--settings', type=str, default=['ddsm_v1'], nargs='+',
        help='settings used for default value'
    )
    setup.add_argument(
        '--entity', type=str, default='jaejungscene',
        help='project space used for wandb logger'
    )
    setup.add_argument(
        '-proj', '--project-name', type=str, default='MESEG',
        help='project name used for wandb logger'
    )
    setup.add_argument(
        '--who', type=str, default='jaejung',
        help='enter your name'
    )
    setup.add_argument(
        '--use-wandb', action='store_true', default=False,
        help='track std out and log metric in wandb'
    )
    setup.add_argument(
        '-exp', '--exp-name', type=str, default=None,
        help='experiment name for each run'
    )
    setup.add_argument(
        '--exp-target', type=str, default=['setting', 'model_name'], nargs='+',
        help='experiment name based on arguments'
    )
    setup.add_argument(
        '--resume', action='store_true',
        help='if true, resume train from checkpoint_path'
    )
    setup.add_argument(
        '-out', '--output-dir', type=str, default='log',
        help='where log output is saved'
    )
    setup.add_argument(
        '--save-weight', action='store_true',
        help='if true, save best weight during train'
    )
    setup.add_argument(
        '-p', '--print-freq', type=int, default=0,
        help='how often print metric in iter'
    )
    setup.add_argument(
        '--valid-freq', type=int, default=500,
        help='if not None, validate model every certain iter'
    )
    setup.add_argument(
        '--seed', type=int, default=42,
        help='fix seed'
    )
    setup.add_argument(
        '--use-deterministic', action='store_true',
        help='use deterministic algorithm'
    )
    setup.add_argument(
        '--amp', action='store_true', default=False,
        help='enable native amp(fp16) training'
    )
    setup.add_argument(
        '--channels-last', action='store_true',
        help='change memory format to channels last'
    )
    setup.add_argument(
        '-c', '--cuda', type=str, default='0,1,2,3,4,5,6,7,8',
        help='CUDA_VISIBLE_DEVICES options'
    )
    setup.add_argument(
        '--parallel', action="store_true", default=False,
        help='nn.Dataparallel activate'
    )
    setup.set_defaults(amp=False, channel_last=True, pin_memory=True, mode='train')

    # 2. augmentation & dataset & dataloader
    data = parser.add_argument_group('data')
    data.add_argument(
        '--dataset-type', type=str, default='chexpert',
        choices=[
            'chexpert', 'nihchest', # chest
            'ddsm', 'vindr', # breast
            'isic2018', 'isic2019', # skin
            'eyepacs', 'messidor2', # eye
            'pcam', # lymph
            'btcv_v1', 'btcv_v2'
        ],
        help='dataset type'
    )
    data.add_argument(
        '--train-size', type=int, default=(224, 224), nargs='+',
        help='train image size'
    )
    data.add_argument(
        '--train-resize-mode', type=str, default='RandomResizedCrop',
        help='train image resize mode'
    )
    data.add_argument(
        '--random-crop-pad', type=int, default=0,
        help='pad size for ResizeRandomCrop'
    )
    data.add_argument(
        '--random-crop-scale', type=float, default=(0.08, 1.0), nargs='+',
        help='train image resized scale for RandomResizedCrop'
    )
    data.add_argument(
        '--random-crop-ratio', type=float, default=(3/4, 4/3), nargs='+',
        help='train image resized ratio for RandomResizedCrop'
    )
    data.add_argument(
        '-hf', '--hflip', type=float, default=0.5,
        help='random horizontal flip'
    )
    data.add_argument(
        '-vf', '--vflip', type=float, default=None,
        help='random vertical flip'
    )
    data.add_argument(
        '--random-affine', action='store_true', default=False,
        help='enable random affine with pre-settings'
    )
    data.add_argument(
        '-aa', '--auto-aug', action='store_true', default=False,
        help='enable timm rand augmentation'
    )
    data.add_argument(
        '--cutmix', type=float, default=None,
        help='cutmix probability'
    )
    data.add_argument(
        '--mixup', type=float, default=None,
        help='mix probability'
    )
    data.add_argument(
        '-re', '--remode', type=float, default=None,
        help='random erasing probability'
    )
    data.add_argument(
        '--test-size', type=int, default=(224, 224), nargs='+',
        help='test image size'
    )
    data.add_argument(
        '--test-resize-mode', type=str, default='resize_shorter', choices=['resize_shorter', 'resize'],
        help='test resize mode'
    )
    data.add_argument(
        '--center-crop-ptr', type=float, default=0.875,
        help='test image crop percent'
    )
    data.add_argument(
        '--interpolation', type=str, default='bicubic',
        help='image interpolation mode'
    )
    data.add_argument(
        '--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+',
        help='image mean'
    )
    data.add_argument(
        '--std', type=float, default=(0.229, 0.224, 0.225), nargs='+',
        help='image std'
    )
    data.add_argument(
        '--aug-repeat', type=int, default=None,
        help='repeat augmentation'
    )
    data.add_argument(
        '--drop-last', default=False, action='store_true',
        help='enable drop_last in train dataloader'
    )
    data.add_argument(
        '-b', '--batch-size', type=int, default=256,
        help='batch size'
    )
    data.add_argument(
        '--max-iter', type=int, default=25000,
        help='batch size'
    )
    data.add_argument(
        '-j', '--num-workers', type=int, default=8,
        help='number of workers'
    )
    data.add_argument(
        '--pin-memory', action='store_true', default=False,
        help='pin memory in dataloader'
    )

    # 3.model
    model = parser.add_argument_group('model')
    model.add_argument(
        '-m', '--model-names', type=str, default=[], nargs='+',
        help='model name'
    )
    model.add_argument(
        '--model-type', type=str, default=None,
        help='timm or torchvision'
    )
    model.add_argument(
        '--in-channels', type=int, default=3,
        help='input channel dimension'
    )
    model.add_argument(
        '--drop-path-rate', type=float, default=0.0,
        help='stochastic depth rate'
    )
    model.add_argument(
        '--sync-bn', action='store_true', default=False,
        help='apply sync batchnorm'
    )
    model.add_argument(
        '--pretrained', action='store_true', default=False,
        help='load pretrained weight'
    )

    # 4.optimizer & scheduler & criterion
    optimizer = parser.add_argument_group('optimizer')
    scheduler = parser.add_argument_group('scheduler')
    criterion = parser.add_argument_group('criterion')
    optimizer.add_argument(
        '--lr', type=float, default=1e-3,
        help='learning rate(lr)'
    )
    optimizer.add_argument(
        '-e', '--epoch', type=int, default=100,
        help='epoch'
    )
    optimizer.add_argument(
        '--optimizer', type=str, default='adamw',
        help='optimizer name'
    )
    optimizer.add_argument(
        '--momentum', type=float, default=0.9,
        help='optimizer momentum'
    )
    optimizer.add_argument(
        '--weight-decay', type=float, default=1e-3,
        help='optimizer weight decay'
    )
    optimizer.add_argument(
        '--nesterov', action='store_true', default=False,
        help='use nesterov momentum'
    )
    optimizer.add_argument(
        '--betas', type=float, nargs=2, default=[0.9, 0.999],
        help='adam optimizer beta parameter'
    )
    optimizer.add_argument(
        '--eps', type=float, default=1e-6,
        help='optimizer eps'
    )
    scheduler.add_argument(
        '--scheduler', type=str, default='cosine',
        help='lr scheduler'
    )
    scheduler.add_argument(
        '--step-size', type=int, default=2,
        help='lr decay step size'
    )
    scheduler.add_argument(
        '--decay-rate', type=float, default=0.1,
        help='lr decay rate'
    )
    scheduler.add_argument(
        '--min-lr', type=float, default=1e-6,
        help='lowest lr used for cosine scheduler'
    )
    scheduler.add_argument(
        '--restart-epoch', type=int, default=20,
        help='warmup restart epoch period'
    )
    scheduler.add_argument(
        '--milestones', type=int, nargs='+', default=[150, 225],
        help='multistep lr decay step'
    )
    scheduler.add_argument(
        '--warmup-scheduler', type=str, default='linear',
        help='warmup lr scheduler type'
    )
    scheduler.add_argument(
        '--warmup-lr', type=float, default=1e-4,
        help='warmup start lr'
    )
    scheduler.add_argument(
        '--warmup-epoch', type=int, default=5,
        help='warmup epoch'
    )
    criterion.add_argument(
        '--criterion', type=str, default='ce', choices=['ce', 'bce', 'mse'],
        help='loss function'
    )
    criterion.add_argument(
        '--metric-names', type=str, nargs='+',
        default=[
            'avg_dice_score'
        ],
        help='metric name'
    )
    criterion.add_argument(
        '--save-metric', type=str, default='auroc',
        choices=['accuracy', 'auroc', 'f1_score', 'specificity', 'recall', 'precision'],
        help='save model weight based on this metric'
    )
    criterion.add_argument(
        '--smoothing', type=float, default=0.0,
        help='label smoothing'
    )
    criterion.add_argument(
        '--grad-norm', type=float, default=None,
        help='gradient clipping threshold'
    )
    criterion.add_argument(
        '--grad-accum', type=int, default=1,
        help='gradient accumulation'
    )
    criterion.add_argument(
        '--dist', action="store_true", default=False,
        help='gradient accumulation'
    )

    return parser


def run(args):
    # 0. setup distributed
    setup(args)

    # 1. define transform & load dataset
    train_dataset, valid_dataset = get_dataset(args, args.mode)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid_dataset, args)

    # 2. load model
    model = get_model(args)
    model, ddp_model = get_ddp_model(model, args)

    # 3. load optimizer, scheduler, criterion
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    criterion, scaler = get_criterion_scaler(args)

    # 4. train model
    print_meta_data(model, train_dataset, valid_dataset, args)

    args.best = 0
    global_step = 0
    while global_step < args.max_iter:
        if args.distributed:
            train_dataloader.sampler.set_epoch()

        global_step = train_one_epoch_with_valid(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=ddp_model if args.distributed else model,
            optimizer=optimizer,
            criterion=criterion,
            args=args,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            max_iter=args.max_iter
        )
        torch.save(model.state_dict(), os.path.join(args.exp_name, "best_metric_model.pth"))


if __name__ == '__main__':
    # 1. parse command
    parser = get_args_parser()
    args = parser.parse_args()

    if len(args.model_names) == 0:
        args.model_names = load_model_list_from_config(args)

    # 2. print batch run info if batch run is enabled
    is_single = os.environ.get('LOCAL_RANK', None) is None
    is_master = is_single or int(os.environ['LOCAL_RANK']) == 0
    is_batch_run = len(args.model_names) > 1 or len(args.settings) > 1
    if is_master and is_batch_run:
        print_batch_run_settings(args)

    # 3. run N(setting) x N(model_names) experiment
    prev_args = None
    for setting in args.settings:
        for model_name in args.model_names:
            new_args = get_args_with_setting(parser, args.config, setting, model_name, prev_args, args.mode)
            run(new_args)
            clear(new_args)
            prev_args = new_args