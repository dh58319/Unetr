import os

import tqdm
import pandas as pd
import torch
import time, datetime
import wandb
from meseg.utils import compute_metrics, Metric, reduce_mean

from meseg.dataset import btcv_v2_inference
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch


@torch.inference_mode()
def test(valid_dataloader, valid_dataset, model, args):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    predictions = list()
    labels = list()
    start_time = time.time()
    args.log(f'start validate of {args.model_name}...')

    for batch_idx, batch in enumerate(valid_dataloader):
        x, y = (batch["image"].to(args.device), batch["label"].to(args.device))

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)

        predictions.append(y_hat)
        labels.append(y)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"{args.mode.upper()}: [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. move prediction & label to cpu and normalize prediction to probability.
    predictions = torch.concat(predictions, dim=0).detach().float().cpu()
    labels = torch.concat(labels, dim=0).detach().cpu()
    if args.task == 'binary':
        predictions = torch.sigmoid(predictions)
    else:
        predictions = torch.softmax(predictions, dim=-1)

    # 4. save inference result or compute metrics
    if args.mode == 'test' and args.dataset_type in ['isic2018', 'isic2019']:
        metrics = []
        save_path = os.path.join(args.log_dir, f"{args.model_name}.csv")
        df = {"image":valid_dataset.id_list}
        df.update({c: predictions[:, i].tolist() for i, c in enumerate(valid_dataset.classes)})
        pd.DataFrame(df).to_csv(save_path, index=False)
        args.log(f'saved prediction to {save_path}')

    else:
        metrics = [x.detach().float().cpu().item() for x in compute_metrics(predictions, labels, args)]
        space = 12
        num_metric = 1 + len(metrics)
        args.log('-'*space*num_metric)
        args.log(("{:>12}"*num_metric).format('Stage', *args.metric_names))
        args.log('-'*space*num_metric)
        args.log(f"{f'{args.mode.upper()}':>{space}}" + "".join([f"{m:{space}.4f}" for m in metrics]))
        args.log('-'*space*num_metric)

    return predictions, labels, metrics




@torch.inference_mode()
def validate(model, epoch_iterator_val, global_step, post_label, post_pred, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val




def train_one_epoch_with_valid(
        train_dataloader, valid_dataloader, model, optimizer, criterion, args,
        scheduler=None, scaler=None, global_step=None, max_iter=None
    ):

    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        batch_size = x.size(0)
        x, y = batch["image"].to(args.device), batch["label"].to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)
            
        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.batch_idx()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.batch_idx()

        loss_m.update(loss, batch_size)
        if args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN(iter {global_step:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)


        if (global_step%args.valid_freq==0 and global_step!=0) or global_step == max_iter:
            post_label = AsDiscrete(to_onehot=args.num_classes)
            post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
            dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            mean_dice_val = validate(model, valid_dataloader, global_step, post_label, post_pred, dice_metric)

            # metric_values.append(mean_dice_val)
            if mean_dice_val > args.best:
                args.best = mean_dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(args.exp_name, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        args.best, mean_dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        args.best, mean_dice_val
                    )
                )
            wandb.log({"train loss": loss_m, "mean dice score on valid data": mean_dice_val})

        global_step += 1
    return global_step

        # start_time = time.time()

    # 3. calculate metric
    # duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    # data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    # f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    # loss = loss_m.compute()

    # # 4. print metric
    # space = 12
    # num_metric = 4 + 1
    # args.log('-'*space*num_metric)
    # args.log(("{:>12}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    # args.log('-'*space*num_metric)
    # args.log(f"{'TRAIN('+str(global)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    # args.log('-'*space*num_metric)

    # if args.valid_freq==None  or  args.valid_freq==0:
    #     validate(valid_dataloader, model, criterion, args, epoch=epoch, train_loss=loss)
    
    args.log('*'*100)


