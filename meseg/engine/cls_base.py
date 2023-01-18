import os

from tqdm import tqdm
import pandas as pd
import torch
import time, datetime
import wandb

from meseg.utils import compute_metrics, Metric, reduce_mean

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import decollate_batch



@torch.inference_mode()
def validate(args, model, val_loader, global_step, post_label, post_pred, dice_metric):
    # epoch_iterator_val = tqdm(
    #     val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    # )
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
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
        x, y = batch["image"].to(args.device), batch["label"].to(args.device)
        # print("x.shape:", x.shape)
        # print("y.shape:", y.shape)
        batch_size = x.size(0)

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
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        loss_m.update(loss, batch_size)
        if args.print_freq and global_step % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN(iter {global_step:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)


        if (global_step%args.valid_freq==0 and global_step!=0) or global_step+1 == max_iter:
            post_label = AsDiscrete(to_onehot=args.num_classes)
            post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
            dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            mean_dice_val = validate(args, model, valid_dataloader, global_step, 
                                     post_label, post_pred, dice_metric)
            # metric_values.append(mean_dice_val)
            if mean_dice_val > args.best:
                args.best = mean_dice_val
                torch.save(model.state_dict(), args.best_weight_path)
                args.log("Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}"
                        .format(args.best, mean_dice_val))
            else:
                args.log("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}"
                        .format(args.best, mean_dice_val))
                    
            if args.use_wandb:
                args.log({
                    "train loss": loss_m.val.item(),
                    "valid mean Dice score": mean_dice_val
                }, True)

        global_step += 1
    return global_step


