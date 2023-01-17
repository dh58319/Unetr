## Single GPU
# python3 train.py -s setting1 setting2 -m model1 model2 -c 0


# # 01
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m densenet121\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 1 --use-wandb

# # 02
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m resnet50\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 1 --use-wandb

# # 03
# python3 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_tiny_patch4_window7_224\
#  --metric-names accuracy auroc f1_score precision recall specificity\
#  --cuda 3 --use-wandb



# 04
python3 train.py --max-iter 25000 -s btcv_v1 -m UNETR\
 --valid-freq 5000 --model-type monai --cuda 0



# 04 TransUnet
# python train.py --epoch 150 -s btcv_v2 -m transunet\
#  --valid-freq -1 --print-freq 50 --cuda 1 --use-wandb



# 05 Inception TransUnet
## {model_name}_d{hidden_size}_p{num_path}_f{pixshuf_factor}_{concat}
# python train.py --epoch 150 -s btcv_v2 -m transunet_inception_d192_p3_f2\
#  --valid-freq -1 --print-freq 50 --cuda 2 --use-wandb

# python train.py --epoch 1 -s btcv_v2 -m transunet_inception_d192_p3_f2\
#  --valid-freq -1 --print-freq 1 --cuda 0



## Multi GPU
# torchrun --nproc_per_node=2 --master_port=12345 train.py --epoch 100 -s messidor2_v1 eyepacs_v1 -m swin_small_patch4_window7_224\
#  --metric-names accuracy auroc f1_score precision recall specificity -c 1,2\
 
# torchrun --nproc_per_node=2 --master_port=12345 train.py \
# -s setting1 setting2 -m model1 model2 -c 0,1

### Test multi GPU