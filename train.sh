# UNETR
python train.py -s btcv -m dh_unet \
--print-freq 100 --valid-freq 500 --max-iter 25000 \
-proj donghyunkim --use-wandb --cuda 3 mul_head 1   # -proj의 이름과 cuda만 바꾸싶쇼



## test test