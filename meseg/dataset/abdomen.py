import os
import json
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch
from .dataset import register_dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)



@register_dataset
class btcv(CacheDataset):
    classes = 14
    
    def __init__(self, root, mode) -> None:
        mode = "training" if mode=="train" else "validation"
        data_split = os.path.join(root, "dataset_0.json")
        data_split = load_decathlon_datalist(data_split, True, mode)
        super().__init__(
            data=data_split,
            transform=self.train_transforms if mode=="training" else self.val_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=4
        )

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), # load ct image
            EnsureChannelFirstd(keys=["image", "label"]), # 1 channel is added
            Orientationd(keys=["image", "label"], axcodes="RAS"), # up and down or righ and left inversion
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged( # normalize image 0~1
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"), # remove background 
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(  
                keys=["image", "label"],
                spatial_axis=[0], # up and down filp
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1], # left and right filp
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2], # 
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)