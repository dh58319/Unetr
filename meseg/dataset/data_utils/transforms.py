from math import floor
from torchvision import transforms
from timm.data import rand_augment_transform


class TrainTransform:
    def __init__(self,
                 resize:tuple=(224, 224),
                 resize_mode:str='RandomResizedCrop',
                 gray_image:bool=False,
                 pad:int=0,
                 scale:tuple=(0.08, 1.0),
                 ratio:tuple=(0.75, 1.333333),
                 hflip:float=0.5,
                 vflip:float=None,
                 auto_aug:bool=False,
                 random_affine:bool=True,
                 remode:bool=0.2,
                 interpolation:str='bicubic',
                 mean:tuple=(0.485, 0.456, 0.406),
                 std:tuple=(0.229, 0.224, 0.225),
                 ):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        if gray_image:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        if vflip:
            transform_list.append(transforms.RandomVerticalFlip(vflip))

        if auto_aug:
            transform_list.append(rand_augment_transform('rand-m9-mstd0.5', {}))

        if random_affine:
            transform_list.append(transforms.RandomAffine(
                degrees=(-15, 15), translate=(0.0, 0.2),
                scale=(0.75, 1.0), interpolation=interpolation
            ))

        if resize_mode == 'RandomResizedCrop':
            transform_list.append(
                transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop':
            transform_list.extend([transforms.Resize(resize, interpolation=interpolation),
                                   transforms.RandomCrop(resize, padding=pad)])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ValTransform:
    def __init__(self,
                 size:tuple=(224, 224),
                 resize_mode:str='resize_shorter',
                 gray_image:bool=False,
                 crop_ptr:float=0.875,
                 interpolation:str='bicubic',
                 mean:tuple=(0.485, 0.456, 0.406),
                 std:tuple=(0.229, 0.224, 0.225),
                 ):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = []

        if gray_image:
            transform_list.append(transforms.Grayscale(num_output_channels=3))

        transform_list.extend([
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)