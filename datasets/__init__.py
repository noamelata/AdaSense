import glob
import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils import data

from datasets.celeba import CelebA
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
import torchvision
from PIL import Image
from functools import partial

circle_mask = torch.from_numpy(np.load("inp_masks/circle_mask_for_ct.npy"))

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class PreporcessedDatasetNoMask(torch.utils.data.Dataset):
    def __init__(self, data_root, data_len=-1):
        imgs = glob.glob(os.path.join(data_root, "image*"))

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

    def __getitem__(self, index):
        path = self.imgs[index]
        img = torch.from_numpy(np.load(path))
        img = img / 7e-5
        from models.mri_utils import ifft2c_new
        img = ifft2c_new(img.permute(1, 2, 0)).permute(2, 0, 1)

        return img, img

    def __len__(self):
        return len(self.imgs)

def tensor2float(x):
    return x.float()

def identity(x):
    return x

class DeepLesion(data.Dataset):
    Mean = -615.9544
    STD = 509.3955

    def __init__(self, root, image_size=(256, 256), rand_flip=False, data_len=-1):
        import cv2
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            tensor2float,
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if rand_flip else identity,

        ])
        self.data_paths = glob.glob(os.path.join(root, "*/*.png"))[:data_len]

    def __getitem__(self, item):
        import cv2
        data = self.tfs(cv2.imread(self.data_paths[item], -1).astype(np.int32) - 32768)
        return (data - DeepLesion.Mean) / DeepLesion.STD, 0

    def __len__(self):
        return len(self.data_paths)

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size[1]), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size[1]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size[1]), transforms.ToTensor()]
        )

    if config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "LSUN":
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', "ood_{}".format(config.data.category)),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            train_folder = "{}_train".format(config.data.category)
            val_folder = "{}_val".format(config.data.category)
            test_dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[val_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                )
            )
            dataset = test_dataset
    
    elif config.data.dataset == "CelebA_HQ" or config.data.dataset == 'FFHQ':
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "ood_celeba"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size[0], config.data.image_size[1]]),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "celeba_hq"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size[0], config.data.image_size[1]]),
                                              transforms.ToTensor()])
            )
            test_dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "celeba_hq_test"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size[0], config.data.image_size[1]]),
                                              transforms.ToTensor()])
            )
    elif config.data.dataset == 'ImageNet':
        # only use validation dataset here
        if config.data.subset_1k:
            from datasets.imagenet_subset import ImageDataset
            dataset = ImageDataset(os.path.join(args.exp, 'datasets', 'imagenet', 'imagenet'),
                     os.path.join(args.exp, 'imagenet_val_1k.txt'),
                     image_size=config.data.image_size,
                     normalize=False)
            test_dataset = dataset
        elif config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', 'ood'),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageNet(
                os.path.join(args.exp, 'datasets', 'imagenet'), split='val',
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
    elif config.data.dataset == "MRI":
        from datasets.singlecoil_knee_data import MICCAI2020Data
        dataset = MICCAI2020Data("datasets/singlecoil_train/", num_cols=368)
        test_dataset = MICCAI2020Data("datasets/singlecoil_val/", custom_split="raw_test", num_cols=368)
        return dataset, test_dataset
    elif config.data.dataset == "CT":
        dataset = DeepLesion("datasets/CT/train/")
        test_dataset = DeepLesion("datasets/CT/test/")
        return dataset, test_dataset
    else:
        dataset, test_dataset = None, None
    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01
    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)
    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]
    return X

def center_crop(x, shape):
    """Center crops a tensor to the desired 2D shape.

    Args:
        x(union(``torch.Tensor``, ``np.ndarray``)): The tensor to crop.
            Shape should be ``(batch_size, height, width)``.
        shape(tuple(int,int)): The desired shape to crop to.

    Returns:
        (union(``torch.Tensor``, ``np.ndarray``)): The cropped tensor.
    """
    assert len(x.shape) == 3
    assert 0 < shape[0] <= x.shape[1]
    assert 0 < shape[1] <= x.shape[2]
    h_from = (x.shape[1] - shape[0]) // 2
    w_from = (x.shape[2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[:, h_from:h_to, w_from:w_to]
    return x

def inverse_data_transform(config, X, _min=None, _max=None):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]
    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0
    elif config.data.complex_abs:
        from models.mri_utils import complex_abs
        if X.dim() == 3:
            X = center_crop(complex_abs(X.permute(1, 2, 0)).unsqueeze(0), (320, 320))
            _min = 0 if _min is None else _min
            _max = X.amax(keepdim=True) if _max is None else _max
            X = (X - _min) / (_max - _min)
            X = X.flip(1)
        elif X.dim() == 4:
            X = center_crop(complex_abs(X.permute(0, 2, 3, 1)), (320, 320)).unsqueeze(1)
            _min = 0 if _min is None else _min
            _max = X.amax((1, 2, 3), keepdim=True) if _max is None else _max
            X = (X - _min) / (_max - _min)
            X = X.flip(2)
    elif config.data.ct_norm:
        X = (X * DeepLesion.STD) + DeepLesion.Mean
        X = torch.clamp(X, min=(-32768), max=32767)
        if X.dim() == 3:
            _min = X.amin((0, 1, 2), keepdim=True) if _min is None else _min
            _max = X.amax((0, 1, 2), keepdim=True) if _max is None else _max
            X = (X - _min) / (_max - _min + 1e-10)
        elif X.dim() == 4:
            _min = X.amin((1, 2, 3), keepdim=True) if _min is None else _min
            _max = X.amax((1, 2, 3), keepdim=True) if _max is None else _max
            X = (X - _min) / (_max - _min + 1e-10)
        X = X * circle_mask.to(X.device)
    return torch.clamp(X, 0.0, 1.0)
