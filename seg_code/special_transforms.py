# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as F
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(T.Resize):
    def __call__(self, image, target):
        #print(Image.BICUBIC, Image.NEAREST)
        return F.resize(image, self.size, self.interpolation), F.resize(target, self.size, interpolation=Image.BICUBIC) #IMAGE.NEAREST)
        # UserWarning: Argument interpolation should be of type
        ## T.Resize((224, 224), T.InterpolationMode.BICUBIC)

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target



import torchvision.transforms.functional as F

class ToGrayscale(object):
    def __init__(self, num_output_channels=1):
        """
        Args:
            num_output_channels (int): 
                1 => single-channel grayscale
                3 => repeated in all 3 channels (for models expecting RGB input)
        """
        self.num_output_channels = num_output_channels

    def __call__(self, image, target):
        # Convert RGB (or RGBA) PIL image to grayscale
        image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
        # Leave the label/mask as-is
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(T.Normalize):
    def __call__(self, image, target):
        return super().__call__(image), super().__call__(target)
    
class RandomRotation(object):
    def __init__(self, angles=[90, 180, 270], prob=0.5):
        self.angles = angles
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.choice(self.angles)
            # Rotate image with bilinear interpolation
            image = F.rotate(
                image, 
                angle, 
                interpolation=F.InterpolationMode.BILINEAR,
                expand=False  # Maintain original image size
            )
            # Rotate label/mask with nearest neighbor (preserve class indices)
            target = F.rotate(
                target, 
                angle, 
                interpolation=F.InterpolationMode.NEAREST,
                expand=False
            )
        return image, target

class Normalize(T.Normalize):
    def __call__(self, image, target):
        return super().__call__(image), target


class ColorJitter(T.ColorJitter):
    def __call__(self, image, target):
        return super().__call__(image), target


class RandomResizedCrop(T.RandomResizedCrop):
    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        return F.resized_crop(image, i, j, h, w, self.size, self.interpolation),\
               F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)
### Image.NEAREST creates warning
### UserWarning: Argument interpolation should be of type InterpolationMode instead of int.
### Image.NEAREST = 0


def label_to_tensor(lbl):
    """
    Reads a PIL pallet Image img and convert the indices to a pytorch tensor
    """
    return torch.as_tensor(np.array(lbl, np.uint8, copy=True)) #### creates problems if copy=False, but does this cause memory issues???


def label_to_pil_image(lbl):
    """
    Creates a PIL Image from a label tensor/array with Cityscapes colors
    """
    # Define Cityscapes color palette (20 classes)
    CITYSCAPES_PALETTE = [
        0, 0, 0,          # unlabelled
        128, 64, 128,     # road
        244, 35, 232,     # sidewalk
        70, 70, 70,       # building
        102, 102, 156,    # wall
        190, 153, 153,    # fence
        153, 153, 153,    # pole
        250, 170, 30,     # traffic light
        220, 220, 0,      # traffic sign
        107, 142, 35,     # vegetation
        152, 251, 152,    # terrain
        70, 130, 180,     # sky
        220, 20, 60,      # person (bright red)
        255, 0, 0,        # rider
        0, 0, 142,        # car
        0, 0, 70,         # truck
        0, 60, 100,       # bus
        0, 80, 100,       # train
        0, 0, 230,        # motorcycle
        119, 11, 32       # bicycle
    ]
    
    # Pad palette to 256 colors (768 values)
    CITYSCAPES_PALETTE += [0] * (768 - len(CITYSCAPES_PALETTE))

    # Handle dimensions
    if isinstance(lbl, torch.Tensor):
        if lbl.ndimension() not in [2, 3]:
            raise ValueError(f'lbl should be 2/3 dimensional. Got {lbl.ndimension()} dimensions.')
        lbl = lbl.squeeze().cpu().numpy()
    elif isinstance(lbl, np.ndarray):
        if lbl.ndim not in [2, 3]:
            raise ValueError(f'lbl should be 2/3 dimensional. Got {lbl.ndim} dimensions.')
        lbl = lbl.squeeze()
    else:
        raise TypeError(f'lbl should be Tensor or ndarray. Got {type(lbl)}.')

    # Create palette image
    im = Image.fromarray(lbl.astype(np.uint8), mode='P')
    im.putpalette(CITYSCAPES_PALETTE)
    return im

class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), label_to_tensor(label)
