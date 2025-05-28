import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import functional as F
import torchvision
import torch.utils.tensorboard as tb
from torchvision import transforms, datasets, models
from skimage.segmentation import find_boundaries as fb
from .import special_transforms as SegT
import numpy as np
import csv
import torch

import os, subprocess

# Mapping of classes to ignore (marked "0") and to keep (given nonzero number)
mapping_20 = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 
    7: 1,    # road
    8: 2,    # sidewalk
    9: 0, 10: 0, 
    11: 3,   # building
    12: 3,   # wall → building
    13: 3,   # fence → building
    14: 0, 15: 0, 16: 0, 
    17: 0,   # pole → building
    18: 0, 19: 0,   # traffic light
    20: 0,   # traffic sign
    21: 4,   # vegetation
    22: 0,   # terrain
    23: 5,   # sky
    24: 6,   # person
    25: 6,   # rider → person
    26: 7,  # car
    27: 7,  # truck → car
    28: 7,  # bus → car
    29: 0, 30: 0, 
    31: 7,  # train → car
    32: 7,  # motorcycle → car
    33: 7,  # bicycle → car
    -1: 0
}

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask

ALL_LABEL_NAMES = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 
                     'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 
                     'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 
                     'traffic sign', 'vegetation', 'terrain','sky','person','rider','car','truck','bus',
                     'caravan', 'trailer', 'train','motorcycle','bicycle']

#### change to different name
SELECT_LABEL_NAMES = [
    'unlabelled', 
    'road', 
    'sidewalk', 
    'building', 
    'vegetation', 
    'sky', 
    'person', 
    'car'  # Merged: car, truck, bus, train, motorcycle, bicycle
]
N_CLASSES = len(SELECT_LABEL_NAMES)  # Now 11

#print("Number of classes", N_CLASSES)

### Weights for Focal loss
FOCAL_LOSS_WEIGHTS = [
    0.0,      # unlabeled
    0.1825,   # road
    0.0525,   # sidewalk
    0.0525,   # building
    0.025,    # vegetation
    0.01,     # sky
    0.0525,   # person
    0.0525    # car
]
class CityDataset(Dataset):
    def __init__(self, dataset_path, transform = SegT.Compose([ SegT.Resize((128, 128)), SegT.ToTensor() ]) ): 
        ## The original images are way too big for my GPU to handle...so I transform both the train and valid data
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_leftImg8bit.png')): 
            self.files.append(im_f.replace('_leftImg8bit.png', '')) 
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_leftImg8bit.png') 
        lbl_orig = Image.open(b + '_gtFine_labelIds.png') 
        #print(lbl.size)
        np_lbl = encode_labels(np.array(lbl_orig, dtype=np.uint8))
        #print(np_lbl.shape)
        convertPIL = T.ToPILImage()
        lbl = convertPIL(np_lbl)
        #print(lbl.size)
        if self.transform is not None:
            im, lbl = self.transform(im, lbl) 
            
        mask=fb(np_lbl,mode='outer').astype(np.uint8)
        mask+=1
        mask=torch.from_numpy(mask).type(torch.FloatTensor)        
        
        return im, lbl, mask

def load_data(dataset_path, num_workers=0, batch_size=8, **kwargs):
    dataset = CityDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

#def accuracy(outputs, labels):
#    outputs_idx = outputs.max(1)[1].type_as(labels)
#    return outputs_idx.eq(labels).float().mean()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Source: https://github.com/pytorch/vision/blob/main/references/video_classification/utils.py
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=N_CLASSES):
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        self.matrix = self.matrix.to(preds.device) ###### issue here...
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)
    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)
    
    @property
    def iou(self):
        return self.class_iou.mean()
    
    @property
    def frequency_weighted_iou(self):
        freq = self.matrix.sum(1) / (self.matrix.sum() + 1e-5)
        return (freq * self.class_iou).sum()

    
###################################################################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.RandomHorizontalFlip(), SegT.Resize((128, 128)), SegT.ToTensor()]))
    dataset0 = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.ToTensor()]))
    dataset1 = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.Resize((128, 128)), SegT.ToTensor()]))
    dataset2 = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.Resize((128, 128)), SegT.RandomHorizontalFlip(), SegT.ToTensor()]))
    dataset3 = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.Resize((128, 128)), SegT.RandomResizedCrop(32), SegT.ToTensor()]))
    dataset4 = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.Resize((128, 128)), SegT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), SegT.ToTensor()]))
    dataset_all = CityDataset('small_dataset/train', transform=SegT.Compose([SegT.Resize((128, 128)), SegT.RandomHorizontalFlip(), SegT.RandomResizedCrop(32), SegT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), SegT.ToTensor()]))
        
    from pylab import show, imshow, subplot, axis

############## Show the original, label, mask ######################################################################
    for i in range(3):
        im_orig, lbl_orig, mask = dataset0[i]
        sub1 = subplot(3, 3, 3*i + 1)
        imshow(F.to_pil_image(im_orig))
        if i ==0:
            sub1.set_title("Original")
        axis('off')
        
        sub2 = subplot(3, 3, 3*i + 2)
        imshow(SegT.label_to_pil_image(lbl_orig))
        if i==0:
            sub2.set_title("Label")
        axis('off')
        
        sub3 = subplot(3, 3, 3*i + 3) 
        imshow(mask.numpy())
        if i==0:
            sub3.set_title("Mask")
        axis('off')
    show()
    
############### Show the data augmentation methods ###################################################################
    for i in range(3):
        #print(i)
        im_orig, lbl, mask = dataset0[i]
        im_resize, lbl, mask = dataset1[i]
        im_flip, lbl, mask = dataset2[i]
        im_crop, lbl, mask = dataset3[i]
        im_color, lbl, mask = dataset4[i]
        im_final, lbl, mask = dataset_all[i]
        #print(im.shape)
        #print(lbl.shape)
        sub1 = subplot(3, 5, 5*i + 1)
        imshow(F.to_pil_image(im_orig))
        if i ==0:
            sub1.set_title("Original")
        axis('off')
        
        sub2 = subplot(3, 5, 5*i + 2)
        imshow(F.to_pil_image(im_resize))
        if i==0:
            sub2.set_title("Resized 128x128")
        axis('off')
        
        sub3 = subplot(3, 5, 5*i + 3) 
        imshow(F.to_pil_image(im_crop))
        if i==0:
            sub3.set_title("Random Crop 32x32")
        axis('off')       
        
        sub4 = subplot(3, 5, 5*i + 4) 
        imshow(F.to_pil_image(im_color))
        if i==0:
            sub4.set_title("Color Jitter")
        axis('off')

        sub5 = subplot(3, 5, 5*i + 5) 
        imshow(F.to_pil_image(im_final))
        if i==0:
            sub5.set_title("All")
        axis('off')
        
    show()    
    
    
#######################################################################################################################
    import numpy as np
    
    #### FULL DATASET
    
    c = np.zeros(len(ALL_LABEL_NAMES)) 
    for im, lbl, mask in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(ALL_LABEL_NAMES)) 
    distribution_segs = 100 * c / np.sum(c)
    print(distribution_segs.shape)
     
    ############### SELECTED DATASET
    
    truncated_distribution = distribution_segs[:20]
    fig = plt.figure(figsize = (15, 5))
    plt.xticks(rotation=30)
    # creating the bar plot
    plt.bar(SELECT_LABEL_NAMES, truncated_distribution, color ='maroon',
            width = 0.4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.xlabel("Object Distribution")
    plt.ylabel("Percentage")
    plt.title("Selected Categories")
    plt.show()
