import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.visualize import *

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# mean_std = ([0.41189489566336, 0.4251328133025, 0.4326707089857], \
#                 [0.27413549931506, 0.28506257482912, 0.28284674400252])

# 1 for bg, 0 for crack
class_color = np.array([[255, 255, 255], [0, 0, 0]])


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).float()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().float()
        return label


class crackDataset(Dataset):
    def __init__(self, root, split, joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor()):
        
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.mean_std = mean_std
        self.imgs = []

        self._base_dir = root
        self._image_dir = os.path.join(self._base_dir, split, "image")
        
        label_path = os.path.join(self._base_dir, 'labels', split + '.txt')
        lines = open(label_path, "r").read().splitlines()
        for line in lines:
            line = line.strip()
            _image = os.path.join(self._image_dir, line)
            assert os.path.isfile(_image)
            self.imgs.append(_image)
        print('Number of images in {}: {:d}'.format(split, len(self.imgs)))

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        
        gt_path = img_path.replace("image", "mask").replace("jpg", "png")
        target = Image.open(gt_path)

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target]) 

        # # # visualization
        # print("idx:%d, trans1 img path:%s"%(index, img_path))
        # view_image(img, idx=index, tensor=False, save=True)
        
        # print("idx:%d, trans1 target path:%s"%(index, gt_path))
        # view_gt(target, idx=index, tensor=False, save=True)

        if self.transform is not None:
            img = self.transform(img)

        target = self.target_transform(target)

        # # # visualization
        # print("idx:%d, trans2 img path:%s"%(index, img_path))
        # view_image(img,  idx=index, tensor=True, save=True)
        
        # print("idx:%d, trans2 target path:%s"%(index, gt_path))
        # view_gt(target,  idx=index, tensor=True, save=True)
        
        # if index > 10:
        #     raise

        return img, target

    def __len__(self):
        return len(self.imgs)

