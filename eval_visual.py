import os
import sys
import time
import random

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import crackForest, joint_transforms
from utils.training import *

# setting
N_CLASSES = 2
# # eval over testset
# Data_PATH = "Data/crackForest_randomSized25crop"
# save_root = "test_predict"
# eval over trainset
Data_PATH = "Data/crackForest_randomSized25crop-bak"
save_root = "train_predict"
if not os.path.isdir(save_root):
    os.makedirs(save_root)
model_name = 'checkpoint-63.pth'
Resume_PATH = os.path.join("output-randomSizedCrop", model_name)

test_bsz = 1

GPU_ID = range(1)
SEED = 123
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


# data loader and transform
MEAN_STD = crackForest.mean_std
val_joint_transform = None
val_dset = crackForest.crackDataset(Data_PATH, "test", 
            joint_transform=val_joint_transform,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*MEAN_STD),
        ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=test_bsz, num_workers=16, shuffle=False, pin_memory=True, drop_last=False)


# build model
model = tiramisu.FCDenseNet103(n_classes=N_CLASSES)

train_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
print("Model size: {:.5f}M".format(train_paras))
print('net', model)

model = torch.nn.DataParallel(model, device_ids=GPU_ID).cuda()

# resuming checkpoint
print('load weights from {}'.format(Resume_PATH))
checkpoint = torch.load(Resume_PATH)
pretrained_dict = checkpoint['state_dict']
model_dict = model.state_dict()
# check input mode
print(list(pretrained_dict.values())[0].size())
# avoid mismatched pytorch version
assert list(pretrained_dict.keys()) == list(model_dict.keys())
update_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(update_dict)
model.load_state_dict(model_dict)
print('load weights finished!')


model.eval()
class_color = np.array([[255, 255, 255], [0, 0, 0]])
cnt = 1
with torch.no_grad():
    for idx, data in enumerate(val_loader):
        inputs = data[0].cuda()
        targets = data[1].numpy()

        output = model(inputs) # size: (bsz, n_classes, h, w)
            
        preds = get_predictions(output)

        preds = preds.cpu().numpy()
        for ii in range(test_bsz):
            # pred
            pred = preds[ii].astype(np.uint8)
            # convert binary map with single channel to color map
            r = pred.copy()
            g = pred.copy()
            b = pred.copy()
            for l in range(0,2):
                r[pred==l] = class_color[l,0]
                g[pred==l] = class_color[l,1]
                b[pred==l] = class_color[l,2]

            pred = np.dstack([r, g, b])
            h, w, c = pred.shape

            # gt
            gt = targets[ii].astype(np.uint8)
            # convert binary map with single channel to color map
            r = gt.copy()
            g = gt.copy()
            b = gt.copy()
            for l in range(0,2):
                r[gt==l] = class_color[l,0]
                g[gt==l] = class_color[l,1]
                b[gt==l] = class_color[l,2]

            gt = np.dstack([r, g, b])
            h, w, c = gt.shape
            
            # # visualize
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(pred)
            # ax[1].imshow(gt)
            # plt.savefig(os.path.join(save_root, "%04d.jpg"%(cnt)))
            # # plt.show() # comment for show
            # plt.close()

            # save for paper
            # pred result
            f1 = plt.figure(1, figsize=(w, h), dpi=1)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.imshow(pred)
            plt.savefig(os.path.join(save_root, "%04d_pred.jpg"%(cnt)))
            # plt.show() # comment for show
            plt.close()

            # gt result
            f1 = plt.figure(1, figsize=(w, h), dpi=1)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.imshow(gt)
            plt.savefig(os.path.join(save_root, "%04d_gt.jpg"%(cnt)))
            # plt.show() # comment for show
            plt.close()

            cnt += 1
            # if cnt > 30:
            #     raise
        # raise
