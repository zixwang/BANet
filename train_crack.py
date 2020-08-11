import os
import sys
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import crackForest, joint_transforms
from utils import training
from test import test

# setting
N_CLASSES = 2
Data_PATH = "Data/crackForest_randomSized25crop"
save_root = "output"
if not os.path.isdir(save_root):
    os.makedirs(save_root)
# # jlbai trained model
# Resume_PATH = 'checkpoint-63-0.017-0.931.pth'

model_name = 'checkpoint-2.pth'
Resume_PATH = os.path.join(save_root, model_name)
Resume_PATH = ''
# True, False
EVAL = False

train_bsz = 8
test_bsz = 1 # must be 1 because of the IOU computation
LR = 1e-4
LR_DECAY = 0.5
LR_DECAY_EVERY_N_EPOCHS = 50
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
PRINT_FREQ = 10000
eval_freq = 1
start_epoch = 0

GPU_ID = range(1)
SEED = 123
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

print('train batch: {}\ntest batch: {}\nlearning rate: {}\nepochs: {}\nsnapshot: {}'.format(
    train_bsz, test_bsz, LR, N_EPOCHS, eval_freq))

# build model
model = tiramisu.FCDenseNet103(n_classes=N_CLASSES)

train_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
print("Model size: {:.5f}M".format(train_paras))
print('net', model)


# optimizer && lr scheduler && criterion
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=LR_DECAY_EVERY_N_EPOCHS, gamma=LR_DECAY)
ce_loss = nn.BCELoss().cuda()
dice_loss = training.DiceLoss().cuda()


# data loader and transform
MEAN_STD = crackForest.mean_std
# train_joint_transform = transforms.Compose([
#             # joint_transforms.JointResize((224,224)), #(w,h)
#             # joint_transforms.JointRandomHorizontalFlip(),
#             # joint_transforms.JointRandomGaussianBlur(),
#             # joint_transforms.JointRandomRotate(15),
#         ])
train_joint_transform = None
train_dset = crackForest.crackDataset(Data_PATH, "train",
            joint_transform=train_joint_transform,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*MEAN_STD),
        ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=train_bsz, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)

# val_joint_transform = transforms.Compose([
#             joint_transforms.JointResize((224,224)),
#         ])
val_joint_transform = None
val_dset = crackForest.crackDataset(Data_PATH, "test", 
            joint_transform=val_joint_transform,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*MEAN_STD),
        ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=test_bsz, num_workers=16, shuffle=False, pin_memory=True, drop_last=False)

model = torch.nn.DataParallel(model, device_ids=GPU_ID).cuda()

# '''
# resuming checkpoint
if Resume_PATH:
    print('load weights from {}'.format(Resume_PATH))
    checkpoint = torch.load(Resume_PATH)
    start_epoch = checkpoint['start_epoch']
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

# model = torch.nn.DataParallel(model, device_ids=GPU_ID).cuda()

if EVAL:
    since = time.time()
    test(model, val_loader, ce_loss, coarse=True) 
    time_elapsed = time.time() - since
    print('Eval Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    sys.exit(0)



# train && test
batch_num = len(train_loader)
for epoch in range(start_epoch+1, N_EPOCHS+1):
    since = time.time()

    ### Train ###
    model.train()

    for idx, data in enumerate(train_loader):
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        optimizer.zero_grad()
        outputs = model(inputs)
        
        total_loss = 0.0
        losses = ""

        for output in outputs:
            output = output.squeeze(dim=1)
            loss = ce_loss(output, targets)
            total_loss += loss
            losses += "[loss %.4f]" % (loss.item())

        loss = dice_loss(outputs[0], targets)
        total_loss += loss
        losses += "[dice loss %.4f]" % (loss.item())
        
        total_loss.backward()
        optimizer.step()
        
        if idx % PRINT_FREQ == 0:
            print('[ep {:4d}][lr {:4f}][i {:4d}]{:s}'.format(epoch, \
                optimizer.param_groups[0]['lr']*1e4, idx, losses))

    scheduler.step()

    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    ### Test ###
    if epoch % eval_freq == 0:
        ### Checkpoint ###
        fname = "checkpoint-%d.pth" % (epoch)
        torch.save({
                'start_epoch': epoch,
                'state_dict': model.state_dict()
            }, os.path.join(save_root, fname))

        since = time.time()
        test(model, val_loader, ce_loss, coarse=True)
        time_elapsed = time.time() - since
        print('Eval Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

