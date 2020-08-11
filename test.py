import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.training import *


def test(model, val_loader, ce_loss, coarse):
    model.eval()

    records = [0]*6
    batch_num = len(val_loader) 
    
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            inputs = data[0].cuda()
            targets = data[1].cuda()
            
            output = model(inputs) # size: (bsz, n_classes, h, w)
            
            output = output.squeeze(dim=1)
            loss = ce_loss(output, targets)
            records[0] += loss.item()
            output[output < 0.5] = 0.0
            output[output >= 0.5] = 1.0
            records[1] += error(output, targets) 
            records[2] += iou(output, targets, coarse)
            percision, recall, f1 = pr(output, targets, coarse)
            records[3] += percision
            records[4] += recall
            records[5] += f1
            # raise RuntimeError("jjj")
        
        records = [record/batch_num for record in records]
        print('Eval - Loss: {:.4f}, Acc: {:.4f}, IoU: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
            records[0], 1 - records[1], records[2], records[3], records[4], records[5]))
