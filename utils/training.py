import os
import sys
import math
import string
import random
import shutil

import numpy as np
import skimage.morphology as sm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from utils.visualize import *


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1.0 - torch.clamp(loss.sum() / N, 0., 1.)
 
		return loss


def get_predictions(outputs):
	bs, c, h, w = outputs.size()
	# values shape: (bsz, h, w), indices shape: (bsz, h, w)
	values, indices = outputs.max(1)
	return indices


def error(preds, targets):
	assert preds.size() == targets.size()
	n_pixels = np.prod(preds.size()) * 1.0
	incorrect = preds.ne(targets).sum().item()
	err = incorrect / n_pixels
	return round(err, 5)


def dilate_target(target, kernel_size):
	target_cp = target.numpy().copy()
	for i in range(target_cp.shape[0]):
		target_cp[i, :, :] = sm.dilation(np.squeeze(target_cp[i,:,:]), sm.square(kernel_size))
	return torch.from_numpy(target_cp)


def iou(preds, targets, coarse=False):
	assert preds.size() == targets.size() 
	preds = preds.cpu()
	targets = targets.cpu()
	if coarse:
		targets_process = dilate_target(targets, kernel_size=2)
	else:
		targets_process = targets
	intersection = (preds * targets_process).sum().item() * 1.0
	union = (preds.sum() + targets.sum() - intersection).item()
	if union:
		return round(min(intersection / union, 1), 5)
	else:
		return 0.0


def pr(preds, targets, coarse=False):
	preds = preds.cpu()
	targets = targets.cpu()
	if coarse:
		targets_process = dilate_target(targets, kernel_size=2)
	else:
		targets_process = targets
	TP = (preds * targets_process).sum().item() * 1.0
	preds = preds.sum().item()
	gts = targets.sum().item()
	# print("TP, preds, gts", TP, preds, gts)
	if TP:
		P = round(min(TP / preds, 1), 5)
		R = round(min(TP / gts, 1), 5)
	else:
		P = 0
		R = 1
	f1 = (2*P*R) / (P + R)        
	return P, R, f1

