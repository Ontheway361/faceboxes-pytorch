#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.box_utils as boxlib

from itertools import product
from torch.autograd import Variable

from IPython import embed


class PriorBox(object):
    def __init__(self, img_size = (1024, 1024)):
        super(PriorBox, self).__init__()
        self.clip      = False
        self.img_size  = img_size
        self.steps     = [32, 64, 128]
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.pyramid   = [[math.ceil(self.img_size[0] / step), \
                             math.ceil(self.img_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for idx, feat_size in enumerate(self.pyramid):

            min_sizes = self.min_sizes[idx]
            for i, j in product(range(feat_size[0]), range(feat_size[1])):

                for min_size in min_sizes:

                    s_kx = min_size / self.img_size[1]
                    s_ky = min_size / self.img_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[idx] / self.img_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y * self.steps[idx] / self.img_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[idx] / self.img_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y * self.steps[idx] / self.img_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[idx] / self.img_size[1]
                        cy = (i + 0.5) * self.steps[idx] / self.img_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class MultiBoxLoss(nn.Module):

    def __init__(self, args):
        super(MultiBoxLoss, self).__init__()
        self.args = args
        self.priors = PriorBox().forward()
        if args.use_gpu:
            self.priors = self.priors.cuda()

    def forward(self, predictions, targets):
        
        loc_data, conf_data = predictions
        num_images = loc_data.size(0)
        num_priors = (self.priors.size(0))

        loc_t = torch.Tensor(num_images, num_priors, 4)
        conf_t = torch.LongTensor(num_images, num_priors)
        priors = self.priors.data
        for idx in range(num_images):
            gt_boxes = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            loc, conf = boxlib.match(gt_boxes, priors, labels, self.args.overlap_th, self.args.variance)
            loc_t[idx], conf_t[idx] = loc, conf
        if self.args.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        # calculate location loss
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # (n_images, n_priors, 4)
        loc_p   = loc_data[pos_idx].view(-1, 4)
        loc_t   = loc_t[pos_idx].view(-1, 4)
        loss_l  = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') # (cx, cy, w, h)

        # hard negative mining
        batch_conf = conf_data.view(-1, self.args.num_classes) # conf_data.shape = (n_images, n_priors, 2)
        loss_c = boxlib.log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) # calculate the -log(softmax()) = - x_y + log_sum_exp 
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num_images, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.args.np_ratio * num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        scores = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.args.num_classes)
        np_gty = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(scores, np_gty, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + α * Lloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
