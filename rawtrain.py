#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/09/01
author: relu
"""


import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import math
from models.faceboxes import FaceBoxes


img_dim = 1024 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']

net = FaceBoxes('train', img_dim, num_classes)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())

    n_iters_per_epoch = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * n_iters_per_epoch

    stepvalues = (200 * n_iters_per_epoch, 250 * n_iters_per_epoch)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * n_iters_per_epoch
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % n_iters_per_epoch == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), save_folder + 'FaceBoxes_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, n_iters_per_epoch)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} '.format(epoch, max_epoch, (iteration % n_iters_per_epoch) + 1, n_iters_per_epoch, iteration + 1, max_iter, loss_l.item(), loss_c.item()))

    torch.save(net.state_dict(), save_folder + 'Final_FaceBoxes.pth')

if __name__ == '__main__':
    train()
