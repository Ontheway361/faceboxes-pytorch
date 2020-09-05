#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/09/02
author: relu
"""

import os
import time
import torch
import shutil
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F
from torch.utils.data import DataLoader

import corelib as clib
import dataset as dlib
from config import train_args
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

from IPython import embed

class FaceBoxesTrainer(object):

    def __init__(self, args):

        self.args     = args
        self.model    = dict()
        self.data     = dict()
        self.result   = dict()
        self.use_cuda = args.use_gpu and torch.cuda.is_available()


    def _model_loader(self):

        self.model['facebox']   = clib.FaceBoxes(phase='train', num_classes=2)
        self.model['boxloss']   = clib.MultiBoxLoss(self.args)
        self.model['optimizer'] = torch.optim.SGD(
                                      self.model['facebox'].parameters(),
                                      lr=self.args.lr,
                                      momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], \
                                      milestones=self.args.milestones, gamma=self.args.gamma)
        if self.use_cuda:
            self.model['facebox'] = self.model['facebox'].cuda()
            if len(self.args.gpu_ids) > 1:
                self.model['facebox'] = torch.nn.DataParallel(self.model['facebox'], device_ids=self.args.gpu_ids)
                print('Parallel mode is going ...')

        if len(self.args.resume) > 3:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['facebox'].load_state_dict(checkpoint['backbone'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):
        self.data['train_loader'] = DataLoader(
                                        dlib.VOCDetection(self.args),
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        num_workers=self.args.workers,
                                        collate_fn=dlib.detection_collate)
        print('Data loading was finished ...')


    def _train_one_epoch(self):
        self.model['facebox'].train()
        loss_l_list, loss_c_list = [], []
        for idx, (imgs, gtys) in enumerate(self.data['train_loader']):
            
            imgs.requires_grad    = False
            gtys[0].requires_grad = False
            gtys[1].requires_grad = False
            if self.use_cuda:
                imgs = imgs.cuda()
                gtys = [anno.cuda() for anno in gtys]
            fout = self.model['facebox'](imgs)
            loss_l, loss_c = self.model['boxloss'](fout, gtys)
            loss = self.args.loc_weight * loss_l + loss_c
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            loss_l_list.append(loss_l.item())
            loss_c_list.append(loss_c.item())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %02d|%02d, iter : %04d|%04d,  loss_l : %.4f, loss_c : %.4f' % (self.result['epoch'], self.args.end_epoch, \
                    idx + 1, len(self.data['train_loader']), np.mean(loss_l_list), np.mean(loss_c_list)))
        ave_loss_l = np.mean(loss_l_list)
        ave_loss_c = np.mean(loss_c_list)
        print('ave_loss_l : %.4f, ave_loss_c : %.4f' % (ave_loss_l, ave_loss_c))
        return (ave_loss_l, ave_loss_c)


    def _save_weights(self, lossinfo):
        ''' save the weights during the process of training '''

        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)

        freq_flag = self.result['epoch'] % self.args.save_freq == 0
        sota_flag = self.result['min_loss'] > sum(lossinfo)
        save_name = '%s/epoch_%02d_loss_%.4f_loss_l_%.4f_loss_c_%.4f.pth' % \
                         (self.args.save_to, self.result['epoch'], sum(lossinfo), lossinfo[0], lossinfo[1])
        if sota_flag:
            save_name = '%s/sota.pth' % self.args.save_to
            self.result['min_loss'] = sum(lossinfo)
            print('%s Yahoo, SOTA model was updated %s' % ('*'*16, '*'*16))

        if sota_flag or freq_flag:
            torch.save({
                'epoch'   : self.result['epoch'],
                'backbone': self.model['facebox'].state_dict(),
                'loss'    : sum(lossinfo)}, save_name)

        if sota_flag and freq_flag:
            normal_name = '%s/epoch_%02d_loss_%.4f_loss_l_%.4f_loss_c_%.4f.pth' % \
                             (self.args.save_to, self.result['epoch'], sum(lossinfo), lossinfo[0], lossinfo[1])
            shutil.copy(save_name, normal_name)


    def _training(self):

        self.result['min_loss'] = 1e5
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()
            self.result['epoch'] = epoch
            loss = self._train_one_epoch()
            self.model['scheduler'].step()
            finish_time = time.time()
            print('single epoch costs %.4f mins' % ((finish_time - start_time) / 60))
            self._save_weights(loss)
            if self.args.is_debug:
                break


    def main_runner(self):
        self._model_loader()
        self._data_loader()
        self._training()


if __name__ == "__main__":

    fas = FaceBoxesTrainer(args=train_args())
    fas.main_runner()
