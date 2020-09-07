#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/09/01
author: relu
"""

import argparse
import os.path as osp

root_dir = '/home/jovyan/jupyter/benchmark_images/faceu/face_detection'
weights_dir = '/home/jovyan/jupyter/checkpoints_zoo/face-detection'

def train_args():
    parser = argparse.ArgumentParser(description='FaceBoxes Training')

    # platform
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--workers', type=int,  default=0)

    # box-loss
    parser.add_argument('--num_classes', type=int,   default=2)
    parser.add_argument('--overlap_th',  type=float, default=0.35)
    parser.add_argument('--np_ratio',    type=float, default=7.0)
    parser.add_argument('--variance',    type=list,  default=[0.1, 0.2])
    parser.add_argument('--loc_weight',  type=float, default=2.0)

    # optimizor
    parser.add_argument('--resume',     type=str,   default='')
    parser.add_argument('--start_epoch',type=int,   default=0)
    parser.add_argument('--end_epoch',  type=int,   default=300)
    parser.add_argument('--batch_size', type=int,   default=64)   # TODO
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--gamma',      type=float, default=0.1)
    parser.add_argument('--momentum',   type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--milestones', type=list,  default=[200, 250])

    # files & path
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--print_freq',type=int, default=50)   # (n_rows=12880, bz=32, n_iters=403)
    parser.add_argument('--is_debug',  type=bool,default=False)  # TODO
    parser.add_argument('--voc_dir',   type=str, default=osp.join(root_dir, 'widerface'))
    parser.add_argument('--save_to',   type=str, default=osp.join(weights_dir, 'faceboxes'))
    args = parser.parse_args()
    return args


def infer_args():
    parser = argparse.ArgumentParser(description='FaceBoxes inference')
    parser.add_argument('--weights',   type=str,   default='checkpoint/faceboxes.pth')
    parser.add_argument('--cpu',       type=bool,  default=True)
    parser.add_argument('--dataset',   type=str,   default='PASCAL', choices=['AFW', 'PASCAL', 'FDDB'])
    parser.add_argument('--conf_thres',type=float, default=0.05)
    parser.add_argument('--top_k',     type=int,   default=5000)
    parser.add_argument('--nms_thres', type=float, default=0.3)
    parser.add_argument('--keep_top_k',type=int,   default=750)
    parser.add_argument('--show_image',type=bool,  default=True)
    parser.add_argument('--vis_thres', type=float, default=0.3)
    args = parser.parse_args()
    return args
