#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/09/01
author: relu
"""

import os
import cv2
import torch
import argparse
import numpy as np
import corelib as clib
from utils.nms_wrapper import nms
from utils.box_utils import decode
from utils.timer import Timer

torch.backends.cudnn.bencmark = True

from IPython import embed

class FaceBoxesInfer(object):
    
    def __init__(self, args):
        
        self.args   = args
        self.model  = clib.FaceBoxes(phase='test', num_classes=2)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self._model_loader()
        
    
    def _model_loader(self):
        if self.args.cpu:
            weights = torch.load(self.args.weights, map_location=lambda storage, loc:storage)
        else:
            device = torch.cuda.current_device()
            weights = torch.load(self.args.weights, map_location=lambda storage, loc:storage.cuda(device))
        if 'state_dict' in weights.keys():
            state_dict = weights['state_dict']
        else:
            state_dict = weights
        f = lambda x: x.split('module.', 1)[-1] if x.startswith('module.') else x
        state_dict = {f(key): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print('checkpoint was loaded successfully ...')
    
    
    def _single_infer(self, img = None):
        pass 
    
    
    def _batch_infer(self, csv_file = None):
        pass
        

def infer_args():
    parser = argparse.ArgumentParser(description='config of inference for FaceBoxes')
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




if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    args = infer_args()
    faceu = FaceBoxesInfer(args)
 
    img_raw = cv2.imread('./imgs/worlds-largest-selfie.jpg')
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(faceu.device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(faceu.device)
    loc, conf = faceu.model(img)  # forward pass
    priorbox = clib.PriorBox(img_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(faceu.device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data)
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.conf_thres)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_thres,force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    
    # show image
    if args.show_image:
        
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imwrite('./imgs/largest_faces.jpg', img_raw)
