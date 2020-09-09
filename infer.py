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
        # weights = weights['backbone']
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
    
    
    def _easy_vis(self, img, dets, save_to):
        
        for box in dets:
            if box[4] < self.args.vis_thres:
                continue
            text = "{:.4f}".format(box[4])
            box = list(map(int, box))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            cx, cy = box[0], box[1] + 12
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.imwrite(save_to, img)
        
    
    def _single_infer(self, img = None, save_to = ''):
        
        img_copy = img.copy()
        im_height, im_width, _ = img.shape
        img = np.float32(img)
        with torch.no_grad():
            scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0).to(faceu.device)
            loc, conf = self.model(img)  # forward pass
        priors = clib.PriorBox(img_size=(im_height, im_width)).forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.args.conf_thres)[0]
        boxes, scores = boxes[inds], scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes, scores = boxes[order], scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.args.nms_thres,force_cpu=self.args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        
        if self.args.save_flag and len(save_to) > 0:
            self._easy_vis(img_copy, dets, save_to)            
        return dets
    
    def _batch_infer(self, csv_file = None):
        pass
        

        
cp_dir = '/home/jovyan/jupyter/checkpoints_zoo/face-detection/faceboxes/sota.pth'
def infer_args():
    parser = argparse.ArgumentParser(description='config of inference for FaceBoxes')
    parser.add_argument('--weights',   type=str,   default='checkpoint/faceboxes.pth')  # default = 'checkpoint/faceboxes.pth'
    parser.add_argument('--cpu',       type=bool,  default=True)
    parser.add_argument('--dataset',   type=str,   default='PASCAL', choices=['AFW', 'PASCAL', 'FDDB'])
    parser.add_argument('--conf_thres',type=float, default=0.05)
    parser.add_argument('--top_k',     type=int,   default=5000)
    parser.add_argument('--nms_thres', type=float, default=0.3)
    parser.add_argument('--keep_top_k',type=int,   default=750)
    parser.add_argument('--show_image',type=bool,  default=True)
    parser.add_argument('--vis_thres', type=float, default=0.3)
    parser.add_argument('--save_flag', type=bool,  default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    faceu = FaceBoxesInfer(infer_args())
    img = cv2.imread('./imgs/worlds-largest-selfie.jpg')
    save_to = 'imgs/detected_face.jpg'
    faceu._single_infer(img, save_to)
