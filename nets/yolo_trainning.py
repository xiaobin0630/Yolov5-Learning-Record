import math
from copy import deepcopy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
class YOLOLoss(nn.Module):
    def __init__(self,anchors,num_classes,input_shape,cuda,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]],label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4,1.0,4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape [1]) / (640 ** 2)
        self.cuda = cuda

    def forward(self,l,input,targets=None,y_true=None):
        # l代表第几的特征层
        # targets 真实框的标签情况 [batch_size,num_gt,5]

        # 获取图片数量,特征图的高与宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原图片上多少个像素点






















