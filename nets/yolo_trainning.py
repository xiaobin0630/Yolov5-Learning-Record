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

    # l代表那一层的特征图序号,input代表特征图,targets真实标签,y_true代表网络应该预测为的真实模板
    def forward(self,l,input,targets=None,y_true=None):
        # l代表第几的特征层
        # input的shape为 (bs, 3 * (5 + num_classes), 20, 20)
        #               (bs, 3 * (5 + num_classes), 40, 40)
        #               (bs, 3 * (5 + num_classes), 80, 80)
        # targets 真实框的标签情况 [batch_size,num_gt,5]

        # 获取图片数量,特征图的高与宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原图片上多少个像素点
        # [640,640]
        # 特征层20x20 ,一个特征点对应原来的图片上32个像素
        # 特征层40x40 ,一个特征点对应原来的图片上16个像素
        # 特征层20x20 ,一个特征点对应原来的图片上8个像素
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        # 获取相对于特征层的scaled_anchors
        scaled_anchors = [(a_w / stride_w,a_h / stride_h)for a_w,a_h in self.anchors]

        # input输入有三类 (bs,3 * (5+num_classes),20,20)(bs,3 * (5+num_classes),40,40)(bs,3 * (5+num_classes),80,80)
        # (bs,3 * (5+num_classes),20,20) -> (bs,3,5+num_clsses,20,20) -> (bs,3,20,20,5+num_classes)
        prediction = input.view(bs,len(self.anchors_mask[l]),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()
        # 获取先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        # 获取先验框的宽高调整参数
        w = torch.sigmoid(prediction[...,2])
        h = torch.sigmoid(prediction[...,3])
        # 获取置信度,是否有物体
        conf = torch.sigmoid(prediction[...,4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[...,5:])

        # 将预测结果进行解码,判断预测结果和真实值的重合程度
        pred_boxes = self.get_pred_boxes(l,x,y,h,w,targets,scaled_anchors,in_h,in_w)

        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0
        # 计算有多少个真实框
        n = torch.sum(y_true[...,4] == 1)
        if n != 0:
            # 计算预测结果和真实结果的giou
            giou = self.box_giou(pred_boxes,y_true[...,:4]).type_as(x)

    def box_giou(self,b1,b2):
        # 预测框左上右下角
        # 预测框的中心坐标
        b1_xy = b1[...,:2]
        b1_wh = b1[...,2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy -




    def get_pred_boxes(self,l,x,y,h,w,targets,scaled_anchors,in_h,in_w):
        # 计算有多少张图片
        bs = len(targets)
        # 生成网格,先验框中心,网格左上角
        # (bs,3,20,20)
        grid_x = torch.linspace(0,in_w - 1,in_w).repeat(in_h,1).repeat(
            int(bs * len(self.anchors_mask[l])),1,1).view(x.shape).type_as(x)
        # (bs, 3, 20, 20)
        grid_y = torch.linspace(0,in_h - 1,in_h).repeat(in_w,1).repeat(
            int(bs * len(self.anchors_mask[l])),1,1).view(y.shape).type_as(x)

        # 生成先验框的宽高 [(),(),()]
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        # 转化为Tensor 提取scaled_anchors_l 宽 [[],[],[]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        # 转化为Tensor 提取scaled_anchors_l 高 [[],[],[]] shape (3,1)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        # (bs, 3, 20, 20)
        anchor_w = anchor_w.repeat(bs,1).repeat(1,1,in_h * in_w).view(w.shape)
        anchor_h = anchor_w.repeat(bs,1).repeat(1,1,in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心于宽高
        pred_boxes_x = torch.unsqueeze(x * 2.0 - 0.5 + grid_x,-1)
        pred_boxes_y = torch.unsqueeze(x * 2.0 - 0.5 + grid_y,-1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w,-1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],dim = -1)
        return pred_boxes

























