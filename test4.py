import random

import numpy as np
import torch
import torch.nn as nn
from utils.utils import get_anchors
class YOLOLoss(nn.Module):
    def __init__(self,anchors,num_classes,input_shape,cuda,anchors_mask = [[6,7,8], [3,4,5], [0,1,2]],label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        # 20x20 的特征层对应的anchor是[[116,90],[156,198],[373,326]]
        # 40x40 的特征层对应的anchor是[[30,61],[62,45],[59,119]]
        # 80x80 的特征层对应的anchor是[[10,13],[16,30],[33,23]]
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing
        self.threshold = 4
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] + input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda
    def forward(self, l, input, targets=None, y_true=None):
        # l表示第几个有效特征层
        # input 的shape 为 (bs,3 * (5 + num_classes), 20, 20)
        #                 (bs,3 * (5 + num_classes), 40, 40)
        #                 (bs,3 * (5 + num_classes), 80, 80)
        # targets         真实框 [batch_size,num_gt,5]
        # 获取图片数量,特征层的高宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # 计算一个特征点对应原来的图片上多少个像素点
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        # print(stride_h,stride_w)
        # 获得相对于特征层的anchor大小
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w,a_h in self.anchors]
        # print(scaled_anchors)
        # input shape
        # (bs, 3 * (5 + num_classes), 20, 20) -> (bs,3,5 + num_classes, 20 ,20) -> (bs, 3, 20, 20, 5 + num_classes)
        # (bs, 3, 20, 20, 5 + num_classes)
        # (bs, 3, 40, 40, 5 + num_classes)
        # (bs, 3, 80, 80, 5 + num_classes)
        # print(input.shape)
        prediction = input.view(bs,len(self.anchors_mask[l]),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()
        # print(prediction.shape)
        # (bs,3,20,20)
        # 取先验框的中心位置调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # 获得置信度,是否有物体
        conf = torch.sigmoid(prediction[...,4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[...,5:])

        # 预测解码
        pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0

        # 统计正样本的个数
        n = torch.sum(y_true[...,4] == 1)

        if n != 0:# pred_boxes (1,3,20,20,4)  y_true[...,:4])
            # 计算 真实框与对应的预测框的giou
            giou = self.box_giou(pred_boxes,y_true[...,:4]).type_as(x)
            # 计算计算真实框与预测框的损失
            loss_loc = torch.mean((1 - giou)[y_true[...,4] == 1])
            # 计算分类loss
            loss_cls    = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            # 对不同的loss取不同的比重相加
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            # 计算置信度的loss
            tobj = torch.where(y_true[..., 4] == 1,giou.detach().clamp(0),torch.zeros_like(y_true[...,4]))
        else:
            # 当没有真实框时
            tobj = torch.zeros_like(y_true[..., 4])
        # 计算是否又物体的置信度loss
        loss_conf = torch.mean(self.BCELoss(conf,tobj))
        # 对loss_conf乘以比重后相加
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss



    def BCELoss(self,pred,target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred,epsilon,1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def clip_by_tensor(self,t,t_min,t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result


    def smooth_labels(self,y_true,label_smoothing,num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    # (bs, 3, 20, 20, 5 + num_classes)

    def box_giou(self, b1, b2):
        # 计算预测的 xy与wh (bs,3,20,20,4)
        # pred_boxes x, y, w, h
        # y_true     x, y, w, h
        # 预测方面
        # 中心点坐标
        b1_xy = b1[..., :2]
        # 宽高
        b1_wh = b1[..., 2:4]

        b1_wh_half = b1_wh / 2
        # 左上点
        b1_mins = b1_xy - b1_wh_half
        # 右下点
        b1_maxs = b1_xy + b1_wh_half

        # 真实
        # 中心点坐标
        b2_xy = b2[..., :2]
        # 宽高
        b2_wh = b2[..., 2:4]

        b2_wh_half = b2_wh / 2
        # 左上点
        b2_mins = b2_xy - b2_wh_half
        # 右上点
        b2_maxs = b2_xy + b2_wh_half

        # 计算iou
        # 真实与预测的左上角的最大值
        internal_mins = torch.max(b1_mins, b2_mins)
        # 真实与预测的右下角的最小值
        internal_maxs = torch.min(b1_maxs, b2_maxs)
        # 计算长宽
        internal_wh = torch.max(internal_maxs - internal_mins, torch.zeros_like(internal_maxs))
        # 相交的面积
        internal_area = internal_wh[..., 0] * internal_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b1_wh[..., 1]
        union_area = b2_area + b1_area - internal_area
        iou = internal_area / union_area

        # 计算giou
        g_mins = torch.min(b1_mins, b2_mins)
        g_maxs = torch.max(b1_maxs, b2_maxs)
        g_wh = torch.max(g_maxs - g_mins, torch.zeros_like(g_maxs))
        g_area = g_wh[..., 0] * g_wh[...,1]
        giou = iou - (g_area - union_area) / g_area
        return giou











    def get_pred_boxes(self, l, x, y, h, w, targets,scaled_anchors, in_h, in_w):
        # 一共多少张图像
        bs = len(targets)

        # 生成网格,先验框中心,网络左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0,in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 提取对应特征层的anchor宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        # 提取在本特征层相对应特征层的大小的anchors
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([1])).type_as(x)
        # (bs,3,20,20)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x * 2.0 - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2.0 - 0.5 + grid_x, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h], dim = -1)
        return pred_boxes






# def clip_by_tensor(t,t_min,t_max):
#     t = t.float()
#     result = (t >= t_min).float() * t + (t < t_min).float() * t_min
#     result = (result <= t_max).float() * result + (result > t_max).float() * t_max
#     return result
if __name__ == "__main__":
    # x = torch.rand([1,3,20,20])
    # grid_x = torch.linspace(0, 19, 20).repeat(20,1).repeat(3,1,1).view(x.shape)
    # grid_y = torch.linspace(0, 19, 20).repeat(20,1).t().repeat(3,1,1).view(x.shape)
    # print(grid_x)
    # print(grid_y)
    # num_classes = 80, input_shape =[640,640] anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors_path = "model_data/yolo_anchors"
    # anchors, num_anchors = get_anchors(anchors_path)
    # num_classes = 80
    # input_shape = [640, 640]
    # cuda = False
    # label_smoothing = 0
    # print(anchors)
    # yololoss = YOLOLoss(anchors,num_classes,input_shape,cuda,anchors_mask,label_smoothing)
    # input = torch.rand((1,255,20,20))
    # yololoss(1,input,None,None)
    # anchors
    # [[10.  13.]
    #  [16.  30.]
    #  [33.  23.]
    #  [30.  61.]
    #  [62.  45.]
    #  [59. 119.]
    #  [116. 90.]
    #  [156. 198.]
    #  [373. 326.]]
    # a = torch.rand([16,3,20,20,15])
    # # print(a.shape)
    # b = torch.mean((1 - a)[a[...,1]>0.5])
    # print((a[...,1]>0.5).shape)
    # print((1 - a)[a[...,1]>0.5])

    # a = np.array(range(0,96))
    # c = np.array(range(0,96))
    # d = c.reshape([2,2,3,4,2])
    # b = a.reshape([2,2,3,4,2])
    # print(np.sum(d[...,0] > 10))
    # print(b)
    # print(b[d[...,0] > 10])
    # print(d[...,0] > 10)
    # print((d[...,0] > 10).shape)
    #print(d[...,0] > 10)
    # print(b[...,1].shape)
    # print(b[...,1]>9)
    # print(b[...,1])
    # print(b[...,1][b[...,1]>9])


    # random.seed(0)
    # torch.random.seed()
    # y_true = torch.eye(2,2)
    # y_pred = torch.rand([2,2])
    # print(y_true,'\n',y_pred)
    # result = - y_true * torch.log(y_pred) - (1.0 - y_true) * torch.log(1.0 - y_pred)
    # print(result)
    # - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    # t = 1e-7
    # new_pred = clip_by_tensor(y_pred,t,1-t)
    # print(new_pred)
    # print(1 * np.log(0.6870))

    # output = torch.randn(3,3)
    # print(output)
    # active = torch.sigmoid(output)
    # print(active)
    # x = 1.4054
    # x1 = -2.0636
    # x2 = -0.6076
    # y = 1/(1+np.exp(-x))
    # y1 = 1/(1+np.exp(-x1))
    # y2 = 1/(1+np.exp(-x2))
    # print(y,y1,y2)
    # target = torch.FloatTensor([[0,1,1],[1,1,1],[0,0,0]])
    # result = - target * torch.log(active) - (1.0 - target) * torch.log(1.0 - active)
    # print(result)
    # print(torch.mean(result))

    condition = torch.randn(3, 2)
    print(condition)
    x = torch.ones(3,2)
    print(x)
    y = torch.zeros(3,2)
    print(y)
    result = torch.where(condition > 0.5,x,y)
    print(result)






















































