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
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0,in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        print(grid_y.shape)
        # 提取对应特征层的anchor宽高
        # scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        # # 提取在本特征层相对应特征层的大小的anchors
        # anchor_w = torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([0])).type_as(x)
        # print(anchor_w)
        # anchor_h = torch.Tensor(scaled_anchors_l).index_select(1,torch.LongTensor([1])).type_as(x)
        # anchor_w = anchor_w.repeat(bs,1).repeat(1, 1, in_h * in_w).view(w.shape)
        # print(anchor_w)
        # print(anchor_w.shape)
        # print(w.shape)
        # 预测结果
        # pred_boxes = self.get_pred_boxes(l, x, y, h, w, targets, scaled_anchors, in_h, in_w)
        # print("123")
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







if __name__ == "__main__":
    # x = torch.rand([1,3,20,20])
    # grid_x = torch.linspace(0, 19, 20).repeat(20,1).repeat(3,1,1).view(x.shape)
    # grid_y = torch.linspace(0, 19, 20).repeat(20,1).t().repeat(3,1,1).view(x.shape)
    # print(grid_x)
    # print(grid_y)
    # num_classes = 80, input_shape =[640,640] anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors_path = "model_data/yolo_anchors"
    anchors, num_anchors = get_anchors(anchors_path)
    num_classes = 80
    input_shape = [640, 640]
    cuda = False
    label_smoothing = 0
    print(anchors)
    yololoss = YOLOLoss(anchors,num_classes,input_shape,cuda,anchors_mask,label_smoothing)
    input = torch.rand((1,255,20,20))
    yololoss(1,input,None,None)
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
























































