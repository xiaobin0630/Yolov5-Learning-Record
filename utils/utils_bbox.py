import numpy as np
import torch
from torchvision.ops import nms

# 框的解码类
class DecodeBox():
    def __init__(self,anchors,num_classes,input_shape,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # 20x20的特征层对应的anchor是[116,90],[156,198],[373,326]
        # 40x40的特征层对应的anchor是[30,61],[62,45],[59,119]
        # 80x80的特征层对应的anchor是[10,13],[16,30],[33,23]

        self.anchors_mask = anchors_mask
    def decode_box(self,inputs):
        outputs = []
        # 逐层解码
        for i,input in enumerate(inputs):
            # 输入的input有三个,他们的shape是
            # batch_size = 1
            # batch_size,3 * (4 + 1 + 80),20,20
            # batch_size,255,40,40
            # batch_size,255,80,80
            # 取出batch_size 特征的高宽
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            # 求出一个特征图上的点相对与原图的大小 当输入input_shape为[640,640] 32,16,8
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # 获取相对于特征层的anchors大小
            scaled_anchors = [(anchor_width / stride_w,anchor_height / stride_h)for anchor_width,anchor_height in self.anchors[self.anchors_mask[i]]]

            # 逐个将特征图转化为
            # batch_size,3,20,20,85
            # batch_size,3,40,40,85
            # batch_size,3,80,80,85
            prediction = input.view(batch_size,len(self.anchors_mask[i]),self.bbox_attrs,input_height,input_width).permute(0,1,3,4,2).contiguous()

            # 提取先验框中心位置的调整参数
            x = torch.sigmoid(prediction[...,0])
            y = torch.sigmoid(prediction[...,1])
            # 提取先验框的宽高的调整参数
            w = torch.sigmoid(prediction[...,2])
            h = torch.sigmoid(prediction[...,3])

            # 获取置信度,是否有物体
            conf = torch.sigmoid(prediction[...,4])
            # 种类置信度
            pred_cls = torch.sigmoid(prediction[...,5:])

            # 判断是否使用gpu,不是就用cpu
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成网格,先验框中心,网格左上角坐标
            grid_x = torch.linspace(0,input_width - 1,input_width).repeat(input_height,1).repeat(
                batch_size * len(self.anchors_mask[i]),1,1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0,input_height - 1,input_height).repeat(input_width,1).t().repeat(
                batch_size * len(self.anchors_mask[i]),1,1).view(y.shape).type(FloatTensor)

            # 按照网格生成先验框的宽高 batch_size,3,20,20
            anchor_w = FloatTensor(scaled_anchors).index_select(1,LongTensor([0])) # shape(3,1)
            anchor_h = FloatTensor(scaled_anchors).index_select(1,LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size,1).repeat(1,1,input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size,1).repeat(1,1,input_height * input_width).view(h.shape)

            # 利用预测结果对先验框进行调整
            # x 0~1 -> 0~2 -> -0.5~1.5 ->负责一定范围的目标预测
            # y 0~1 -> 0~2 -> -0.5~1.5 ->负责一定范围的目标预测
            # w 0~1 -> 0~2 -> 0~4 -> 先验框的宽高调节范围为0~4倍
            # h 0~1 -> 0~2 -> 0~4 -> 先验框的宽高调节范围为0~4倍
            pred_boxes = FloatTensor(prediction[...,:4].shape)# batch_size,3,20,20,5
            pred_boxes[...,0] = x.data * 2.0 - 0.5 + grid_x
            pred_boxes[...,1] = y.data * 2.0 - 0.5 + grid_y
            pred_boxes[...,2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[...,3] = (h.data * 2) ** 2 * anchor_h

            # 将输出结果归一化成小数 1 1200 4  1 1200 1 1 1200 80
            _scale = torch.Tensor([input_width,input_height,input_width,input_height]).type(FloatTensor)# [20,20,20,20]
            output = torch.cat((pred_boxes.view(batch_size,-1,4) / _scale,conf.view(batch_size,-1,1),
                                pred_cls.view(batch_size,-1,self.num_classes)),-1)
            # 1 1200 85
            outputs.append(output.data)
        return outputs
    def yolo_correct_boxes(self,box_xy,box_wh,input_shape,image_shape,letterbox_image):
        # 把y放前面为后续预测框和图像的宽高进行相乘
        # 坐标交换
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape) / 2.0 / input_shape
            scale = input_shape / new_shape
            # 对框坐标与宽高进行偏移处理
            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        # 左上角
        box_mins = box_yx - (box_hw / 2.0)
        # 右下角
        box_maxes = box_yx + (box_hw / 2.0)
        # 左上角与右下角坐标拼接
        boxes = np.concatenate([box_mins[...,0:1],box_mins[...,1:2],box_maxes[...,0:1],box_maxes[...,1:2]],axis=-1)
        # 将相对于特征图的预测框对应到图像的预测框
        boxes *= np.concatenate([image_shape,image_shape],axis=-1)
        return boxes



    def non_max_suppression(self,prediction,num_classes,input_shape,image_shape,letterbox_image,conf_thres=0.5,
                            nms_thres=0.4):
        # 将预测结果的格式转化为左上角右下角的格式
        # prediction [batch_size,num_anchors,85]
        # 按照predication的shape创建一个多维矩阵
        box_corner = prediction.new(prediction.shape)
        box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2] / 2
        box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3] / 2
        box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2] / 2
        box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3] / 2
        # 回传
        prediction[:,:,:4] = box_corner[:,:,:4]

        output = [None for _ in range(len(prediction))]
        # 降维
        for i,image_pred in enumerate(prediction):
            # 对象种类预测部分取max 对1维度
            # image_pred shape [num_anchors,85]
            class_conf,class_pred = torch.max(image_pred[:,5:5 + num_classes],1,keepdim=True)
            # class_conf [num_anchors,1] 种类置信度
            # class_pred [num_anchors,1] 种类

            # 利用置信度进行第一轮筛选
            conf_mask = (image_pred[:,4] * class_conf[:,0] >= conf_thres).squeeze()

            # 根据置信度进行预测结果的筛选 将conf_mask为False的行去掉
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            # 判断如果筛选后没有框了就不必进行下面的处理了
            if not image_pred.size(0):
                continue

            # detections [num_anchors,7]
            # 7的内容为: x1,y1,x2,y2,obj_conf,class_conf,class_pred
            detections = torch.cat((image_pred[:,:5],class_conf.float(),class_pred.float()),1)
            # 获取预测结果中包含的所有种类
            unique_labels = detections[:,-1].cpu().unique()

            # 判断是否使用GPU,有就将数据放在GPU上
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 获取某一类得分筛选后的全部的预测结果
            for c in unique_labels:
                # 提取预测为c的类别的预测数据
                detections_class = detections[detections[:,-1] == c]

                # 使用官方自带的mns 要传入的参数有 框的坐标,框的得分,mns_thres门限值
                keep = nms(
                    detections_class[:,:4],
                    detections_class[:,4] * detections_class[:,5],
                    nms_thres
                )
                # 提取最后的结果
                max_detections = detections_class[keep]
                # 如果列表i中是None 直接存入output[i] 如不是空,就cat拼接
                output[i] = max_detections if output[i] is None else torch.cat((output[i],max_detections))
            # 如果output[i]不是None
            if output[i] is not None:
                # 将数据边为array类型
                output[i] = output[i].cpu().numpy()
                # 将左上角右下角坐标边为中心宽高形式坐标
                box_xy,box_wh = (output[i][:,0:2] + output[i][:,2:4]) / 2,output[i][:,2:4] - output[i][:,0:2]
                output[i][:,:4] = self.yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape,letterbox_image)
        return output











