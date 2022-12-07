import numpy as np
import torch


def bbox_iou(b1, b2):
    # 提取最高分框的左上角与右下角坐标
    b1_min = b1[:, :2]
    b1_max = b1[:, 2:4]
    b1_wh = b1_max - b1_min

    # 提取非最大分的框
    b2_mins = b2[:, :2]
    b2_maxs = b2[:, 2:4]
    b2_wh = b2_maxs - b2_mins

    intersect_mins = torch.max(b1_min, b2_mins)
    intersect_maxs = torch.min(b1_max, b2_maxs)
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    ious = intersect_area / union_area
    return ious
# yolo网络输出特征层的解码类

class DecodeBox():
    def __init__(self, anchors,num_classes,input_shape,anchors_mask=[[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        # 20x20 特征层对应anchor大小[116,90],[156,198],[373,326]
        # 40x40 特征层对应anchor大小[30,61],[62,45],[59,119]
        # 80x80 特征层对应anchor大小[10,13],[16,30],[33,23]
        self.anchors_mask = anchors_mask

    def decode_box(self,inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # 输入的特征层共有三个
            # batch_size = 1
            # (batch_size, 3 * (4 + 1 + 80), 20, 20)
            # batch_size, 255, 40, 40
            # batch_size, 255, 80, 80
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # 计算特征层上一点对应原图多少的步距
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # 获得相对于特征层的大小先验框大小
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)for anchor_width,anchor_height in self.anchors[self.anchors_mask[i]]]

            # (batch_size, 3 * (4 + 1 + 80), 20, 20)
            prediction = input.view(batch_size,len(self.anchors_mask[i]),self.bbox_attrs,input_height,input_width).permute(0,1,3,4,2).contiguous()

            # 获取先验框中心调整参数
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            # 获取先验框宽高调整参数
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])

            # 是否有物体 置信度
            conf = torch.sigmoid(prediction[...,4])

            # 种类置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 创建网格
            grid_x = torch.linspace(0,input_width - 1,input_width).repeat(input_width, 1).repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0,input_height - 1,input_height).repeat(input_width, 1).t().repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # 按照网格生成先验框的宽高
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            pred_boxes = FloatTensor(prediction[...,:4].shape)
            pred_boxes[..., 0] = x.data * 2.0 - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2.0 - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            _scale = torch.Tensor([input_width,input_height,input_width,input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1,self.num_classes)), -1)
            outputs.append(output.data)
        return outputs
    def yolo_correct_boxes(self,box_xy,box_wh,input_shape,image_shape,letterbox_image):
        # 吧y轴放前面方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        # 创建两个矩阵
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2.0 /input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxs = box_yx + (box_hw / 2.0)

        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[...,0:1], box_maxs[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape,image_shape],axis=-1)
        return boxes

    def non_max_suppression(self,prediction,num_classes,input_shape,image_shape,letterbox_image,conf_thres=0.5,mns_thres=0.4):
        # 将中心宽高转化为左上角与右下角格式

        box_corner = prediction.new(prediction.shape)
        box_corner[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
        box_corner[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
        box_corner[..., 2] = prediction[..., 0] + prediction[..., 2] / 2
        box_corner[..., 3] = prediction[..., 1] + prediction[..., 3] / 2
        prediction[..., :4] = box_corner[..., :4]
        # (bs,anchors,85)
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 取得类别最大率与最大率的类别索引
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # 进行置信度进行第一次筛选
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            # concat [num_anchors, 7]
            # x1, y1, x2, y2, obj_conf, class_conf, class_pred
            detections = torch.cat((image_pred[:, :5], class_conf.float(),class_pred.float()), 1)

            # 获取预测结果中包含的所有种类
            unique_labels = detections[:,-1].cpu().unique()

            # 使用GPU
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 按类别进行NMS
            for c in unique_labels:
                # 得到一个类别的所有预测结果
                detections_class = detections[detections[:,-1] == c]

                # 按照存在物体的置信度排序
                # 获取从小到大的索引
                _, conf_sort_index = torch.sort(detections_class[:, 4] * detections_class[:, 5], descending=True)

                # 按索引排序排序
                detections_class = detections_class[conf_sort_index]

                # 进行nms
                max_detections = []
                # 只有还有anchors数就一直循环
                while detections_class.size(0):
                    # 取出这类的最高置信度的
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # 如果只剩下一个框就推出循环
                    if len(detections_class) == 1:
                        break

                    ious = bbox_iou(max_detections[-1],detections_class[1:])
                    detections_class = detections_class[1:][ious < mns_thres]
                # 堆叠
                max_detections = torch.cat(max_detections).data

                # 将本批次的mns后的预测框放入对应的列表位置处
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            # 判断本批次预测框是否不为None:
            if output[i] is not None:
                # 提取出来本批次的预测框转化为numpy格式
                output[i] = output[i].cpu().numpy()
                # 将左上角与右下角坐标格式转化为中心宽高格式
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:,2:4]) / 2,output[i][:,2:4] - output[i][:,0:2]
                output[i][:,:4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output












if __name__ == "__main__":
    # output = torch.rand([10,18])
    # num = 10
    # # print(output)
    # # print('*****************')
    # # print(output[:,5:5 + num ])
    # class_conf, class_pred = torch.max(output[:,5:5 + num],1,keepdim=True)
    # # print(class_conf, "\n", class_pred)
    # image_ = torch.rand([10,5])
    # # print(image_)
    # print(class_conf[:, 0].shape)
    # a = (image_[:,4] * class_conf[:, 0] >= 0.5)
    # # print(image_ * class_conf[:,0])
    # print(a)
    # print(image_ * class_conf[:, 0] >= 0.5)
    # print((image_ * class_conf[:, 0] >= 0.5).squeeze())
    # output = torch.rand([1,2550,85])
    # box_corner = output.new(output.shape)
    # box_corner[..., 0] = output[..., 0] - output[..., 2] / 2
    # box_corner[..., 1] = output[..., 1] - output[..., 3] / 2
    # box_corner[..., 2] = output[..., 0] + output[..., 2] / 2
    # box_corner[..., 3] = output[..., 1] + output[..., 3] / 2
    # output[...,:4] = box_corner[..., :4]
    # for i, image_pred in enumerate(output):
    #     print(image_pred.size(0))
    # data = torch.rand(10,5)
    # print(data[1], "\n", data[2],"\n", data[1] * data[2])
    # value, index = torch.sort(data[1] * data[2], descending=True)
    # print(value, "\n", index)
    # print(data[0].unsqueeze(0))
    # b1 = torch.Tensor([[20,20,100,100,1,1,1]])
    #
    # b2 = torch.Tensor([[50,50,150,150,1,1,1],[40,40,120,120,1,1,1]])
    #
    # ious = bbox_iou(b1,b2)
    # print(ious)
    # data = torch.cat([b1, b2])
    # print(data)
    # print(data)
    # print(ious < 0.2)
    # print(data[1:][ious < 0.2])

    # w = torch.rand([2,300,22,22])
    # output = [None for _ in range(len(w))]
    # print(output)
    a = np.array([[1,2,3,4,5,6,7,8,9],
                  [1,2,3,4,5,6,7,8,9]])

    print(a[:, ::-1])

















    # print('123')
    # [[10.  13.]
    #  [16.  30.]
    #  [33.  23.]
    #  [30.  61.]
    #  [62.  45.]
    #  [59. 119.]
    #  [116. 90.]
    #  [156. 198.]
    #  [373. 326.]]