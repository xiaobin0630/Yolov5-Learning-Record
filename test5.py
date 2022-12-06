import torch
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





if __name__ == "__main__":
    print('123')
    # [[10.  13.]
    #  [16.  30.]
    #  [33.  23.]
    #  [30.  61.]
    #  [62.  45.]
    #  [59. 119.]
    #  [116. 90.]
    #  [156. 198.]
    #  [373. 326.]]