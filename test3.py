import torch
import torch.nn as nn

from test2 import Conv,C3,CSPDarknet
class YoloBody(nn.Module):
    # anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    def __init__(self, anchors_mask, num_classes,phi,backbone='cspdarknet',pretrained=False,input_shape=[640,640]):
        super(YoloBody, self).__init__()
        # 用于控制 bottleneck 残差循环的次
        depth_dict = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33}
        # 用于控制通道数
        width_dict = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25}

        dep_mul,wid_mul = depth_dict[phi],width_dict[phi]

        # 计算基础的循环次数 与对应的通道数
        base_channels = int(wid_mul * 64) #
        base_depth = max(round(dep_mul * 3), 1)
        self.backbone_name  = backbone
        # CSPDarknet(base_channels, base_depth)
        if self.backbone_name == 'cspdarknet':
            self.backbone = CSPDarknet(base_channels,base_depth)

        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')

        # (bs, 1024, 20, 20) -> (bs, 512, 20, 20)
        self.conv_for_feat3 = Conv(base_channels * 16,base_channels * 8, 1, 1)
        # (bs, 1024, 40, 40) -> (bs, 512, 40, 40)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth,shortcut=False)

        # (bs, 512, 40, 40) -> (bs, 256, 40, 40)
        self.conv_for_feat2 = Conv(base_channels * 8,base_channels * 4 ,1 ,1)
        # (bs, 512, 80, 80) -> (bs, 256, 80, 80)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4,base_depth,shortcut=False)

        # (bs, 256, 80, 80) -> (bs, 256, 40, 40)
        self.down_sample1 = Conv(base_channels * 4,base_channels * 4, 3, 2)
        # (bs, 512, 40, 40) -> (bs, 512, 40, 40)
        self.conv3_for_downsample1 = C3(base_channels * 8,base_channels * 8,base_depth,shortcut=False)

        # (bs, 512, 40, 40) -> (bs, 512, 20, 20)
        self.down_sample2 = Conv(base_channels * 8,base_channels * 8, 3, 2)
        # (bs, 1024, 20, 20) -> (bs, 1024, 20, 20)
        self.conv3_for_downsample2 = C3(base_channels * 16,base_channels * 16,base_depth,shortcut=False)

        # (bs, 256, 80, 80) -> (bs, 3 * (5 + num_classes), 80, 80) -> (bs, 3 * (4 + 1 + num_classes), 80, 80)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # (bs, 512, 40, 40) -> (bs, 3 * (5 + num_classese), 40, 40) -> (bs, 3 * (4 + 1 + num_classes), 40, 40)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # (bs, 1024, 20, 20) -> (bs, 3 * (5 + num_classes), 20, 20) -> (bs, 3 * (4 + 1 + num_classes), 20, 20)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16,len(anchors_mask[0]) * (5 + num_classes), 1)
        # feat1(bs, 256, 80, 80)
        # feat2(bs, 512, 40, 40)
        # feat3(bs, 1024, 20, 20)

    def forward(self,x):
        # feat1 (bs, 256, 80, 80)
        # feat2 (bs, 512, 40, 40)
        # feat3 (bs, 1024, 20, 20)
        feat1, feat2, feat3 = self.backbone(x)

        # 降维 (bs, 1024, 20, 20) -> (bs, 512, 20, 20) P5
        P5 = self.conv_for_feat3(feat3)
        # 上采样                (bs, 512, 20, 20) -> (bs, 512, 40, 40) P5_upsample
        P5_upsample = self.upsample(P5)
        # 拼接 P5_upsample cat feat2 (bs, 512, 40, 40) cat (bs, 512, 40, 40) -> (bs, 1024, 40, 40) P4
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 卷积降维 (bs, 1024, 40, 40) -> (bs, 512, 40, 40)
        P4 = self.conv3_for_upsample1(P4)

        # 降维(bs, 512, 40, 40) -> (bs, 256, 40, 40) P4
        P4 = self.conv_for_feat2(P4)
        # 上采样           (bs, 256, 40, 40) -> (bs, 256, 80, 80)
        P4_upsample = self.upsample(P4)
        # 拼接 P4_upsample cat feat1 (bs, 256, 80, 80) cat (bs, 256, 80, 80) -> (bs, 512, 80, 80) P3
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 卷积降维 (bs, 512, 80, 80) -> (bs, 256, 80, 80) P3
        P3 = self.conv3_for_upsample2(P3)

        # 下采样 (bs, 256, 80, 80) -> (bs, 256, 40, 40) P3_downsample
        P3_downsample = self.down_sample1(P3)
        # 融合 P4 (bs, 256, 40, 40) cat  P3_downsample (bs, 256, 40, 40)  -> (bs, 512, 40, 40) P4
        P4 = torch.cat([P3_downsample,P4], 1)
        #  (bs, 512, 40, 40) -> (bs, 512, 40, 40) P4
        P4 = self.conv3_for_downsample1(P4)

        # 下采样 (bs, 512, 40, 40) -> (bs, 512, 20, 20)
        P4_upsample = self.down_sample2(P4)
        # 拼接 (bs, 512, 20, 20) P4_upsample cat P5 (bs, 512, 20, 20) -> (bs, 1024, 20, 20)
        P5 = torch.cat([P4_upsample,P5], 1)
        # (bs, 1024, 20, 20) -> (bs, 1024, 20, 20)
        P5 = self.conv3_for_downsample2(P5)

        # 第三个特征层 (bs, 256, 80, 80) -> (bs, 3 * (5 + num_classes), 80, 80)
        out2 = self.yolo_head_P3(P3)

        # 第二个特征层 (bs, 512, 40, 40) -> (bs, 3 * (5 + num_classes), 40, 40)
        out1 = self.yolo_head_P4(P4)

        # 第一个特征层 (bs, 1024, 20, 20) -> (bs, 3 * (5 + num_classes), 20, 20)
        out0 = self.yolo_head_P5(P5)

        return out0, out1, out2

if __name__ == "__main__":
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80
    phi ='s'
    image = torch.rand([1,3,640,640])

    model = YoloBody(anchors_mask, num_classes, phi, backbone='cspdarknet')
    predict = model(image)
    shape = []
    for i in predict:
        shape.append(i.shape)
    print(shape)



    

    
    





























