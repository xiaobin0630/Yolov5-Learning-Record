import torch
import torch.nn as nn

from nets.CSPDarknet import C3,Conv,CSPDarknet
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict  = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.backbone_name = backbone
        if backbone == "cspdarknet":
            # 3个特征层
            # 80,80,256
            # 40,40,512
            # 20,20,1024
            self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)


        # 上采样 长宽翻倍 W * scale_factor ,H * scale_factor
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # (80,80,256) -> (80,80,3 * (5 + num_classes))
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # (40,40,512) -> (40,40,3 * (5 + num_classes))
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # (20,20,1024) -> (20,20,3 * (5 + num_classes))
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)
    def forward(self,x):
        # backbone  (80,80,256) (40,40,512) (20,20,1024)
        feat1, feat2, feat3 = self.backbone(x)

        # (20,20,1024) -> (20,20,512)
        P5 = self.conv_for_feat3(feat3)
        # upsample (20,20,512) -> (40,40,512)
        P5_upsample = self.upsample(P5)
        # cat P5_upsample feature2 (40,40,512) (40,40,512) -> (40,40,1024)
        P4 = torch.cat([P5_upsample, feat2], 1)
        # (40,40,1024) -> (40,40,512)
        P4 = self.conv3_for_upsample1(P4)

        # (40,40,512) -> (40,40,256)
        P4  = self.conv_for_feat2(P4)
        # (40,40,256) -> (80,80,256)
        P4_upsample = self.upsample(P4)
        # cat P4_upsample featrue1 (80,80,256) (80,80,256) -> (80,80,512)
        P3 = torch.cat([P4_upsample, feat1], 1)
        # (80,80,512) -> (80,80,256)
        P3  = self.conv3_for_upsample2(P3)

        # (80,80,256) -> (40,40,256)
        P3_downsample = self.down_sample1(P3)
        # (40,40,256) cat (40,40,256) -> (40,40,512)
        P4 = torch.cat([P3_downsample, P4], 1)
        # (40,40,512) -> (40,40,512)
        P4 = self.conv3_for_downsample1(P4)

        # (40,40,512) -> (20,20,512)
        P4_downsample = self.down_sample2(P4)
        # (20,20,512) cat (20,20,512) -> (20,20,1024)
        P5 = torch.cat([P4_downsample, P5], 1)
        # (20,20,1024) -> (20,20,1024)
        P5 = self.conv3_for_downsample2(P5)

        # 第三个特征层 (bs,256,80,80) -> (bs,75,80,80)
        out2 = self.yolo_head_P3(P3)

        # 第二个特征层 (bs,512,40,40) -> (bs,75,40,40)
        out1 = self.yolo_head_P4(P4)

        # 第一个特征层 (bs,1024,20,20) -> (bs,75,20,20)
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    anchors_mask = [[10, 13, 16], [30, 33, 23], [30, 61, 62], [10, 13, 16], [30, 33, 23], [30, 61, 62]]
    img = torch.rand([1,3, 640, 640])
    phi = 1
    pretrained = 1
    model = YoloBody(anchors_mask,num_classes=20,phi="s",backbone='cspdarknet',pretrained=True,input_shape=[640,640])

    out = model(img)
    for i in out:
        print(i.shape)




























