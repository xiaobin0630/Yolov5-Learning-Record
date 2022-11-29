import torch
import torch.nn as nn

# f(x) = x * sigmoid(x)
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# 3 -> 1  [2,4,6,8] -> [1,2,3,4]
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self,x):
        # (320,320,12) -> (320,320,64)

        return self.conv(
            # (640,640,3) -> (320,320,12)
            torch.cat(
                # 在0维度cat -> (320,320,12)
                [
                    # 每个取值后的shape为(320,320,3)
                    # 从每通道每行每列0,0开始,步长2

                    x[...,::2,::2],
                    # 从每通道每行每列1,0开始,步长2
                    x[...,1::2,::2],
                    # 从每通道每行每列0，1开始,步长2
                    x[...,::2,1::2],
                    # 从每通道每行每列1，1开始,步长2
                    x[...,1::2,1::2]
                ],1
            )
        )
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        # 512
        c_ = int(c2 * e)
        # (160, 160, 512)
        self.cv1 = Conv(c1, c_, 1, 1)
        #
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # in_c,out_c 128 128
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        # hidden_channels 512
        c_ = int(c2 * e)
        # 1x1卷积进行提取通道数为过去4倍
        # (160,160,128) -> (160,160,512)
        self.cv1 = Conv(c1, c_, 1, 1)
        # (160,160,128) -> (160,160,512)
        self.cv2 = Conv(c1, c_, 1, 1)
        # (160,160,1024) -> (160,160,128)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    def forward(self,x):
        # self.conv3 (160,160,1024) -> (160,160,128)
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        # (20, 20, 1024)
        super(SPP, self).__init__()
        # 512
        c_ = c1 // 2
        # (20, 20, 512)
        self.cv1 = Conv(c1, c_, 1, 1)
        # (20, 20, 2048) -> (20, 20, 512)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, phi, pretrained):
        super().__init__()
        # 输入图像 (640,640,3)
        # 初始base_channels是64

        # 利用Focus结构进行特征提取
        # (640,640,3) -> (320,320,12) -> (320,320,64)
        self.stem = Focus(3, base_channels, k=3)


        self.dark2 = nn.Sequential(
            # (320,320,64) -> (160,160,128)
            # 长宽大小 = ((输入长宽大小 - 卷积核大小 + 2 * padingg) / 2 + 1)向下取整
            Conv(base_channels, base_channels * 2, 3, 2),
            # (160,160,128) -> (160,160,128)
            C3(base_channels * 2, base_channels * 2, base_depth),
        )

        self.dark3 = nn.Sequential(
            # (160,160,128) -> (80,80,256)
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            # (80,80,256) -> (80,80,256)
            C3(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        self.dark4 = nn.Sequential(
            # (80,80,256) -> (40,40,512)
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            # (40,40,512) -> (40,40,512)
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )

        self.dark5 = nn.Sequential(
            # (40,40,512) -> (20,20,1024)
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            # (20,20,1024) -> (20,20,1024)
            SPP(base_channels * 16,base_channels * 16),
            # (20,20,1024) -> (20,20,1024)
            C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )

    def forward(self,x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

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
    img = torch.rand([1,3, 640, 640])
    phi = 1
    pretrained = 1
    model = CSPDarknet(64, 1, phi, pretrained)
    out = model(img)
    print(model)
