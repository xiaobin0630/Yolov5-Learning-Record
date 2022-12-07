import torch
import torch.nn as nn
# 创建类静态方法 可以通过类名调用方法
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        # f(x) = x * 1/ (1 + e-x)
        return x * torch.sigmoid(x)

# 自动算pudding大小的函数
def autopad(k,p=None):
    # 如果p是None 进行计算pudding大小, 否则直接返回p
    if p is None:
        # 判断k是否是整数,是整数,则进行整除2,然后返回,如果k是是列表,则就进行遍历进行整除2形成列表返回
        p = k // 2 if isinstance(k,int) else [x // 2 for x in k]
    return p
class Conv(nn.Module):
    # 初始化 输入通道数 输出通常数 卷积核大小 pudding group 是否使用激活函数
    def __init__(self, c1, c2, k=1, s=1,p=None,g=1,act=True):
        super(Conv, self).__init__()
        # 初始化卷积操作
        self.conv = nn.Conv2d(c1, c2, k, s,autopad(k, p),groups=g,bias=False)
        # 初始化bn操作 eps防止分母为0 momentum 使计算更稳定
        self.bn = nn.BatchNorm2d(c2, eps=0.0001,momentum=0.03)
        # 初始化激活函数 act 如果是True 这进行 f(x) = x * 1/ (1 + e-x)激活函数
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # 设置前向传播路径
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class Focus(nn.Module):
    # 初始化 输入通道数 输出通道数 卷积核大小 步长 pudding group 是否使用激活函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        # (320,320,12) - (320,320,64)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        print(x[...,::2,::2].shape,x[...,1::2,::2].shape,x[...,::2,1::2].shape,x[...,1::2,1::2].shape)
        print(torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1).shape)

        return self.conv(
            torch.cat(
                [
                    x[...,::2,::2],
                    x[...,1::2,::2],
                    x[...,::2,1::2],
                    x[...,1::2,1::2]
                ], 1
            )
        )

class Bottleneck(nn.Module):
    # 初始化 输入通道数,输出通道数,循环迭代的次数,shortcut是否继续捷径分支,g分组卷积 e为通道数的扩展倍数
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_,c2,3,1,g=g)
        self.add = shortcut and c1 == c2
    def forward(self,x):
        # x                         (bs,64,160,160)
        # self.cv1(x)               (bs,64,160,160) -> (bs,64,160,160)
        # self.cv2(self.cv1(x))     (bs,64,160,160) -> (bs,64,160,160)
        # x + self.cv2(self.cv1(x)) (bs,64,160,160) + (bs,64,160,160) -> (bs,64,160,160)
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # c1 = 256,c2 = 256 n=9
    # c_ = 256 * 1 = 256
    # C3结构图
    #                                   input ((bs, 256, 80, 80))
    #                                     |
    #                       -------------------------------
    #     (bs, 128, 80, 80) cv1                          cv2 (bs, 128, 80, 80)
    #                        |                            |
    #                       m(bottleneck,n)               |
    #                        |                            |
    #                       --------------cat--------------- (bs, 256, 80, 80)
    #                                      |
    #                                      |
    #                                     cv3 (bs, 256, 80, 80)
    #                                      |
    #                                    output (bs, 256, 80, 80)


    # bottleneck()   n次                  input(((bs, 128, 80, 80)))
    #                                       |
    #                       -------------------------------------
    #                       |                                  cv1  (bs, 128, 80, 80)
    #                       |                                   |                        # 这个结构在每个C3里循环迭代n次 输出数据维度不变
    #                       |                                  cv2  (bs, 128, 80, 80)
    #                       ------------------+------------------
    #                                  (bs, 128, 80, 80)
    #

    # CSPbottleneck
    # 输入通道数,输出通道数,循环迭代的次数,shortcut是否继续捷径分支,g分组卷积 e为通道数的扩展倍数
    def __init__(self, c1, c2, n=1, shortcut=True, g=1,e=0.5):
        super(C3, self).__init__()
        # 在残差里的隐藏层里的通道数
        c_ = int(c2 * e)
        # (bs,128,160,160) -> (bs,64,160,160)
        self.cv1 = Conv(c1,c_,1,1)
        # (bs,128,160,160) -> (bs,64,160,160)
        self.cv2 = Conv(c1,c_,1,1)
        # (bs,128,160,160) -> (bs,128,160,160)
        self.cv3 = Conv(2 * c_,c2,1)
        self.m = nn.Sequential(*[Bottleneck(c_,c_,shortcut,g,e=1.0) for _ in range(n)])
    def forward(self,x):
        # self.cv1(x)           (bs,128,160,160) -> (bs,64,160,160)
        # self.m(self.cv1(x))   (bs,64,160,160)  -> (bs,64,160,160) -> (bs,64,160,160) -> (bs,64,160,160)
        # self.cv2(x)           (bs,128,160,160) -> (bs,64,160,160)
        # self.cv3(x)           (bs,64,160,160) cat (bs,64,160,160) -> (bs,128,160,160)
        return self.cv3(torch.cat((self.m(self.cv1(x)),self.cv2(x)),dim=1))
class SSP(nn.Module):
    # SSP结构图
    #                                           input (bs, 1024, 20, 20)
    #                                                        |
    #                                                       cv1 (bs, 512, 20, 20)
    #                                                        |
    #                  -------------------------------------------------------------------------------
    #                  |                         |                          |                        |
    #             不变动x(bs, 512, 20, 20)      Pool k=5 p=2               Pool k=9 p=4            Pool k=13 p=6
    #                  |                         |                          |                        |
    #                 out1(bs, 512, 20, 20)  out2(bs, 512, 20, 20)    out3(bs, 512, 20, 20)   out4(bs, 512, 20, 20)
    #                  -------------------------------------------------------------------------------
    #                                                           |
    #                                                          cat (bs, 2048, 20, 20)
    #                                                           |
    #                                                          cv2 (bs, 1024, 20, 20)
    #
    # 输入c1 = 1024 c2 = 1024   input(bs, 1024, 20, 20)
    def __init__(self, c1, c2,k=(5, 9, 13)):
        super(SSP, self).__init__()
        # 隐藏层通道数
        c_ = c1 // 2 # 1024 -> 512
        self.cv1 = Conv(c1, c_, 1, 1) # (bs, 1024, 20, 20) -> (bs, 512, 20, 20)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1,padding= x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x)for m in self.m],1))

class CSPDarknet(nn.Module):
    # 初始化 假定基础通道数是64  base_channels = 64  基础深度数是3 base_depth = 3
    def __init__(self,base_channels,base_depth):
        super().__init__()
        # 输入图片是 (bs, 3, 640, 640)
        # base_channels是64
        # 线进行focus网络操作进行特征提取 输入通道数为3 输出通道数为64 卷积核大小为3
        # (bs,3,640,640) -> (bs, 12, 320, 320) -> (bs, 64, 320, 320)
        self.stem = Focus(3,base_channels,k=3)


        # Resblock_body
        self.dark2 = nn.Sequential(
            # 卷积 bn 激活函数
            # W' = [(W - F + 2P) / 2 + 1]向下取整 得卷积后的长宽
            # (bs, 64, 320, 320) -> (bs, 128, 160, 160)
            Conv(base_channels,base_channels * 2, 3, 2),
            # (bs, 128, 160, 160) -> (bs, 128, 160, 160)
            C3(base_channels * 2,base_channels * 2,base_depth)
        )

        self.dark3 = nn.Sequential(
            # W' = [(W - F + 2P) / 2 + 1]向下取整 得卷积后的长宽
            # (160 - 3 + 2) / 2 +1
            # (bs, 128, 160, 160) -> (bs, 256, 80, 80)
            Conv(base_channels * 2,base_channels * 4, 3, 2),
            # (bs, 256, 80, 80) -> (bs, 256, 80, 80)
            # 9次 bottleneck
            C3(base_channels * 4, base_channels * 4,base_depth * 3)
        )

        self.dark4 = nn.Sequential(
            # W' = [(W - F + 2P) / 2 + 1] 向下取整
            # (bs, 256, 80, 80) -> (bs, 512, 40, 40)
            Conv(base_channels * 4,base_channels * 8, 3, 2),
            # (bs, 512, 40, 40) -> (bs, 512, 40, 40)
            C3(base_channels * 8,base_channels * 8,base_depth * 3)
        )
        self.dark5 = nn.Sequential(
            # W' = [(W - F + 2P) / 2 + 1] 向下取整
            # (bs, 512, 40 , 40) -> (bs, 1024, 20, 20)
            Conv(base_channels * 8,base_channels * 16, 3, 2),
            # SPP (bs, 1024, 20, 20) - > (bs, 1024, 20, 20)
            SSP(base_channels * 16,base_channels * 16),
            # C3 (bs, 1024, 20, 20) -> (bs, 1024, 20, 20) bottleneck 循环3次,且不走捷径分支
            C3(base_channels * 16, base_channels * 16,base_depth,shortcut=False)
        )



    def forward(self, x):
        # (bs, 3, 640, 640) -> (bs, 64, 320, 320)
        x = self.stem(x)
        # (bs, 64, 320, 320) -> (bs, 128, 160, 160)
        x = self.dark2(x)
        # (bs, 128, 160, 160) -> (bs, 256, 80, 80)
        x = self.dark3(x)
        feat1 = x
        # (bs, 256, 80, 80) -> (bs, 512, 40, 40)
        x = self.dark4(x)
        feat2 = x
        # (bs, 512, 40, 40) -> (bs, 1024, 20, 20)
        x = self.dark5(x)
        feat3 = x
        return feat1,feat2,feat3
if __name__ == "__main__":
    image = torch.rand([1,3,640,640])
    model = CSPDarknet(64,3)
    predict = model(image)
    shape = []
    for i in predict:
        shape.append(i.shape)
    print(shape)


    print("123")