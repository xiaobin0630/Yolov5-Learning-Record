import torch
import torch.nn

if __name__ == '__main__':
    img = torch.rand([3,640,640])
    x1 = img[...,::2,::2]
    x2 = img[...,1::2,::2]
    x3 = img[...,::2,1::2]
    x4 = img[...,1::2,1::2]
    # print(x1)
    # print(x1.shape)
    x = torch.cat([x1,x2,x3,x4],0)
    # print(x.shape)
    print("1" if False else "2")