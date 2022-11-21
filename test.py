import torch
import torch.nn
import numpy as np
from utils.utils import get_classes
from random import sample, shuffle
from PIL import Image
def rand(a = 0,b = 1):
    return np.random.rand() * (b - a) + a

if __name__ == '__main__':
    a = [(1,2),(3,4),(5,6)]
    b = np.array(a)
    print(torch.Tensor(b))
    anchor_w = torch.Tensor(b).index_select(1, torch.LongTensor([1]))
    print(anchor_w)
    print(anchor_w.repeat(8, 1))
    print(anchor_w.repeat(8, 1).shape)
    print(anchor_w.repeat(8,1).repeat(1,1,400))
    print(anchor_w.repeat(8,1).repeat(1,1,400).shape)

    # a = [(1,2),(3,4)]
    # b = np.array(a)
    # print(b)
    # x = torch.linspace(0, 19, 20).repeat(20,1).repeat(3,1,1).view()
    # print(x)
    # print(x.shape)
    # a = np.random.uniform(-1, 1, 3)
    # print(a)
    # print(a * [0.5,1.5,1])
    # print(a * [0.5, 1.5, 1] + 1)
    # path = '1.jpg'
    # img = Image.open(path)
    # print(img)
    # img.show()
    # out = img.transpose(Image.FLIP_LEFT_RIGHT)
    # out.show()
    # print(rand(0.3,b = 0.7))
    # num = range(100)
    # lines = sample(num, 3)
    # print(lines)
    # shuffle(lines)
    # print(lines)
    # img = torch.rand([3,640,640])
    # x1 = img[...,::2,::2]
    # x2 = img[...,1::2,::2]
    # x3 = img[...,::2,1::2]
    # x4 = img[...,1::2,1::2]
    # print(x1)
    # print(x1.shape)
    # x = torch.cat([x1,x2,x3,x4],0)
    # print(x.shape)
    # print("1" if False else "2")
    # voc_class_path = "model_data/voc_classes"
    # classes_name,class_num = get_classes(voc_class_path)
    # print(classes_name,class_num)
    # print(.5)






















