import time
import os
import torch
import torch.nn
import numpy as np
from tqdm import tqdm
from utils.utils import get_classes
from random import sample, shuffle
from PIL import Image
import datetime
def rand(a = 0,b = 1):
    return np.random.rand() * (b - a) + a

if __name__ == '__main__':
    x = np.array([1, 4, 3, -1, 6, 9])
    print(x.argsort())
    # x = np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
    # print(x[...,::-1])
    # box_xy[..., ::-1]
    # x = torch.rand([1,1200,85])
    # output = [None for _ in range(len(x))]
    # print(output)
    # for i, j in enumerate(x):
    #
    #     class_conf, class_pred = torch.max(j[:, 5:5 + 80], 1, keepdim=True)
    #     conf_mask = (j[:, 4] * class_conf[:, 0] >= 0.5).squeeze()
    #     m = 0
    #     for k in conf_mask:
    #         if k == True:
    #             m = m + 1
    #     print(m)
    #     print(conf_mask)
    #     print(conf_mask.shape)
        # print(j.shape)
        # print(class_conf.shape)
        # print(class_conf)
        # print(class_pred.shape)
        # print(class_pred)
    # print(x)
    # print(x.shape)
    # for i,j in enumerate(x):
    #     print(j.shape)
    # output = [None for _ in range(len(x))]
    # print(output)
    # pass
    # x = torch.rand([1,3,20,20])
    # print(torch.linspace(0,19,20).repeat(20,1))
    # print(torch.linspace(0,19,20).repeat(20,1).t())
    # y_trues = [[] for _ in [1,2,3,4]]
    # print(y_trues)
    # time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    # log_dir = os.path.join('logs', "loss_" + str(time_str))
    # print(time_str)
    # print(log_dir)
    # ngpus_per_node = torch.cuda.device_count()
    # print(ngpus_per_node)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    # text = ""
    # for char in tqdm(["a","b","c","d"]):
    #     time.sleep(0.25)
    #     text = text + char

    # a = [(1,2),(3,4),(5,6)]
    # b = np.array(a)
    # print(torch.Tensor(b))
    # anchor_w = torch.Tensor(b).index_select(1, torch.LongTensor([1]))
    # print(anchor_w.repeat(1,1).repeat(1,1,400).view(1,3,20,20))
    # print(anchor_w.repeat(8, 1))
    # print(anchor_w.repeat(8, 1).shape)
    # print(anchor_w.repeat(8,1).repeat(1,1,400))
    # print(anchor_w.repeat(8,1).repeat(1,1,400).shape)

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






















