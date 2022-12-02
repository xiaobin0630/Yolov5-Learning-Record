import time
import os
import torch
import torch.nn
import numpy as np
import cv2
import glob
from tqdm import tqdm
# from utils.utils import get_classes,get_anchors
from random import sample, shuffle
from PIL import Image
import datetime
# from utils.dataloader import YoloDataset,yolo_dataset_collate
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
def rand(a = 0,b = 1):
    return np.random.rand() * (b - a) + a
def foo(*number):
    for i in number:
        print(i)
    print(type(number))

def foo1(a,*number):
    print("a:",a)
    print("number:",number)
    for i in number:
        print(i)
    print(type(number))

def bar(**number):
    print(number)
def bar1(a,b):
    print(a)
    print(b)
class Data_test(object):
    day = 1
    month = 2
    year = 3
    def __init__(self,year=0,month=0,day=0):
        self.day = day
        self.month = month
        self.year = year
    @classmethod
    def get_time(cls):
        print(cls.day)
        print(cls.month)
        print(cls.year)
def get_classes(path):
    with open(path,encoding='utf-8') as f:
        classes = f.readlines()
        classes_name = [i.split("\n")[0] for i in classes]
    return classes_name,len(classes)
def get_anchors(path):
    with open(path,encoding='utf-8') as f:
        anchors = f.readline()
        # print(anchors,type(anchors))
        anchor = [float(x) for x in anchors.split(",")]
        anchor = np.array(anchor).reshape(-1,2)
        return anchor,len(anchor)
def cvtColor(image):
    # 判断图像是否RBG图像 shape数是否是3 第三维数是否是3
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    # 不是就直接转化为RBG图像
    else:
        image.convert('RBG')
        return image
def preprocess_input(image):
    image /= 255.0
    return image
class YoloDataset(Dataset):
    def __init__(self,annotation_lines,input_shape,num_classes,anchors,anchors_mask,epoch_length,mosaci,mixup,
                 mosaic_prob,mixup_prob,train,special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4
    # 返回数据条数
    def __len__(self):
        return self.length

    # 数据预处理过程
    def __getitem__(self, item):
        # 保证item在0-self.length之间
        item = item % self.length

        # 判断这次是否要进行马赛克数据增强
        if self.mosaic and self.mosaic_prob < self.rand() and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # 随机在提取三张图片
            lines = sample(self.annotation_lines,3)
            # 将本次的item数据也添加进去
            lines.append(self.annotation_lines[item])
            # 打乱列表的顺序
            shuffle(lines)
            # 进行马克思数据增强
            image, box = self.get_random_data_with_Mosaic(lines,self.input_shape)

            # 是否进行mixup数据增强
            if self.mixup and self.rand() < self.mixup_prob:
                # 再随机提取一张图片数据
                lines = sample(self.annotation_lines,1)
                # 数据预处理
                image_2 ,box_2 = self.get_random_data(lines[0],self.input_shape,random= self.train)
                # minup数据增强 其实就是两张图片数据值折半相加
                image,box = self.get_random_data_with_MixUp(image,box,image_2,box_2)
        # 进行数据预处理
        else:
            image,box = self.get_random_data(self.annotation_lines[item],self.input_shape,random=self.train)
        # 将第三维的通道数放0维度来,并对数据进行归一化处理
        image = np.transpose(preprocess_input(np.array(image,dtype=np.float32)),(2,0,1))
        # 将真实框变为array形式数据 数据类型设置为np.float32
        box = np.array(box,dtype=np.float32)
        if len(box) != 0:
            # 对真实框进行归一化 0 ~ 1
            box[:,[0,2]] = box[:,[0,2]] / self.input_shape[1]
            box[:,[1,3]] = box[:,[1,3]] / self.input_shape[0]

            # 将左上角与右下角 转化为 中心与高宽的形式
            # 获取真实框的高宽
            box[:,2:4] = box[:,2:4] - box[:,0:2]
            # 获取真实框的中心点坐标
            box[:,0:2] = box[:,0:2] + box[:,2:4] / 2
        # 为网络建立目标模板
        y_true = self.get_target(box)
        return image,box,y_true

    def get_target(self,targets):
        # 一共有三个特征层数
        # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_layers = len(self.anchors_mask)
        # [640,640]
        input_shape = np.array(self.input_shape,dtype='int32') # [640,640]
        # [[20,20],[40,40],[80,80]]
        grid_shapes = [input_shape // {0:32,1:16,2:8,3:4}[l] for l in range(num_layers)] # [[20,20],[40,40],[80,80]]
        # [[3,20,20,25],[3,40,40,25],[3,80,80,25]]
        y_true = [np.zeros((len(self.anchors_mask[l]),grid_shapes[l][0],grid_shapes[l][1],self.bbox_attrs),dtype='float32') for l in range(num_layers)]
        # [[3,20,20],[3,40,40],[3,80,80]]
        box_bets_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        # 如果没有真实框,直接返回y_true
        if len(targets) == 0:
            return y_true
        # 逐层循环,进行建立网络目标
        for l in range(num_layers):
            # 提取特征层的大小
            in_h, in_w = grid_shapes[l] # 20 20
            # anchors 对应的大小 就是将先验框anchors 对应缩放道特征层上
            anchors = np.array(self.anchors) / {0:32,1:16,2:8,3:4}[l]
            # 创立一个维度和target一样的零矩阵
            batch_target = np.zeros_like(targets)

            # 计算正样本在特征层上的中心点
            batch_target[:,[0,2]] = targets[:,[0,2]] * in_w
            batch_target[:,[1,3]] = targets[:,[1,3]] * in_h
            batch_target[:,4] = targets[:,4]
            # 计算真实框的高宽与先验框的高宽之比
            ratios_of_gt_anchors = np.expand_dims(batch_target[:,2:4],1) / np.expand_dims(anchors,0)
            # 计算先验框的高宽与真实框的高宽之比
            ratios_of_anchors_gt = np.expand_dims(anchors,0) / np.expand_dims(batch_target[:,2:4],1)
            # 将真实框比先验框 与 先验框比真实框 拼接起来
            ratios = np.concatenate([ratios_of_gt_anchors,ratios_of_anchors_gt],axis=-1)
            # 在ratios 最后一维度取最大值
            max_ratios = np.max(ratios,axis = -1)

            for t,ratio in enumerate(max_ratios):
                # 判断是否小于阀值
                over_threshold = ratio < self.threshold
                # 确保至少有一个是True
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    # over_threshold[mask] 是否是真 是假则下一次循环
                    if not over_threshold[mask]:
                        continue

                    # 获取真实框属于那个网格点 中心点向下取整
                    i = int(np.floor(batch_target[t,0]))
                    j = int(np.floor(batch_target[t,1]))

                    offsets = self.get_near_points(batch_target[t,0],batch_target[t,1],i,j)
                    for offset in offsets:
                        #
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        # 判断坐标点是否出界 出界则进行下一次循环
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue
                        if box_bets_ratio[l][k,local_i,local_j] != 0:
                            if box_bets_ratio[l][k,local_j,local_i] > ratio[mask]:
                                y_true[l][k,local_j,local_i,:] = 0
                            else:
                                continue

                        # 取出真实框的种类
                        c = int(batch_target[t, 4])

                        # tx ty 代表中心调整参数的真实值
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        # 获取当前先验框最好的比列
                        box_bets_ratio[l][k,local_j,local_i] = ratio[mask]
        return y_true




    # 得到附近的点
    def get_near_points(self, x, y, i, j):
        # 计算x y 与向下取整的ix iy的差值
        sub_x = x - i
        sub_y = y - j
        # 差值都大于0.5 则取 0, 0  1, 0  0, 1 就是右 上 加自己的原点
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        # sub_x 小于0.5 sub_y 大于0.5  则取 左 上 加自己的原点
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0],[0, 1]]
        # sub_x < 0.5 and sub_y < 0.5 则取左 下 加自己的原点
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else: # 取下 右 加自己的原点
            return [[0, 0], [1, 0], [0, -1]]








    def get_random_data_with_MixUp(self,image_1,box_1,image_2,box_2):
        # 对应图像数值折半相加
        new_image = np.array(image_1,np.float32) * 0.5 + np.array(image_2,np.float32) * 0.5
        # 判断是否有真实框
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            # 按行堆叠
            new_boxes = np.concatenate([box_1,box_2],axis=0)
        return new_image,new_boxes
    def get_random_data(self,annotation_line,input_shape,jitter=0.3,hue=0.1,sat=0.7,val=0.4,random=True):
        # 字符串按空格切割
        line = annotation_line.split()
        # 读取图片 并将其转化为RBG
        image = Image.open(line[0])
        image = cvtColor(image)

        # 获取图像的高宽与目标的高宽
        iw, ih = image.size
        h, w = input_shape

        # 获的预测框
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # 如果不是训练模式
        if not random:
            # 取最小值的比列,进行高宽的缩放
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            # 计算出来黏贴坐标点
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 对图像进行resize
            image = image.resize((nw,nh),Image.BICUBIC)
            # 创建一张目标大小的图像
            new_image = Image.new('RGB',(w, h), (128,128,128))
            # 将image粘贴到new_image的dx dy位置上
            new_image.paste(image,(dx,dy))
            # 将new_image 变为array
            image_data = np.array(new_image,np.float32)

            # 对真实框进行调整
            if len(box) > 0:
                # 打乱
                np.random.shuffle(box)
                # 映射到粘贴图上的真实框的坐标点
                box[:,[0,2]] = box[:,[0,2]] * nw / iw + dx
                box[:,[1,3]] = box[:,[1,3]] * nh / ih + dy
                # 将左上角的坐标点值出界的全部设置为0
                box[:,0:2][box[:,0:2]<0] = 0
                # 将右下角的坐标点值出界的全部设置为目标的高宽
                box[:,2][box[:,2]>w] = w
                box[:,3][box[:,3]>h] = h
                # 计算框的高宽
                box_w = box[:,2] - box[:,0]
                box_h = box[:,3] - box[:,1]
                # 保留真实框高宽大于1的框
                box = box[np.logical_and(box_w>1,box_h>1)]
            return image_data, box
        # 训练模式下
        # 对图像进行缩放且长宽的扭曲
        new_ar = iw / ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25,2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        # resizeimage图像到大小(nw,nh)
        image = image.resize((nw,nh),Image.BICUBIC)
        # 计算粘贴点 这里是在nw 与w的差值中随机取个数为dx的值
        dx = int(self.rand(0,w-nw))
        # 同上
        dy = int(self.rand(0,h-nh))
        # 创建一张目标大小的图像
        new_image = Image.new('RGB',(w,h),(128,128,128))
        # 将image 粘贴到new_image dx,dy位置上
        new_image.paste(image,(dx,dy))
        # 赋值
        image = new_image

        # 翻转图像
        flip = self.rand() < 0.5
        # 判断是否进行翻转
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 将图像图像转化array
        image_data = np.array(image,np.uint8)

        # 在均匀分布中随机取3个值
        r = np.random.uniform(-1, 1, 3) * [hue,sat,val] + 1

        # 将RGB图像转化到HSV图像然后进行通道分割
        hue,sat,val = cv2.split(cv2.cvtColor(image_data,cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # 变换
        x = np.arange(0,256,dtype=r.dtype)
        # 生成最后的rgb的通道
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1],0,255).astype(dtype)
        lut_val = np.clip(x * r[2],0,255).astype(dtype)

        # 融合
        image_data = cv2.merge((cv2.LUT(hue,lut_hue),cv2.LUT(sat,lut_sat),cv2.LUT(val,lut_val)))
        image_data = cv2.cvtColor(image_data,cv2.COLOR_HSV2RGB)

        # 进行真实框的调整
        if len(box) > 0:
            # 打乱框的顺序
            np.random.shuffle(box)
            # 映射
            box[:,[0,2]] = box[:,[0,2]] * nw/iw + dx
            box[:,[1,3]] = box[:,[1,3]] * nw/ih + dy
            # 判断是否图像翻转
            if flip:
                # 水平调整真实框的x
                box[:,[0,2]] = w - box[:,[2,0]]
            # 左上角的坐标值是否出界 如果出界,则设置为0
            box[:,0:2][box[:,0:2]<0] = 0
            # 右下角的坐标值是否出界,如果出界,用目标的高宽设置
            box[:,2][box[:,2]>w] = w
            box[:,3][box[:,3]>h] = h
            # 计算真实框的高宽
            box_w = box[:,2] - box[:,0]
            box_h = box[:,3] - box[:,1]
            # 将真实框小于1的框抛弃
            box = box[np.logical_and(box_w>1,box_h>1)]

        return image_data,box











    # 马赛克数据增强
    def get_random_data_with_Mosaic(self,annotation_line,input_shape,jitter=0.3,hue=0.1,sat=0.7,val=0.4):
        # 得到传入的高宽
        h, w = input_shape
        # 生成最小补偿x与y的 0.3 ~ 0.7随机数
        min_offset_x = self.rand(0.3,0.7)
        min_offset_y = self.rand(0.3,0.7)
        # 保存图像数据的列表
        image_datas = []
        # 保存坐标框的列表
        box_datas = []
        # 标记正在处理第几张图像
        index = 0
        # 循环单独对每张图像进行处理
        for line in annotation_line:
            # 对line进行分割
            line_content = line.split()

            # 读取图像
            image = Image.open(line_content[0])

            # 将图像换化为RBG图像
            image = cvtColor(image)

            # 获取本轮循环的图像高宽
            iw, ih = image.size

            # 保存框的位置 [[x1, y1, x2, y2, class]...]
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])

            # 生成随机数判断是否进行图像的左右翻转
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                # 图像左右翻转
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 进行图像的左右翻转后,真实框的y轴不变化,x要做处理
                '''
                0,0 -----------> x (width)
                 |
                 |     
                 | (x11,y11)                    (x1,y1)
                 |   ---------                   ---------
                 |  |         |                 |         | 
                 |  |         |                 |         |
                 |   ---------                   ---------
                 |         (x22,y22)                       (x2,y2) 
                 y
              (height)
                           x11 = w - x2
                           x22 = w - x1
                '''
                box[:,[0,2]] = iw - box[:,[2,0]]
            # 对图像进行缩放且进行长宽的扭曲
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(0.4,1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            # 对图像进行resize(nw,nh)的大小
            image = image.resize((nw,nh),Image.BICUBIC)
            # 将图像进行放置,分别对应四张分割图片的位置

            # 这里操作是为已经预处理的图像进行计算粘贴坐标点的操作
            # 具体是这样:
            # 在高宽为h,w的新图像中,0.3 ~ 0.7 的随机数各自乘以w与h,计算出来一个点
            # 将这个点对应不同的第几张图像有不同的减法操作,计算出对应的粘贴点
            if index == 0:
                dx = int(w*min_offset_x) - nw # 64
                dy = int(h*min_offset_y) - nh # 201
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh


            # 创建一张新图片
            new_image = Image.new('RGB',(w,h),(128,128,128))
            # 将image黏贴到new_image的(dx,dy)上
            new_image.paste(image,(dx,dy))
            # 将图像数据变为array
            image_data = np.array(new_image)
            # index加1
            index = index + 1
            # 零时存放列表
            box_data = []
            # 图像做了变化,真实框也要进行相关的变化
            # 是否有真实框,有就做处理
            if len(box)>0:
                np.random.shuffle(box)
                # 做与图像相同高宽缩放然后在加上对应的dx dy
                box[:,[0,2]] = box[:,[0,2]] * nw/iw + dx
                box[:,[1,3]] = box[:,[1,3]] * nh/ih + dy
                # 对出界的坐标点继续规范处理
                # 左上角坐标小于0则全部设置为0
                box[:,0:2][box[:,0:2] < 0] = 0
                # 右下角坐标大于图像的new_image图像的宽高,则对应设置为w,h
                box[:,2][box[:,2]>w] = w
                box[:,3][box[:,3]>h] = h
                # 求框的宽高
                box_w = box[:,2] - box[:,0]
                box_h = box[:,3] - box[:,1]
                # 保留真实框的宽高大于于1框
                box = box[np.logical_and(box_w>1,box_h>1)]
                # 创建一个与box同样大小的zeros矩阵 用于保存处理好的真实框
                box_data = np.zeros((len(box),5))
                # 存入真实框
                box_data[:,len(box)] = box
            # 将每次循环处理好的图像与真实框保存在image_datas与box_datas 对应列表中
            image_datas.append(image_data)
            box_datas.append(box_data)
        # 之所以要做这样的操作是因为,之前那个循环操作最后处理好的图像 都是单独保存的
        # 将图像分割后,放在一起
        # 这里计算的那个粘贴点之前的原点
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        # 创建一个shape为(h,w,3)的新零矩阵
        new_image = np.zeros([h, w, 3])
        # 高在前 宽在后
        # 第一张图片 image_datas[0][:cuty,:cutx,:] 赋值到new_image[:cuty,:cutx,:]
        new_image[:cuty,:cutx,:] = image_datas[0][:cuty,:cutx,:]
        # 第二张图片
        new_image[cuty:,:cutx,:] = image_datas[1][cuty:,:cutx,:]
        # 第三张图片
        new_image[cuty:,cutx:,:] = image_datas[2][cuty:,cutx:,:]
        # 第四张图片
        new_image[:cuty,cutx:,:] = image_datas[3][:cuty,cutx:,:]

        # 设置数据类型
        new_image = np.array(new_image,np.uint8)

        # 在[-1,1)均匀分布中采样3次 之后在进行对应位置相乘再加1 [hue,sat,val] 0.1 0.7 0.4
        r = np.random.uniform(-1,1,3) * [hue,sat,val] + 1

        # 将new_image 转为HSV 后进行通道分割
        hue,sat,val = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2HSV))
        # 暂存数据类型
        dtype = new_image.dtype
        # 生成0到255的array数据,且类型为uint8
        x = np.array(0,256,dtype=r.dtype)
        # 生成对应通过的 hue sat val
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 融合
        new_image = cv2.merge((cv2.LUT(hue,lut_hue),cv2.LUT(sat,lut_sat),cv2.LUT(val,lut_val)))
        # 将hsv图像转化为RGB图像
        new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)

        # 最后再对真实框进行处理
        new_boxes = self.merge_bboxes(box_datas,cutx,cuty)

        return new_image,new_boxes
    def merge_bboxes(self,bboxes,cutx,cuty):
        # 最后保存真实框
        merge_bbox = []
        # 按图片的个数进行循环
        for i in range(len(bboxes)):
            # 提取第i张图像对应真实框
            for box in bboxes[i]:
                # 暂时存储真实框变量
                temp_box = []
                # 提取出坐标点值
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                # 第一张图像对应的真实框的处理
                if i == 0:
                    # 真实框左上角的坐标点已经大于了图像的右下角的坐标点就不要了这个框了
                    if y1 > cuty or x1 > cutx:
                        continue
                    # 有一部分在对应图像的内部,要进行相应的处理
                    # 真实框右下角的y的值大于了图像右下角的y的值 且真实框左上角的y的值小于图像右下角的y的值,则将y2设置为
                    # 图像右下角的y的值
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    # 理由同上
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                # 第二张图像对应的真实框处理
                if i == 1:
                    # 当真实框右下角坐标点y小于了cuty 或者左上角x的坐标x 大于了cutx 则说明真实完全出界,得抛弃这个坐标点
                    if y2 < cuty or x1 > cutx:
                        continue
                    # 有一部分在对应图像的内部,要进行相应的处理
                    # 真实框左下角上角坐标y在图像内部,右上角坐标y出界,用图像的右上角坐标cuty赋值
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    # 真实框右下角x坐标值出界,右上角的x坐标在图像内部,得用图像的cutx给x2赋值
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                # 第三图片对应的真实框处理
                if i == 2:
                    # 图像的范围(cuty:,cutx:)
                    # 如果真实框标签右下角x,y值小于cutx,cuty 则不要这个框
                    if y2 < cuty or x2 < cutx:
                        continue
                    # 真实框右下角的y大于cuty 而左上角的y小于cuty 则用cuty赋值给y1
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    # 真实框右下角的x大于cutx 而左上角的x 小于cutx 则用cutx 赋值给x1
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    # 图像的范围(:cuty,cutx:)
                    # 如果真实框的左上角的y大于cuty 或者 真实框的右下角的x小于cutx 则不要这个框
                    if y1 > cuty or x2 < cutx:
                        continue
                    # 真实框的右下角的y大于cuty 且真实框的左上角的y小于cuty 则用cuty赋值给真实框的右下角的y
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    # 真实框的右下角的x大于cutx 且 真实框的左上角的x 小于cutx 则用cutx 赋值给真实框的左上角的x
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                # 暂保存处理好的真实框坐标
                temp_box.append(x1)
                temp_box.append(y1)
                temp_box.append(x2)
                temp_box.append(y2)
                # 保存类类
                temp_box.append(box[-1])
                # 保存
                merge_bbox.append(temp_box)
        return merge_bbox



























    # 生成[a,b)的随机数
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a) + a

if __name__ == '__main__':
    classes_path = "model_data/voc_classes"
    classes_name,classes_num = get_classes(classes_path)
    # print(classes_name,classes_num)
    input_shape = [640,640]
    anchors_path = "model_data/yolo_anchors"
    anchors,anchors_num = get_anchors(anchors_path)
    # print(anchors,anchors)
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    epoch_length = 20
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    r = np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.4] + 1

    print(r,type(r))
    # lines = [1,2,3,4,5,6]
    # l = sample(lines,3)
    # print(l,type(l))
    # l.append(8)
    # print(l,type(l))
    # shuffle(l)
    # print(l,type(l))


    # b = 10
    # a = 0
    # s = np.random.rand() * (b - a) + a
    # print(s)
    # a1 = [3, 5, 2, 1, 6]
    # a2 = [7, 1, 2, 4, 2]
    # a = np.maximum(a1, 1)
    # print(a)

    # ground_truth_files_list = glob.glob('model_data' + '/*.txt')
    # id = ground_truth_files_list[0].split('.txt',1)[0]
    # print(os.path.normpath(id))
    # os.path.basename(os.path.normpath(id))
    # print(os.path.basename(os.path.normpath(id)))
    # Data_test.get_time()

    # bar(**{'a':1,'b':2})
    # foo1(1,2,3,4)
    # classes_path = "model_data/voc_classes"
    # anchors_path = 'model_data/yolo_anchors'
    # train_annotation_path = '2007_train.txt'
    # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # input_shape = [640,640]
    # UnFreeze_Epoch = 20
    # mosaic = True
    # mosaic_prob = 0.5
    # mixup = True
    # mixup_prob = 0.5
    # special_aug_ratio = 0.7
    # class_names, num_classes = get_classes(classes_path)
    # anchors, num_anchors = get_anchors(anchors_path)
    # with open(train_annotation_path,encoding='utf-8') as f:
    #     # 按行读取
    #     train_lines = f.readlines()
    # batch_size = 2
    # num_workers = 0
    # train_sampler = None
    # shuffle = True
    # train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask,
    #                             epoch_length=UnFreeze_Epoch,
    #                             mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True,
    #                             special_aug_ratio=special_aug_ratio)
    # gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                  drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
    #
    # for i,batch in enumerate(gen):
    #     print(batch[0],batch[1],batch[2])

    # anchors = [10,13, 16,30, 33,23,  30,61, 62,45, 59,119,  116,90, 156,198, 373,326]
    # new_anchors = np.array(anchors) / {0: 32, 1: 16, 2: 8, 3: 4}[0]

    # print(new_anchors)
    # x = np.array([1, 4, 3, -1, 6, 9])
    # print(x.argsort())
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






















