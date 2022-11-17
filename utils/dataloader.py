from random import sample,shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from utils.utils import cvtColor
from torch.utils.data.dataset import Dataset
class YoloDataset(Dataset):
    # 初试化
    def __init__(self,annotation_lines,input_shape,num_classes,anchors,anchors_mask,epoch_length,mosaic,mixup,mosaic_prob,\
                 mixup_prob,train,special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        # 数据
        self.annotation_lines = annotation_lines
        # 输入图片的大小[640,640]
        self.input_shape = input_shape
        # 类别数
        self.num_classes = num_classes
        # anchors
        self.anchors = anchors
        #
        self.anchors_mask = anchors_mask
        # 总的训练epoch数
        self.epoch_length = epoch_length
        # 是否使用马赛克数据增强
        self.mosaic = mosaic
        # 使用马赛克数据增强的机率
        self.mosaic_pro = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        # 判断是训练还是val
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        # 数据数量
        self.length = len(self.annotation_lines)
        self.epoch_now = -1
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4

    # 返回数据数
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # 当self.mosaic为True 且 self.rand()数 < self.mosaic_pro 再且self.epoch_now < self.epoch_length * self.special_aug_ratio时
        # 进行mosaic数据增强
        if self.mosaic and self.rand() < self.mosaic_pro and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # 随机提取3张图片
            lines = sample(self.annotation_lines,3)
            # 再将index对应的索引添加道lines列表里，构成4张图片
            lines.append(self.annotation_lines[index])
            # 打乱图片顺序
            shuffle(lines)
            # 进行mosaic数据增强
            image,box = self.get_random_data_with_Mosaic(lines, self.input_shape)

    # 获取随机数函数
    def rand(self,a = 0,b = 1):
        return np.random.rand()

    def get_random_data_with_Mosaic(self,annotation_line,input_shape,jitter=0.3,hue=.1,sat=0.7,val=0.4):
        # 获取长宽
        h,w = input_shape

        # 获取0.3-0.7的随机数
        min_offset_x = self.rand(0.3,0.7)
        min_offset_y = self.rand(0.3,0.7)
        # 存放图片
        image_datas = []
        # 存放对应图片的框
        box_datas = []
        index = 0

        for line in annotation_line:
            # 对空格切割字符串
            line_content = line.split()
            # 打开图片 line_content[0]是绝对图片路径
            image = Image.open(line_content[0])
            # 检查是否是RGB图片，不是，则转化为RGB图片
            image = cvtColor(image)

            # 获取图片的大小
            iw,ih = image.size
            # 保存框的位置[[48,240,195,371,11],[8,12,352,498,14]]
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])

            # 翻转图片
            flip = self.rand() < 0.5
            # 当数据数小于0.5且有框进行图片左右翻转
            if flip and len(box) > 0:
                # 左右翻转
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 对应图片的标签进行调整
                box[:,[0,2]] = iw - box[:,[2,0]]

                # 对图像进行缩放并且进行长和宽的扭曲
                new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
                scale = self.rand(0.4,1)
                # 当new_ar小于1就在高上进行,当大于1就在宽上进行
                if new_ar<1:
                    nh = int(scale * h)
                    nw = int(nh * new_ar)
                else:
                    nw = int(scale * w)
                    nh = int(nw / new_ar)
                image = image.resize((nw,nh,Image.BICUBIC))

                # 将图片进行放置,分别对应四张分割图片的位置
                # 第一张图片时 计算后面的缩放大小
                if index == 0:
                    dx = int(w*min_offset_x) - nw
                    dy = int(h*min_offset_y) - nh
                # 第二张图片时
                elif index == 1:
                    dx = int(w*min_offset_x) - nw
                    dy = int(h*min_offset_y)
                # 第三张图片时
                elif index == 2:
                    dx = int(w*min_offset_x)
                    dy = int(h*min_offset_y)
                # 第四张图片时
                elif index == 3:
                    dx = int(w*min_offset_x)
                    dy = int(h*min_offset_y) - nh
                # 创建一张宽长为(w,h)的新图
                new_image = Image.new('RGB',(w,h),(128,128,128))
                # 将image的大小为(dx,dy)的图片放在new_image图像上
                new_image.paste(image,(dx,dy))
                image_data = np.array(new_image)

                # index加1
                index = index + 1
                box_data = []

                # 对框重新进行处理
                if len(box) > 0:
                    # 打乱顺序
                    np.random.shuffle(box)
                    # 对标签进行相同的缩放与扭曲
                    box[:,[0,2]] = box[:,[0,2]] * nw / iw + dx
                    box[:,[1,3]] = box[:,[1,3]] * nh / ih + dy
                    # 将左上角的坐标(x,y)小于0的都设置为0,防止左上角坐标出界
                    box[:,0:2][box[:,0:2] < 0] = 0
                    # 将右下角的坐标长宽大于w,h的全部设置为w,h防止右下角坐标出界
                    box[:,2][box[:,2] > w] = w
                    box[:,3][box[:,3] > h] = h
                    # 获取框的宽与高
                    box_w = box[:,2] - box[:,0]
                    box_h = box[:,3] - box[:,1]
                    # 获取box_w与box_h都大于1像素点的框
                    box = box[np.logical_and(box_w > 1,box_h > 1)]
                    # 创建一个len(box)行与5列的0矩阵
                    box_data = np.zeros((len(box),5))
                    # 将box复制到零矩阵上
                    box_data[:len(box)] = box
                #将image_data,box_data存入image_datas,box_datas
                image_datas.append(image_data)
                box_datas.append(box_data)
            # 将图片分割，放在一起
            # 求出w,h对应倍数的cutx,cuty
            cutx = int(w * min_offset_x)
            cuty = int(h * min_offset_y)

            # 创建一张shape为(h,w,3)的0矩阵
            new_image = np.zeros([h,w,3])
            # 第一张图片切割 (0:cuty,0:cutx,:) 长宽各切到cuty与cutx
            new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
            # 第二张图片切割 (cuty:,:cutx,:) 长从cuty切到: 宽从0到cutx
            new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
            # 第三种图片切割 (cuty:,cutx:,:)
            new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
            # 第四张图片切割 (:cuty,cutx:,:)
            new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

            # 变化数据类型为uint8
            new_image = np.array(new_image, np.uint8)

            # 对图像进行色域变化 通过np.random.uniform(low,high,size)在[low,high)中随机取三次数形成一个列表 在对应乘以[hue,sat,val] 最后在对应值上加1
            r = np.random.uniform(-1,1,3) * [hue,sat,val] + 1

            # 将图像转到HSV上 通过cv2split拆分通道
            hue,sat,val = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2HSV))
            dtype = new_image.dtype

            # 变换
            # 创建0-256的array
            x = np.arange(0,256,dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            # np.clip(a,min,max)所有小于min 都被设置为min，所有大于max都被设置为max
            lut_sat = np.clip(x * r[1],0,255).astype(dtype)
            lut_val = np.clip(x * r[2],0,255).astype(dtype)

            # 通过cv2.LUT来查找table将hue转到对应的lut_hue  cv2.merge()用来合并通道
            new_image = cv2.merge((cv2.LUT(hue,lut_hue),cv2.LUT(sat,lut_sat),cv2.LUT(val,lut_val)))
            # 将HSV图像转为RGB图像
            new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2BGR)

            # 对框进行进一步处理
            new_boxes = self.merge_bboxes(box_datas,cutx,cuty)
            return new_image,new_boxes

    def merge_bboxes(self,bboxes,cutx,cuty):
        merge_bbox = []
        # 4张图片 进行循环处理
        for i in range(len(bboxes)):
            # 循环第一张图片对应的框
            for box in bboxes[i]:
                temp_box = []
                # 取得每次迭代的框的左上右下角的坐标点
                x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
                # 若是第一张图片对应的框
                if i == 0:
                    # 当左上点的坐标大于cuty与cutx 就不要这个框,因为这个框整个都出界了 第一张图片的范围为0:cuty 0:cutx
                    if y1 > cuty or x1 > cutx:
                        continue
                    # 当右下角的y2大于cuty 左上角的y1小于cuty时将 右下角的y2设置为cuty
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    # 当走下角的x2大于cutx 右上角的x1 小于cutx时 将右下角的x2设置为cutx
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                # 若是第二张图片对应的框
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx















































