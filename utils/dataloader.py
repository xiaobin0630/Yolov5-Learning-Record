from random import sample,shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from utils.utils import cvtColor,preprocess_input

from torch.utils.data.dataset import Dataset
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio
        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4

    # 返回数据数
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # 当self.mosaic为True 且 self.rand()数 < self.mosaic_pro 再且self.epoch_now < self.epoch_length * self.special_aug_ratio时
        # 进行mosaic数据增强
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            # 随机提取3张图片
            lines = sample(self.annotation_lines, 3)
            # 再将index对应的索引添加道lines列表里，构成4张图片
            lines.append(self.annotation_lines[index])
            # 打乱图片顺序
            shuffle(lines)
            # 进行mosaic数据增强
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        # 对象图像的数据类型变为float32,且在除以255,最后调整维度顺序 吧最后一维度数放第一维度位置上
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            # 对真实框进行归一化 调整到0-1之间
            # 对应图像的框的x1与x2除以input_shape[1]
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            # 对应图像的框的y1与y2除以input_shape[0]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # 将左上右下两坐标表示法变为中心宽高表示法
            # 序号2,3为框的宽高
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            # 序号0,1为框的中心
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        return image, box, y_true

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    def get_target(self, targets):
        # 计算特征层数
        num_layers  = len(self.anchors_mask)
        # 将列表转化为np.array

        input_shape = np.array(self.input_shape, dtype='int32')
        # 获取网格长宽的倍率因子 [(20,20) (40,40),(80,80)]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        # 生成[[3,20,20,5 + num_class],[3,40,40,5 + num_class],[3,80,80,5 + num_class]]零矩阵列表
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        # 生成[[3,20,20],[3,40,40],[3,80,80]]零矩阵列表
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        # 如果没有框就直接返回y_true
        if len(targets) == 0:
            return y_true
        # 逐层循环
        for l in range(num_layers):
            # 提取当前特征层的高宽

            in_h, in_w = grid_shapes[l]
            # 将anchors转化为np.array 在除以倍率因子,得到anchors对应特征图的anchors大小

            anchors = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]

            batch_target = np.zeros_like(targets)
            # 计算处正样本在特征层上的中心点
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]

            # np.expand_dims增加维数

            # np.expand_dims(batch_target[:,2:4],1)在第一维度上增加一维 (num_true_box,2) -> (num_true_box,1,2)
            # np.expand_dims(anchors,0)在零维度上增加一维 (9,2) -> (1,9,2)
            # 每一个真实框和每一个先验框的宽高比,最后的shape为(num_true_box,9,2)
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            # 同上,反过来,每一个先验框和每一个真实框的宽高比,最后的shape为(num_true_box,9,2)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            # 将ratios_of_gt_anchors,ratios_of_anchors_gt 在最后一维度进行拼接 (num_true_box,9,4)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            # 取ratios 每个真实框和每个先验框的宽高比值的最大值 (num_true_box,9)
            max_ratios = np.max(ratios, axis=-1)

            # (num_true_box,9)
            for t, ratio in enumerate(max_ratios):
                # 判断ratio  其中有9个值
                over_threshold = ratio < self.threshold
                # 至少有一个为真
                over_threshold[np.argmin(ratio)] = True
                # 提取anchors_mask列表 [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                for k, mask in enumerate(self.anchors_mask[l]):
                    # 如果mask列表在over_threshold中全是False进行下一次循环
                    if not over_threshold[mask]:
                        continue
                    # 提取中心点并向下取整
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    # 循环提取列表中的列表 如:[[0,0],[1,0],[0,1]]
                    for offset in offsets:
                        # 取整后的中心加上对应的坐标数
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        # 当local_i大于特征图的宽或者小于0或者local_j大于特征图的高或者小于0 直接进行下一次循环
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue
                        #
                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                        # 提取真实框的种类
                        c = int(batch_target[t, 4])
                        # tx,ty代表代表中心调整参数的真实值
                        # 将对应的框框值放入网络应该预测的y_true,简单来说,是在为网络做正确预测的模板值
                        # 对应的中心坐标与宽高
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        # 表示该框有物体
                        y_true[l][k, local_j, local_i, 4] = 1
                        # 在对应的类别上吧0设置为1,表示该框的真实类别
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        # 获取当前先验框的最好比列
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
        # 返回网络模型最后预测的真实模板
        return y_true
    # 获取附近的点
    def get_near_points(self, x, y, i, j):
        # 取整后的x,y与未取整的x,y求出差值
        sub_x = x - i
        sub_y = y - j
        # 取上与右还有自身[0,0]
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        # 取上与左还有自身[0,0]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        # 取下与左还有自身[0,0]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        # 取下与右还有自身[0, 0]
        else:
            return [[0, 0], [1, 0], [0, -1]]




    # 获取随机数函数
    def rand(self,a = 0,b = 1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        # 按空格切割字符串
        # 如字符串 D:\AI-Project\Yolov5-Learning-Record\VOCdevkit/VOC2007/JPEGImages/000001.jpg 48,240,195,371,11 8,12,352,498,14
        # line[0] D:\AI-Project\Yolov5-Learning-Record\VOCdevkit/VOC2007/JPEGImages/000001.jpg
        # line[1] 48,240,195,371,11
        # line[2] 8,12,352,498,14
        line = annotation_line.split()
        # 读取图像并转化为RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        # 获取图像的高宽与目标高宽
        iw, ih = image.size
        h, w = input_shape
        # 获取框的坐标
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # 如果为非训练模式
        if not random:
            # input_shape 比上image 得出长宽比列
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            # 将图像多余的部分加上灰条
            # 将图像resize为(nw,nh)
            image = image.resize((nw,nh), Image.BICUBIC)
            # 新建一张大小为(w,h)的RGB图像
            new_image = Image.new('RGB', (w,h), (128,128,128))
            # 将image贴到new_image图像上(dx,dy)位置
            new_image.paste(image, (dx, dy))

            image_data = np.array(new_image, np.float32)

            # 对真实框调整
            if len(box)>0:
                # 随机打乱
                np.random.shuffle(box)
                # 将框的坐标转化到变化后的图像对应的坐标上去
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                # 将左上角坐标出小于0坐标全部设置为0
                box[:, 0:2][box[:, 0:2]<0] = 0
                # 右下角的x 如果超出了input_shape的w则设为w
                box[:, 2][box[:, 2]>w] = w
                # 右下角的y 如果超出了input_shape的h则设为y
                box[:, 3][box[:, 3]>h] = h
                # 计算每个框对应的长宽
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 将长宽大于1像素点的留下
                box = box[np.logical_and(box_w>1, box_h>1)]
            return image_data, box

        # 对象图像进行缩放且进行长宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25,2)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        # resize图像
        image = image.resize((nw,nh),Image.BICUBIC)

        # 加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 图像翻转
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        # 计算色域变化参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # 变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # 合并通道
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # 将图像转为RBG图像
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            # 如何图像翻转过,对应的坐标也要进行翻转调整
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 丢弃无效框
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box





    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        # 获取长宽
        h, w = input_shape

        # 获取0.3-0.7的随机数
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
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
            iw, ih = image.size
            # 保存框的位置[[48,240,195,371,11],[8,12,352,498,14]]
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])

            # 翻转图片
            flip = self.rand()<.5
            # 当数据数小于0.5且有框进行图片左右翻转
            if flip and len(box)>0:
                # 左右翻转
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 对应图片的标签进行调整
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对图像进行缩放并且进行长和宽的扭曲
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            # 当new_ar小于1就在高上进行,当大于1就在宽上进行
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

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
            new_image = Image.new('RGB', (w,h), (128,128,128))
            # 将image的大小为(dx,dy)的图片放在new_image图像上
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            # index加1
            index = index + 1
            box_data = []

            # 对框重新进行处理
            if len(box)>0:
                # 打乱顺序
                np.random.shuffle(box)
                # 对标签进行相同的缩放与扭曲
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                # 将左上角的坐标(x,y)小于0的都设置为0,防止左上角坐标出界
                box[:, 0:2][box[:, 0:2]<0] = 0
                # 将右下角的坐标长宽大于w,h的全部设置为w,h防止右下角坐标出界
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                # 获取框的宽与高
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 获取box_w与box_h都大于1像素点的框
                box = box[np.logical_and(box_w>1, box_h>1)]
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
        new_image = np.zeros([h, w, 3])
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
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        # 将图像转到HSV上 通过cv2split拆分通道
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype

        # 变换
        # 创建0-256的array
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        # np.clip(a,min,max)所有小于min 都被设置为min，所有大于max都被设置为max
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 通过cv2.LUT来查找table将hue转到对应的lut_hue  cv2.merge()用来合并通道
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # 将HSV图像转为RGB图像
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # 对框进行进一步处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        # 4张图片 进行循环处理
        for i in range(len(bboxes)):
            # 循环第一张图片对应的框
            for box in bboxes[i]:
                tmp_box = []
                # 取得每次迭代的框的左上右下角的坐标点
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
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
                # 若是第二张图片对应的框 范围为cuty:, :cutx 当右下角坐标y2小于cuty，而左上角x1大于cutx,则不要这个框,因为出界了
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    # y2与y1 最小值为y1
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    # x1与x2最大值为cutx
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                # 若是第三张图片 范围为 cuty:, cutx:
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                # 第四张图片 :cuty, cutx:
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                # 吧对应图片对应的框保存在列表里
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                # 汇总
                merge_bbox.append(tmp_box)
        return merge_bbox



# 如何读取数据
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    y_trues = [[] for _ in batch[0][2]]
    # 循环提取对应的数据
    for img, box,y_true in batch:
        images.append(img)
        bboxes.append(box)
        for i, sub_y_true in enumerate(y_true):

            y_trues[i].append(sub_y_true)
    # 将numpy格式数据变为tensor
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # 因为循环,所以得在假一层列表,将每个数据放在列表中
    bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    return images, bboxes, y_trues












































