import os.path

import numpy as np
import torch
import torch.nn as nn
from utils.utils import get_classes,get_anchors,cvtColor,resize_image,preprocess_input
from utils.utils_bbox import DecodeBox
from utils.utils import show_config
import colorsys
from nets.yolo import YoloBody
from PIL import ImageFont,ImageDraw
# 用于预测的类
class YOLO(object):
    # 设置默认值字典配置
    _defaults = {
        # 模型权重
        "model_path" : "model_data/yolov5_s.pth",
        # 类别文件
        "classes_path" : "model_data/coco_classes.txt",
        # 先验框大小文件
        "anchors_path" : 'model_data/yolo_anchors',
        # 先验框掩码
        "anchors_mask" : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # 输入图片大小
        "input_shape" : [640,640],
        # 主干网络
        "backbone" : 'cspdarknet',
        # yolov5版本
        "phi" : "s",
        # 置信度阀值
        "confidence" : 0.5,
        # 非极大值阀值
        "nms_iou" : 0.3,
        # 是否进行不失真的图像缩放
        "letterbox_image" : True,
        # 是否使用GPU
        "cuda" : True,
    }

    # 得到默认值
    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "没有这个属性名'" + n + "'."
    # 初始化
    def __init__(self,**kwargs): # 如:传入**{'a': 1, 'b': 2, 'c': 3}
        # 将上面默认设置更新为类的字典
        self.__dict__.update(self._defaults)
        # 循环提取实例化对象参数
        for name,value in kwargs.items():
            # setattr 给对象设置key value
            setattr(self,name,value)
            self._defaults[name] = value
        # print(self.__dict__)
        # self.__dict__ {'model_path': 'model_data/yolov5_s.pth', 'classes_path': 'model_data/coco_classes.txt', 'anchors_path': 'model_data/yolo_anchors.txt', 'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]], 'input_shape': [640, 640], 'backbone': 'cspdarknet', 'phi': 's', 'confidence': 0.5, 'nms_iou': 0.3, 'letterbox_image': True, 'cuda': True, 'a': 1, 'b': 2, 'c': 3}
        # 获取种类和先验框数量
        self.class_names,self.num_classes = get_classes(self.classes_path)
        self.anchors,self.num_anchors = get_anchors(self.anchors_path)
        # 实例化解码类
        # print(self.anchors,self.anchors[8],self.anchors[7],self.anchors[6])
        self.bbox_util = DecodeBox(self.anchors,self.num_classes,(self.input_shape[0],self.input_shape[1]),self.anchors_mask)

        # 设置不同的颜色框 # [(0.0, 1.0, 1.0), (0.05, 1.0, 1.0)...]
        hsv_tuples = [(x / self.num_classes,1.0,1.0)for x in range(self.num_classes)]

        # (0.0, 1.0, 1.0) -> (1.0, 0.0, 0.0)  ->[(1.0, 0.0, 0.0), (1.0, 0.30000000000000004, 0.0)...]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
        # (1.0, 0.0, 0.0) -> (255, 0, 0) -> [(255, 0, 0), (255, 76, 0)...]
        self.colors = list(map(lambda x:(int(x[0] * 255) ,int(x[1] * 255), int(x[2] * 255)),self.colors))

        # 生成模型
        self.generate()

        # 显示配置
        # show_config(**self._defaults)

    def generate(self,onnx=False):
        # 建立yolo模型,载入yolo模型的权重
        self.net = YoloBody(self.anchors_mask,self.num_classes,self.phi,backbone=self.backbone,input_shape=self.input_shape)
        # 设置gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 载入权重
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        # 启动验证模式
        self.net = self.net.eval()
        print("{} 模型和类别已载入.".format(self.model_path))
        # 把网络模型放入gpu上加速
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
    # 检测图片
    def detect_image(self, image, crop = False, count = False):
        # 计算输入图片的高与宽
        image_shape = np.array(np.shape(image)[0:2])

        # 将图像转化为RBG图像
        image = cvtColor(image)
        # 给图像加灰条,进行不失真的resize
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        # 图像现在是 3维,得将通道数放前边,得还得在最前面加上一维 还要做归一化,把图像数据值映射到0-1之间 [bs,通道,宽,高]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2, 0, 1)),0)

        # 不计算梯度
        with torch.no_grad():
            # 将图像变为tensor
            images = torch.from_numpy(image_data)
            # 判断是否要使用GPU
            if self.cuda:
                # 数据放在GPU上
                images = images.cuda()
            # 预测,通过网络得到3个特征层
            outputs = self.net(images)
            # 对特征层解码,得到预测框
            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠,然后进行最大值抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,
                                                         image_shape,self.letterbox_image,conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            # 如果 没有检测到结果之间返回原图
            if results[0] is None:
                return image
            # 将预测到的种类提取出来 从tensor变为np.array
            top_label = np.array(results[0][:,6],dtype='int32')
            # 提取置信度 置信度是用是否又物体与物体的预测可信度相乘得到
            top_conf = results[0][:,4] * results[0][:,5]
            # 提取预测框
            top_boxes = results[0][:,:4]

        # 设置字体与边框厚度
        font = ImageFont.truetype(font="model_data/simhei.ttf",size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # 计数
        if count:
            # 打印这张图像预测出来的种类
            print("top_label:",top_label)
            # 创建20个0组成的一个向量
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                # 统计预测每个种类的个数
                num = np.sum(top_label == i )
                if num > 0:
                    # 打印预测的每个类别的个数
                    print(self.class_names[i], " : ",num)
                # 保存预测的每个类别的个数
                classes_nums[i] = num
            # 打印
            print("classes_nums:",classes_nums)
        # 是否进行目标的裁剪
        if crop:
            for i,c in list(enumerate(top_boxes)):
                # 提取框的坐标 x1,y1,x2,y2
                top,left,bottom,right = top_boxes[i]
                # 将左上角的坐标是为负值和不是正整数的变为0或者其最大值的正整数
                top = max(0,np.floor(top).astype('int32'))
                letf = max(0,np.floor(left).astype('int32'))
                # 将超出的右下角坐标变为原图的最大宽与长,如果右下角小于原图大小,则是取其右下角的正整数
                bottom = min(image.size(1),np.floor(bottom).astype('int32'))
                right = min(image.size(0),np.floor(right).astype('int32'))
                # 保存的文件夹路径
                dir_save_path = "img_crop"
                # 判断文件夹是否存在,不存在就创建
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                # 截取预测框图片
                crop_image = image.crop([letf,top,right,bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            # 提取预测的类名
            predicted_class = self.class_names[int(c)]
            # 提取预测的框的坐标
            box = top_boxes[i]
            # 提取置信度
            score = top_conf[i]

            top,left,bottom,right = box
            # 对坐标进行合理的出来,比如让规范出界的坐标值
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # 做类名与置信度的字符串
            label = '{} {:.2f} '.format(predicted_class,score)
            draw = ImageDraw.Draw(image)
            # 设置文字大小
            label_size = draw.textsize(label,font)
            # 设置文字编码
            label = label.encode('utf-8')
            print(label,top,left,bottom,right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i,top + i,right - i,bottom - i],outline=self.colors[c])
            draw.rectangle([tuple(text_origin),tuple(text_origin + label_size)],fill=self.colors[c])
            draw.text(text_origin,str(label,'UTF-8'),fill=(0,0,0),font=font)
            del draw
        return image
    # 得到txt
    def get_map_txt(self, image_id, image, class_names,map_out_path):
        # 创建对应图像id的txt
        f = open(os.path.join(map_out_path,"detection-results/" + image_id + ".txt"),"w",encoding='utf-8')
        # 提取图片的shape
        image_shpae = np.array(np.shape(image)[0:2])
        # 将图片转化为RGB图像
        image = cvtColor(image)
        # 给图像增加灰条,实现不失真的resize
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2, 0, 1)),0)

        # 不计算梯度
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            # 是否使用gpu
            if self.cuda:
                images = images.cuda

            # 将图像输入网络中进行预测
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            # 堆叠预测框,进行非极大值抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,
                                                         image_shpae,self.letterbox_image,conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou
                                                         )

            # 如果没有框直接返回
            if results[0] is None:
                return
            # 提取预测的类别数
            top_label = np.array(results[0][:,6],dtype='int32')
            # 置信度
            top_conf = results[0][:,4] * results[0][:,5]
            # 框坐标
            top_boxes = results[0][:,:4]

        # 循环提取并保存
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            # 坐标 y1,x1,y2,x2
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue
            # 写入
            f.write("%s %s %s %s %s %s\n" % (predicted_class,score[:6],str(int(left)),str(int(top)),str(int(right)),str(int(bottom))))

        f.close()
        return






















if __name__ == '__main__':
    print(np.zeros([20]))
    # yolo = YOLO()
    # img = torch.rand([416,416,3])
    # yolo.detect_image(image=img)




































