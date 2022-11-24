import os
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

from .utils_bbox import DecodeBox
from .utils import cvtColor,resize_image,preprocess_input
# 创建损失历史类
class LossHistory():
    def __init__(self,log_dir,model,input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        # 创建 文件夹
        os.makedirs(self.log_dir)
        # 实例化SummaryWriter,传入保存的文件路径
        self.writer = SummaryWriter(self.log_dir)
        try:
            dimmy_input = torch.randn(2,3,input_shape[0],input_shape[1])
            self.writer.add_graph(model,dimmy_input)
        except:
            pass

    # 记录loss
    def append_loss(self,epoch,loss,val_loss):
        # 判断存储日志与权重的文件夹是否存在,不存在就创建
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # 存入本次epoch的训练loss与验证loss
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 创建txt文件
        with open(os.path.join(self.log_dir,"epoch_loss.txt"),'a') as f:
            # 写入loss
            f.write(str(loss))
            # 加上换行符
            f.write("\n")
        with open(os.path.join(self.log_dir,"epoch_val_loss.txt"),'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        # 调用方法 添加标量
        self.writer.add_scalar('loss',loss,epoch)
        self.writer.add_scalar('val_loss',val_loss,epoch)
        # 画图
        self.loss_plot()
    def loss_plot(self):
        # 计算有多少个loss
        iters = range(len(self.losses))
        # 创建画布
        plt.figure()
        # 描绘数据plt.plot(x,y)
        plt.plot(iters,self.losses,'red',linewidth = 2,label = 'train loss')
        plt.plot(iters,self.val_loss,'coral',linewidth = 2,label = 'val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters,scipy.signal.savgol_filter(self.losses,num,3),'green',linestyle = '--',linewidth = 2,label='smooth train loss')
            plt.plot(iters,scipy.signal.savgol_filter(self.val_loss,num,3),'#8B4513',linestyle = '--',linewidth = 2,label='smooth val loss')
        except:
            pass

        # 添加各类标签
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc = "upper right")
        # 保存
        plt.savefig(os.path.join(self.log_dir,"epoch_loss.png"))
        plt.cla()

        plt.close("all")

class EvalCallback():
    # 初始化
    def __init__(self,net,input_shape,anchors,anchors_mask,class_names,num_classes,val_lines,log_dir,cuda,
                 map_out_path='.temp_map_out',max_boxes=100,confidence=0.05,nms_iou=0.5,letterbox_image=True,
                 MINOVERLAP=0.5,eval_flag=True,period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        self.bbox_util = DecodeBox(self.anchors,self.num_classes,(self.input_shape[0],self.input_shape[1]),
                                   self.anchors_mask)
        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir,"epoch_map.txt"),'a') as f:
                f.write(str(0))
                f.write("\n")
    #
    def get_map_txt(self,image_id,image,class_names,map_out_path):
        # 创建文件
        f = open(os.path.join(map_out_path,"detection-results/" + image_id + ".txt"),"w",encoding='utf-8')
        # 读取图片长宽大小
        image_shape = np.array(np.shape(image)[0:2])
        # 将图片转化为RBG图像防止灰度图预测时报错
        # 代码支持RBG图像预测,其他图像都要转化为RBG
        image = cvtColor(image)
        # 给图像增加灰条,实现不失真的resize
        image_data = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)
        # 将通道数提到第0维度,然后在0维度添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data,dtype='float32')),(2,0,1)),0)
        # 不计算梯度
        with torch.no_grad():
            # 将图像数据转变为tensor
            images = torch.from_numpy(image_data)
            # 是否使用gpu
            if self.cuda:
                # 将数据放在gpu上
                images = images.cuda()
            # 将图像输入网络中进行预测
            outputs = self.net(images)
            # 进行验证阶段的解码
            outputs = self.bbox_util.decode_box(outputs)
            # 将预测框进行堆叠,然后进行非极大大值抑制 outputs是个每个特征图特征后处理后的列表
            results = self.bbox_util.non_max_suppression(torch.cat(outputs,1),self.num_classes,self.input_shape,
                                                         image_shape,self.letterbox_image,conf_thres = self.confidence,
                                                         nms_thres = self.nms_iou)

            # 判断如果results没有框返回空
            if results[0] is None:
                return
            # 类别
            top_label = np.array(results[0][:,6],dtype='int32')
            # 种类置信度
            top_conf = results[0][:,4] * results[0][:,5]
            # 框的坐标
            top_boxes = results[0][:,:4]
        # 取top_conf从大到小排列的前一百的索引 np.argsort是按从小到大 [::-1]是取反
        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        # 提取对应的前面100的数据,不能超过100个
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]
        # 按预测了多少个进行循环
        for i,c in list(enumerate(top_label)):
            # 提取预测的数据
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            # y1,x1,y2,x2
            top,left,bottom,right = box
            # 预测的类不在class_names中则直接进行下次循环
            if predicted_class not in class_names:
                continue

            # 将结果写入
            f.write("%s %s %s %s %s %s\n" % (predicted_class,score[:6],str(int(left)),str(int(top)),str(int(right)),str(int(bottom))))
        # 关闭文件
        f.close()
        return







    # 在一个epoch结束后,调用
    def on_epoch_end(self,epoch,model_eval):
        # 如果epoch整除10且 eval_flag 为True才进行内部操作
        if epoch % self.period == 0 and self.eval_flag:
            # 更新下网络
            self.net = model_eval
            # 判断下列各文件夹是否存在,不存在就创建
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path,"groud-truth")):
                os.makedirs(os.path.join(self.map_out_path,"ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path,"detection-results")):
                os.makedirs(os.path.join(self.map_out_path,"detection-results"))
            print("建图.")
            # 循环提取验证集数据
            for annotation_line in tqdm(self.val_lines):
                # 分割字符串
                line = annotation_line.split()
                # 返回文件名 效果如:000001.jpg -> 000001
                image_id = os.path.basename(line[0]).split('.')[0]
                # 读取图片
                image = Image.open(line[0])
                # 获取预测框 效果如:48,240,195,371,11 8,12,352,498,14 ->[[48,240,195,371,11],[8,12,352,498,14]]
                gt_boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                # 获得预测txt
                self.get_map_txt(image_id,image,self.class_names,self.map_out_path)






