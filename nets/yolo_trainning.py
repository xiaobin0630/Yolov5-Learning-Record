import math
from copy import deepcopy
from functools import partial
import numpy as np
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self,anchors,num_classes,input_shape,cuda,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]],label_smoothing = 0):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4,1.0,4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape [1]) / (640 ** 2)
        self.cuda = cuda

    # l代表那一层的特征图序号,input代表特征图,targets真实标签,y_true代表网络应该预测为的真实模板
    def forward(self,l,input,targets=None,y_true=None):
        # l代表第几的特征层
        # input的shape为 (bs, 3 * (5 + num_classes), 20, 20)
        #               (bs, 3 * (5 + num_classes), 40, 40)
        #               (bs, 3 * (5 + num_classes), 80, 80)
        # targets 真实框的标签情况 [batch_size,num_gt,5]

        # 获取图片数量,特征图的高与宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长
        # 每一个特征点对应原图片上多少个像素点
        # [640,640]
        # 特征层20x20 ,一个特征点对应原来的图片上32个像素
        # 特征层40x40 ,一个特征点对应原来的图片上16个像素
        # 特征层20x20 ,一个特征点对应原来的图片上8个像素
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        # 获取相对于特征层的scaled_anchors
        scaled_anchors = [(a_w / stride_w,a_h / stride_h)for a_w,a_h in self.anchors]

        # input输入有三类 (bs,3 * (5+num_classes),20,20)(bs,3 * (5+num_classes),40,40)(bs,3 * (5+num_classes),80,80)
        # (bs,3 * (5+num_classes),20,20) -> (bs,3,5+num_clsses,20,20) -> (bs,3,20,20,5+num_classes)
        prediction = input.view(bs,len(self.anchors_mask[l]),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()
        # 获取先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        # 获取先验框的宽高调整参数
        w = torch.sigmoid(prediction[...,2])
        h = torch.sigmoid(prediction[...,3])
        # 获取置信度,是否有物体
        conf = torch.sigmoid(prediction[...,4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[...,5:])

        # 将预测结果进行解码,判断预测结果和真实值的重合程度
        pred_boxes = self.get_pred_boxes(l,x,y,h,w,targets,scaled_anchors,in_h,in_w)

        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0
        # 计算有多少个真实框
        n = torch.sum(y_true[...,4] == 1)
        if n != 0:
            # 计算预测结果和真实结果的giou
            giou = self.box_giou(pred_boxes,y_true[...,:4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[y_true[...,4] == 1])
            # 计算类别分类loss
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[...,4] == 1],self.smooth_labels(y_true[...,5:][y_true[...,4] == 1],self.label_smoothing,self.num_classes)))
            # 汇总框与分类loss
            loss += loss_loc * self.bbox_attrs + loss_cls * self.cls_ratio
            # 计算置信度的loss 意味着先验框对应的预测框预测的更准确
            # 它用来预测这个物体的 torch.where(a>0,a,b) 函数作用是按照一定的规则合并两个tensor 满足a>0 返回a 否则返回b
            tobj = torch.where(y_true[...,4] == 1,giou.detech().clamp(0),torch.zeros_like(y_true[...,4]))
        else:
            tobj = torch.zeros_like(y_true[...,4])
        # 计算置信度loss
        loss_conf = torch.mean(self.BCELoss(conf,tobj))
        # 汇总置信度loss
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    # 平滑标签
    def smooth_labels(self,y_true,label_smoothing,num_classes): # label_smoothing = 0 num_classes = 20
        # 这儿等于没做 因为label_smoothing = 0
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    def clip_by_tensor(self,t,t_min,t_max):# t_min = 1e-7 t_max = 1 - t_min
        # 将预测结果数字类型变为float
        t = t.float()
        # t >= 1e-7 然后数字类型变为float 乘以t + t < t_min 然后数字类型变为float 乘以 t_min
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    def BCELoss(self,pred,target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred,epsilon,1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    def box_giou(self,b1,b2): # pred_boxes,y_true
        # 预测框左上右下角
        # 预测框的中心坐标与长宽大小
        b1_xy = b1[...,:2]
        b1_wh = b1[...,2:4]
        # 求预测框长宽的一半
        b1_wh_half = b1_wh / 2.
        # 预测左上角坐标
        b1_mins = b1_xy - b1_wh_half
        # 预测右下角坐标
        b1_maxes = b1_xy + b1_wh_half

        # 真实框左上角右下角
        # 真实的中心坐标与长宽大小
        b2_xy = b2[...,:2]
        b2_wh = b2[...,2:4]
        # 求真实框长宽的一半
        b2_wh_half = b2_wh / 2.0
        # 真实左上角坐标
        b2_mins = b2_xy - b2_wh_half
        # 真实右下角坐标
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框与预测框的所有的iou
        # 求出真实框与预测框所有左上角的坐标的最大值坐标
        intersect_mins = torch.max(b1_mins,b2_mins)
        # 求出真实框与预测框所有右下角的坐标的最小值坐标
        intersect_maxes = torch.min(b1_maxes,b2_maxes)
        # 求出真实框与预测框所有左上角的坐标的最大值坐标与真实框与预测框所有右下角的坐标的最小值坐标的长宽,如果长宽为负值则设置为0
        intersect_wh = torch.max(intersect_maxes - intersect_mins,torch.zeros_like(intersect_maxes))
        # 求出真实框与预测框的交集的面积
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        # 求出预测框的面积
        b1_area = b1_wh[...,0] * b2_wh[...,1]
        # 求出真实框的面积
        b2_area = b2_wh[...,0] * b2_wh[...,1]
        # 求出预测框与真实框并集面积
        union_area = b1_area + b2_area - intersect_area
        # 求出iou 交集面积/并集面积
        iou = intersect_area / union_area

        # 求出包裹两个框的左上角和右下角 就是刚好包住预测框与真实框的框框的左上角与右下角坐标
        # 求出预测框与真实框左上角的最小值坐标
        enclose_mins = torch.min(b1_mins,b2_mins)
        # 求出预测框与真实框右下角的最大值坐标
        enclose_maxes = torch.max(b1_maxes,b2_maxes)
        # 求出刚好包住预测框与真实框的框框长宽 如果长宽有负值就用 真实框与预测框所有右下角的坐标的最小值坐标值替换
        enclose_wh = torch.max(enclose_maxes - enclose_mins,torch.zeros_like(intersect_maxes))

        # 计算刚好包住预测框与真实框的框框的面积
        enclose_area = enclose_wh[...,0] * enclose_wh[...,1]
        # 计算机giou
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou







    def get_pred_boxes(self,l,x,y,h,w,targets,scaled_anchors,in_h,in_w):
        # 计算有多少张图片
        bs = len(targets)
        # 生成网格,先验框中心,网格左上角
        # (bs,3,20,20)
        grid_x = torch.linspace(0,in_w - 1,in_w).repeat(in_h,1).repeat(
            int(bs * len(self.anchors_mask[l])),1,1).view(x.shape).type_as(x)
        # (bs, 3, 20, 20)
        grid_y = torch.linspace(0,in_h - 1,in_h).repeat(in_w,1).t().repeat(
            int(bs * len(self.anchors_mask[l])),1,1).view(y.shape).type_as(x)

        # 生成先验框的宽高 [(),(),()]
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        # 转化为Tensor 提取scaled_anchors_l 宽 [[],[],[]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        # 转化为Tensor 提取scaled_anchors_l 高 [[],[],[]] shape (3,1)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        # (bs, 3, 20, 20)
        anchor_w = anchor_w.repeat(bs,1).repeat(1,1,in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs,1).repeat(1,1,in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x * 2.0 - 0.5 + grid_x,-1)
        pred_boxes_y = torch.unsqueeze(x * 2.0 - 0.5 + grid_y,-1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w,-1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],dim = -1)
        return pred_boxes

def is_parallel(model):
    # 如果模型是DP DDP则返回True
    return type(model) in (nn.parallel.DataParallel,nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # 如果这个模型是DP DDP 就返回单GPU模型
    return model.module if is_parallel(model) else model

def copy_attr(a,b,include=(),exclude=()):
    # 将属性从b复制到a
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
# EMA指数移动平均 作用提升模型性能
class ModelEMA:
    def __init__(self,model,decay=0.9999,tau=2000,updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        # 遍历ema的参数,设置为不需要自动求导记录
        for p in self.ema.parameters():
            p.requires_grad_(False)
    def update(self,model):
        # 更新ema参数
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = de_parallel(model).state_dict() # 包含state和param_groups的字典对象
            for k,v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
    def update_attr(self,model,include=(),exclude=('process_group','reducer')):
        # 更新ema属性
        copy_attr(self.ema,model,include,exclude)

# 模型权重值初始化
def weights_init(net,init_type='normal',init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        # hasattr用来判断m对象是否有'weight'属性
        if hasattr(m,'weight') and classname.find('Conv') != -1:
            # 给不同的类型赋予初始值
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data,0.0,init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_uniform_(m.weight.data,gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data,gain=init_gain)
            else:
                raise NotImplementedError('为找到[%s]初始化方法' % init_gain)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data,1.0,0.02)
            torch.nn.init.constant_(m.bias.data,0.0)
    print('初始化网络%s' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type,lr,min_lr,total_iters,warmup_iters_ratio = 0.05,warmup_lr_ratio = 0.1,no_aug_iter_ratio = 0.05,step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr
    # 判断是那种类别的学习率下降,再进行不同的处理
    # 余弦退火
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters,1),3)
        warmup_lr_start = max(warmup_lr_ratio * lr,1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters,1),15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func



def set_optimizer_lr(optimizer,lr_scheduler_func,epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





















