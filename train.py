# 使用数据进行训练

import os
import numpy as np
import torch
if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    # 是否使用单机多卡分布式运行
    distributed = False
    # 若使用DDP多卡,可用
    sync_bn = False
    # 是否使用混合精度训练
    fp16 = False
    # classes_path 类别路径文件
    classes_path = "model_data/voc_classes.txt"
    # anchors_path 先验框对应的txt文件
    anchors_path = 'model_data/yolo_anchors.txt'
    # anchors_mosk 帮助代码找到对应的先验框
    anchors_path = [[6,7,8],[3,4,5],[0,1,2]]
    # model_path 加载预训练权重
    model_path = ''
    # input_shape 传入输入图片shape大小
    input_shape = [640,640]
    # backbone 传入主体特征提取网络backbone
    backbone = 'cspdarknet'
    # pretrained 如果设置model_path 这里就无需加载
    pretrained = False
    # phi 用于选择yolov5的版本
    phi = 's'
    # mosaic 是否使用马赛克数据增强
    mosaic = True
    # 设置进行马赛克数据增强的概率
    mosaic_prob = 0.5
    # 是否使用mixup数据增强,且仅在mosaic=True时有效 但还未实现
    mixup = True
    # 设置进行mixup数据增强的概率
    mixup_prob = 0.5






























