import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

classes_path = "model_data/voc_classes"
classes, _ = get_classes(classes_path)
def convert_annotation(year,image_id,list_file):
    in_file = open(os.path.join(VOCdevkit_path,'VOC%s/Annotations/%s.xml'%(year,image_id)),encoding='utf-8')
    # 解析xml格式
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 从object标签(一张图片对应的框可能有多个)对开始提取对应数据
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        # 获取类的id
        cls_id = classes.index(cls)
        # 获取框框
        xmlbox = obj.find('bndbox')
        # 保存为元组(xmin,ymin,xmax,ymax)
        b = (int(float(xmlbox.find('xmin').text)),int(float(xmlbox.find('ymin').text)),int(float(xmlbox.find('xmax').text)),int(float(xmlbox.find('ymax').text)))
        # 写入txt文件中
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
if __name__ == "__main__":
    # 存放voc类的路径
    classes_path = "model_data/voc_classes"
    # (训练集+验证集):测试集 = 9:1
    trainval_percent = 0.9
    # 训练集:验证集 = 9:1
    train_percent = 0.9
    # voc数据集的根目录
    VOCdevkit_path = 'VOCdevkit'
    VOCdevkit_sets = [('2007','train'),('2007','val')]
    # 获取类名
    clases, _ = get_classes(classes_path)
    # 统计目标数量
    photo_nums = np.zeros(len(VOCdevkit_sets))
    nums = np.zeros(len(clases))
    # 设置随机种子
    random.seed(0)

    # 判断数据路径不能有空格
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称有空格")

    # 在ImageSets 文件下生成txt
    xmlfilepath = os.path.join(VOCdevkit_path,'VOC2007/Annotations')
    saveBasePath = os.path.join(VOCdevkit_path,'VOC2007/ImageSets/Main')
    # 读取xmlfilepath路径下的xml文件
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    # 获取xml文件
    for xml in temp_xml:
        # 判断以.xml结尾的
        if xml.endswith(".xml"):
            total_xml.append(xml)
    # 获取xml的数量
    num = len(total_xml)
    # 创建num的数字列表
    list = range(num)
    # 获取训练集与验证集的大小
    tv = int(num * trainval_percent)
    # 获取训练集大小
    tr = int(tv * train_percent)

    # 获取训练集与验证集的随机样本索引
    trainval = random.sample(list,tv)
    # 获取训练集随机样本索引
    train = random.sample(list,tr)

    # 创建保存对应数据的txt
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'),'w')
    ftest = open(os.path.join(saveBasePath,'test.txt'),'w')
    ftrain = open(os.path.join(saveBasePath,'train.txt'),'w')
    fval = open(os.path.join(saveBasePath,'val.txt'),'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("完成在ImageSets下txt")

    # 生成2007_train.txt 与 2007_val.txt 为后面训练做准备
    type_index = 0
    for year,image_set in VOCdevkit_sets:
        # 读取对应的训练与验证的数据ids
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            # 写入image的绝对路径
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path),year,image_id))
            # 写入对应图片的类别与左上角与右下角坐标点
            convert_annotation(year,image_id,list_file)
            # 加入换行符
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print('完成2007_train.txt 和 2007_val.txt生成')
























































