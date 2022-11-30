import os
from utils.utils import get_classes
import xml.etree.ElementTree as ET
from yolo import YOLO
from tqdm import tqdm
from utils.utils_map import get_map
from PIL import Image
if __name__ == "__main__":
    # map_mode 用于指定文件运行时计算的内容
    map_mode = 0
    # classes_path 指定类别路径
    classes_path = 'model_data/voc_classes'
    # MINOVERLAP 用于指定获得mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    MINOVERLAP = 0.5
    # 由于mAP计算原理的限制,网络在计算mAP时要近乎所有的预测框,这样才可以计算mAP
    # 所以confidence的值设置的尽量小
    confidence = 0.001
    # 预测时使用的非极大值抑制的大小
    mns_iou = 0.5
    # 设置score_threhold
    score_threhold = 0.5
    # map_vis 用于指定是否开启VOC_map 计算的可视化
    map_vis = False
    # VOCdevkit_path 指定VOC数据集的文件夹
    VOCdevkit_path = 'VOCdevkit'
    # 结果输出的文件夹
    map_out_path = 'map_out'

    # 读取数据id
    image_ids = open(os.path.join(VOCdevkit_path,"VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    # 创建文件夹
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path,'ground-truth')):
        os.makedirs(os.path.join(map_out_path,'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path,'detection-results')):
        os.makedirs(os.path.join(map_out_path,'detection-results'))
    if not os.path.exists(os.path.join(map_out_path,'images-optional')):
        os.makedirs(os.path.join(map_out_path,'images-optional'))

    # 获取类的列表
    class_names , _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("加载模型.")
        yolo = YOLO(confidence = confidence,mns_iou = mns_iou)
        print("加载模型完成")

        print("得到预测结果.")
        # 循环提取图片
        for image_id in tqdm(image_ids):
            # 获取图片路径
            image_path = os.path.join(VOCdevkit_path,"VOC2007/JPEGImages/" + image_id + ".jpg")
            # 读取图片
            image = Image.open(image_path)
            # 判断是否进行可视化
            if map_vis:
                image.save(os.path.join(map_out_path,"images-optional" + image_id + ".jpg"))

            yolo.get_map_txt(image_id,image,class_names,map_out_path)
        print("得到预测结果完成.")

    if map_mode == 0 or map_mode == 2:
        print("得到真实框")
        for image_id in tqdm(image_ids):
            # 创建对应图像id的txt
            with open(os.path.join(map_out_path,"ground-truth/" + image_id + ".txt"),"w") as new_f:
                # 解析图像id 对应的xml文件
                root = ET.parse(os.path.join(VOCdevkit_path,"VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    # 只要不为None,就赋值
                    if obj.find('difficult') != None:
                        # 将值赋值给difficult
                        difficult = obj.find('difficult').text
                        # difficult == 1
                        if int(difficult) == 1:
                            difficult_flag = True
                    # 提取类名
                    obj_name = obj.find('name').text
                    # 如果不输入class_names的里的类,就进行下次循环
                    if obj_name not in class_names:
                        continue
                    #
                    bndbox = obj.find('bndbox')
                    # 提取坐标
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    # 如果difficult_flag 是True
                    # 写入
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name,left,top,right,bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name,left,top,right,bottom))
        print("获得真实框完成.")

    if map_mode == 0 or map_mode == 3:
        print("得到map.")
        get_map(MINOVERLAP,True,score_threhold = score_threhold,path = map_out_path)
        print("得到map完成.")























