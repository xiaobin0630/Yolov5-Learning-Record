import glob
import os
import shutil
import matplotlib
import sys
import json
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# 抛出错误,并退出
def error(msg):
    print(msg)
    sys.exit(0)
# 将文件中的所有行保存到列表中
def file_lines_to_list(path):
    # 按行读取
    with open(path) as f:
        content = f.readlines()
    # 去除每行的'\n'
    content = [x.strip() for x in content]
    return content


# 获得map
def get_map(MINOVERLAP,draw_plot,score_threhold=0.5,path='./map_out'):
    # 真实框路径文件夹
    GT_PATH = os.path.join(path,'ground-truth')
    # 预测框路径文件夹
    DR_PATH = os.path.join(path,'detection-results')
    # 图像路径
    IMG_PATH = os.path.join(path,'images-optional')
    # 暂存路径文件夹
    TEMP_FILES_PATH = os.path.join(path,'.temp_files')
    # 结果路径文件夹
    RESULTS_FILES_PATH = os.path.join(path,'results')

    # 是否展现动画
    show_animation = True
    # 判断IMG_PATH文件夹是否存在
    if os.path.exists(IMG_PATH):
        for dirpath,dirname,files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False

    # 创建TEMP_FILES_PATH
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        # 创建RESULTS_FILES_PATH
        os.makedirs(RESULTS_FILES_PATH)

    # 是否画图
    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass

        # 创建文件夹
        os.makedirs(os.path.join(RESULTS_FILES_PATH,"AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))

    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

    # 获取所有真实框的txt文件路径
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')

    # 判断文件路径
    if len(ground_truth_files_list) == 0:
        error("错误: 没发现真实框文件!")
    # 整理
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    # 循环提取真实框路径
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt",1)[0]
        # 获取file_id
        file_id = os.path.basename(os.path.normpath(file_id))
        # 合成文件路径
        temp_path = os.path.join(DR_PATH,(file_id + ".txt"))
        # 判断对应预测文件是否存在
        if not os.path.exists(temp_path):
            error_msg = "出现错误 没找到文件: {}\n".format(temp_path)
            error(error_msg)
        # 将真实框的数据按行提取保存在列表中
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        # 遍历列表
        for line in lines_list:
            try:
                # 判断
                if "difficult" in line:
                    # 按空格分割 提取数据
                    class_name,left,top,right,bottom,_difficult = line.split()
                    is_difficult = True
                else:
                    class_name,left,top,right,bottom = line.split()
            except:
                if "difficut" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    class_name = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    class_name = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]
            # 合并
            bbox = left + " " + top + " " + right + " " + bottom
            # 保存在列表中
            if is_difficult:
                bounding_boxes.append({"class_name" : class_name,"bbox" : bbox,"used": False,"difficult" : True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name" : class_name,"bbox": bbox,"used":False})
                # 计算每个类的数
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1

                    already_seen_classes.append(class_name)
        # 保存这个真实框的相关数据
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json",'w') as outfile:
            json.dump(bounding_boxes,outfile)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    # 真实框的类别数
    n_classes = len(gt_classes)

    # 获取预测框的结果
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    # 按类别名循环
    for class_index,class_name in enumerate(gt_classes):
        bounding_boxes = []
        # 循环每一个预测结果文件
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH,(file_id + ".txt"))
            # 判断temp_path是否存在
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "出现错误,没有发现文件: {}\n".format(temp_path)
                    error(error_msg)
            # 按行读取存入列表中
            lines = file_lines_to_list(txt_file)
            # 循环提取lines的值进行处理
            for line in lines:
                try:
                    tem_class_name,confidence,left,top,right,bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence" : confidence,"file_id" : file_id,"bbox" : bbox})
        # 按置信度进行sort
        bounding_boxes.sort(key=lambda x:float(x['confidence']),reverse=True)
        # 保存
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json",'w') as outfile:
            json.dump(bounding_boxes,outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(RESULTS_FILES_PATH + "/result.txt","w") as results_file:
        results_file.write("# AP and percision/recall per class\n")
        count_true_positive = {}
        for class_index,class_name in enumerate(gt_classes):
            count_true_positive[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score_threhold_idx = 0
            for idx,detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score






















































