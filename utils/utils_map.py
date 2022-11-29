import glob
import os
import shutil
import matplotlib
import sys
import json
import math
import cv2
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
def voc_ap(rec,prec):
    # 在rec前面加0.0的值
    rec.insert(0, 0.0)
    # 在rec最后面加1.0的值
    rec.append(1.0)
    mrec = rec[:]
    # 在prec前面加0.0的值
    prec.insert(0, 0.0)
    # 在prec最后面加1.0的值
    prec.append(0.0)
    mpre = prec[:]
    # 从len(mpre) 一直到-1停止 步长为-1 训练两两比较,把较大值存入mpre[i]中 作用是是precision 从头到尾单调下降
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def log_average_miss_rate(precision,fp_cumsum,num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


# 获得map
def get_map(MINOVERLAP,draw_plot,score_threhold=0.5,path='./map_out'):
    # 真实框路径文件夹
    GT_PATH = os.path.join(path, 'ground-truth')
    # 预测框路径文件夹
    DR_PATH = os.path.join(path, 'detection-results')
    # 图像路径
    IMG_PATH = os.path.join(path, 'images-optional')
    # 暂存路径文件夹
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    # 结果路径文件夹
    RESULTS_FILES_PATH = os.path.join(path, 'results')

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
                    tmp_class_name,confidence,left,top,right,bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name  = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name  = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        # 按置信度进行sort
        bounding_boxes.sort(key=lambda x:float(x['confidence']),reverse=True)
        # 保存
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json",'w') as outfile:
            json.dump(bounding_boxes,outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # 创建文件夹
    with open(RESULTS_FILES_PATH + "/result.txt","w") as results_file:
        #
        results_file.write("# AP and precision/recall per class\n")
        #
        count_true_positives = {}
        # 循环按类进行
        for class_index, class_name in enumerate(gt_classes):
            # 对每一次循环的类,创建key与value 如:{'aeroplane' : 0}
            count_true_positives[class_name] = 0
            # 组合在.temp_files文件下的 如:aeroplane_dr.json文件路径
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            # 读取json格式数据
            dr_data = json.load(open(dr_file))
            # 统计在当前class_name次循环有多少条数据
            nd = len(dr_data)
            # 创建tp对应条数据那么多的列表[0...] tp truepositive
            tp = [0] * nd
            # 创建fp对应条数据那么多的列表[0...] fp falsepotive
            fp = [0] * nd
            # 创建score对应条数据那么多的列表[0...]
            score = [0] * nd
            score_threhold_idx = 0
            # 循环当前class_name的temp_dr.json数据
            for idx, detection in enumerate(dr_data):
                # 赋值id 如:file_id = 007857
                file_id = detection["file_id"]
                # 将对应条的数据confidence 保存在对应的score[idx]中
                score[idx] = float(detection["confidence"])
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx
                # 判断是否可视化 暂不实现
                if show_animation:
                    pass
                # 组成对应file_id 的_ground_truth.json文件路径
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                # 读取gt_file文件数据
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # 提取预测的框的坐标
                bb = [float(x) for x in detection["bbox"].split()]

                # 循环提取真实框数据
                for obj in ground_truth_data:
                    # 如果真实框的类别等于本次循环的class_name则进行进行内部操作
                    if obj["class_name"] == class_name:
                        # 得到真实坐标
                        bbgt = [ float(x) for x in obj["bbox"].split() ]
                        # 得到∩框 取真实框与预测框左上角最大坐标点与右下角最小坐标点
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]

                        # 得到∩框的宽高
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        # 判断∩框的宽高都大于0
                        if iw > 0 and ih > 0:
                            # (预测框的宽 + 1) * (预测框的高 + 1) + (真实框的宽 + 1) *(真实框的高 + 1) - iw*ih(∩框的宽高)
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            # 计算重叠率 就是交并比
                            ov = iw * ih / ua
                            # 判断
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if show_animation:
                    status = "未发现匹配!"


                min_overlap = MINOVERLAP

                # 判断交并比是否大于设置的最小重叠率
                if ovmax >= min_overlap:
                    # 判断字符串difficult 是否在真实框keys对里
                    if "difficult" not in gt_match:
                        # 判断gt_match的used是否已经判断过的,没有就进行判断的内部操作
                        if not bool(gt_match["used"]):
                            # 循环当前class_name的第idx的tp设置 0 -> 1
                            tp[idx] = 1
                            # 将真实数据的key["used"]的值设置为True
                            gt_match["used"] = True
                            # 将当前count_true_positive[class_name]加1
                            count_true_positives[class_name] += 1
                            # 将更改的gt_match重新写入到gt_file文件里
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))

                            if show_animation:
                                status = "匹配成功!"

                        else:
                            # 循环当前class_name的第idx的tp设置 0 -> 1
                            fp[idx] = 1
                            if show_animation:
                                status = "重复匹配!"
                # 如果没有大于最小阀值
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "小于重叠率!"
                # 暂不实现
                if show_animation:
                    pass
            # tp 与 fp 进行累加复制到原位
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            #
            # 把tp每一个数据都除以真实该类别的数量 gt_counter_per_class里面存的就是真实标签中的各类别数量
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            # 求p 正确预测样本/ 所有正样本
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1  = np.array(rec)*np.array(prec)*2 / np.where((np.array(prec)+np.array(rec))==0, 1, (np.array(prec)+np.array(rec)))

            sum_AP  += ap
            text    = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec)>0:
                F1_text         = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                Recall_text     = "{0:.2f}%".format(rec[score_threhold_idx]*100) + " = " + class_name + " Recall "
                Precision_text  = "{0:.2f}%".format(prec[score_threhold_idx]*100) + " = " + class_name + " Precision "
            else:
                F1_text         = "0.00" + " = " + class_name + " F1 "
                Recall_text     = "0.00%" + " = " + class_name + " Recall "
                Precision_text  = "0.00%" + " = " + class_name + " Precision "

            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if len(prec)>0:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=" + "{0:.2f}".format(F1[score_threhold_idx])\
                    + " ; Recall=" + "{0:.2f}%".format(rec[score_threhold_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score_threhold_idx]*100))

            else:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=0.00% ; Recall=0.00% ; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            # 画图 暂不实现
            if draw_plot:
                pass
        if show_animation:
            cv2.destroyAllWindows()

        if n_classes == 0:
            print("未检测到然后种类")
            return 0

        results_file.write("\n# mAP of all classes\n")
        mAP     = sum_AP / n_classes
        text    = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)
    return mAP














































































