# 计算cocomap
import os
#
def get_coco_map(class_names,path):
    GT_PATH = os.path.join(path,"ground-truth")
    DR_PATH = os.path.join(path,"detection-results")
    COCO_PATH = os.path.join(path,'coco_eval')