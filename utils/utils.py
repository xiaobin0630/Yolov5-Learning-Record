import numpy as np
from PIL import Image

# 获得类与类的数量
def get_classes(classes_path):
    with open(classes_path,encoding="utf-8") as f:
        class_name = f.readlines()
        classes_name = [c.strip() for c in class_name]
    return classes_name,len(classes_name)

# 将图片转化为RGB图片
def cvtColor(image):
    # 如何image的shape的维度数是3且第2维的通通道数是3则是RGB图片
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        # 不是RGB图片,就转化为RGB图片
        image = image.convert('RGB')
        return image









