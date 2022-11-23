import numpy as np
from PIL import Image

# 获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
# 显示所有配置
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' %('keys','values'))
    print('-' * 70)
    for key,value in kwargs.items():
        print('|%25s | %40s|' % (str(key),str(value)))
    print('-' * 70)


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

# 对图像的像素点都除以255
def preprocess_input(image):
    image /= 255.0
    return image

def download_weights(backbone,phi,model_dir="./model_data"):
    import os
    # hub 下载模型序列化权重
    from torch.hub import load_state_dict_from_url
    if backbone == "cspdarknet":
        # 合并 backbone 与 phi
        backbone = backbone + "_" + phi
    # urls
    download_urls = {
        "cspdarknet_s": 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_s_backbone.pth',
        'cspdarknet_m': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_m_backbone.pth',
        'cspdarknet_l': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_l_backbone.pth',
        'cspdarknet_x': 'https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/cspdarknet_x_backbone.pth',
    }
    url = download_urls[backbone]

    # 判断model_dir 路径存在不,不存在就创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 将预训练权重下载对应的路径中
    load_state_dict_from_url(url,model_dir)


# 对输入图像进行resize
def resize_image(image,size,letterbox_image):
    # 获取图像的宽高
    iw,ih = image.size
    # 获取输入形状的宽高
    w,h = size
    # 判断如果为真,就进行添加灰条的不失真的resize,否则就直接进行resize
    if letterbox_image:
        # 求出最小输入形状与图像大小的比列
        scale = min(w/iw,h/ih)
        nw = int(iw*scale) # 求出宽
        nh = int(ih*scale) # 求出高
        # 将图像resize为(nw,nh)
        image = image.resize((nw,nh),Image.BICUBIC)
        # 创建一张大小为size的RGB图像
        new_image = Image.new('RGB',size,(128,128,128))
        # resize图片黏贴在new_image的((w-nw)//2,(h-nh)//2)位置上
        new_image.paste(image,((w-nw)//2,(h-nh)//2))
    else:
        new_image = image.resize((w,h),Image.BICUBIC)
    return new_image
# 图像归一化
def preprocess_input(image):
    image /= 255.0
    return image
















