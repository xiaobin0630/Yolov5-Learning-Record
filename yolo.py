from utils.utils import get_classes,get_anchors

# 用于预测的类
class YOLO(object):
    # 设置默认值字典配置
    _defaults = {
        # 模型权重
        "model_path" : "model_data/yolov5_s.pth",
        # 类别文件
        "classes_path" : "model_data/coco_classes.txt",
        # 先验框大小文件
        "anchors_path" : 'model_data/yolo_anchors.txt',
        # 先验框掩码
        "anchors_mask" : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # 输入图片大小
        "input_shape" : [640,640],
        # 主干网络
        "backbone" : 'cspdarknet',
        # yolov5版本
        "phi" : "s",
        # 置信度阀值
        "confidence" : 0.5,
        # 非极大值阀值
        "nms_iou" : 0.3,
        # 是否进行不失真的图像缩放
        "letterbox_image" : True,
        # 是否使用GPU
        "cuda" : True,
    }

    # 得到默认值
    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "没有这个属性名'" + n + "'."
    # 初始化
    def __init__(self,**kwargs): # 如:传入**{'a': 1, 'b': 2, 'c': 3}
        # 将上面默认设置更新为类的字典
        self.__dict__.update(self._defaults)
        # 循环提取实例化对象参数
        for name,value in kwargs.items():
            # setattr 给对象设置key value
            setattr(self,name,value)
            self._defaults[name] = value
        # print(self.__dict__)
        # self.__dict__ {'model_path': 'model_data/yolov5_s.pth', 'classes_path': 'model_data/coco_classes.txt', 'anchors_path': 'model_data/yolo_anchors.txt', 'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]], 'input_shape': [640, 640], 'backbone': 'cspdarknet', 'phi': 's', 'confidence': 0.5, 'nms_iou': 0.3, 'letterbox_image': True, 'cuda': True, 'a': 1, 'b': 2, 'c': 3}
        # 获取种类和先验框数量
        self.class_names,self.num_classes = get_classes(self.classes_path)
        self.anchors,self.num_anchors = get_anchors(self.anchors_path)




if __name__ == '__main__':
    yolo = YOLO()



































