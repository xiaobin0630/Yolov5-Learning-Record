from yolo import YOLO
from PIL import Image
if __name__ == "__main__":
    # 实例化
    yolo = YOLO()
    # 预测模式
    mode = "predict"
    # 是否保存预测框
    crop = False
    # 是否计数
    count = False
    if mode == "predict":
        while True:
            img = input("输入图像路径文件名")
            try:
                image = Image.open(img)
            except:
                print("无法打开,请重试！")
                continue
            else:
                r_image = yolo.detect_image(image,crop=crop,count=count)
                r_image.show()


