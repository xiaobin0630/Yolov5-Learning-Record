import os
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
# 创建损失历史类
class LossHistory():
    def __init__(self,log_dir,model,input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        # 创建 文件夹
        os.makedirs(self.log_dir)
        # 实例化SummaryWriter,传入保存的文件路径
        self.writer = SummaryWriter(self.log_dir)
        try:
            dimmy_input = torch.randn(2,3,input_shape[0],input_shape[1])
            self.writer.add_graph(model,dimmy_input)
        except:
            pass

    # 记录loss
    def append_loss(self,epoch,loss,val_loss):
        # 判断存储日志与权重的文件夹是否存在,不存在就创建
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # 存入本次epoch的训练loss与验证loss
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 创建txt文件
        with open(os.path.join(self.log_dir,"epoch_loss.txt"),'a') as f:
            # 写入loss
            f.write(str(loss))
            # 加上换行符
            f.write("\n")
        with open(os.path.join(self.log_dir,"epoch_val_loss.txt"),'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        # 调用方法 添加标量
        self.writer.add_scalar('loss',loss,epoch)
        self.writer.add_scalar('val_loss',val_loss,epoch)
        # 画图
        self.loss_plot()
    def loss_plot(self):
        # 计算有多少个loss
        iters = range(len(self.losses))
        # 创建画布
        plt.figure()
        # 描绘数据plt.plot(x,y)
        plt.plot(iters,self.losses,'red',linewidth = 2,label = 'train loss')
        plt.plot(iters,self.val_loss,'coral',linewidth = 2,label = 'val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters,scipy.signal.savgol_filter(self.losses,num,3),'green',linestyle = '--',linewidth = 2,label='smooth train loss')
            plt.plot(iters,scipy.signal.savgol_filter(self.val_loss,num,3),'#8B4513',linestyle = '--',linewidth = 2,label='smooth val loss')
        except:
            pass

        # 添加各类标签
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc = "upper right")
        # 保存
        plt.savefig(os.path.join(self.log_dir,"epoch_loss.png"))
        plt.cla()

        plt.close("all")




