# 使用数据进行训练
import datetime
import os
import numpy as np
from nets.yolo import YoloBody
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.callbacks import LossHistory,EvalCallback
from nets.yolo_trainning import weights_init,YOLOLoss,ModelEMA,get_lr_scheduler,set_optimizer_lr
from utils.utils import get_classes,download_weights,show_config
from utils.dataloader import YoloDataset,yolo_dataset_collate
from torch.utils.data import DataLoader
from utils.utils_fit import fit_one_epoch
if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    # 是否使用单机多卡分布式运行
    distributed = False
    # 若使用DDP多卡,可用
    sync_bn = False
    # 是否使用混合精度训练
    fp16 = False
    # classes_path 类别路径文件
    classes_path = "model_data/voc_classes"
    # anchors_path 先验框对应的txt文件
    anchors_path = 'model_data/yolo_anchors'
    # anchors_mosk 帮助代码找到对应的先验框
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    # model_path 加载预训练权重
    model_path = 'model_data/yolov5_s.pth'
    # input_shape 传入输入图片shape大小
    input_shape = [640,640]
    # backbone 传入主体特征提取网络backbone
    backbone = 'cspdarknet'
    # pretrained 如果设置model_path 这里就无需加载
    pretrained = False
    # phi 用于选择yolov5的版本
    phi = 's'
    # mosaic 是否使用马赛克数据增强
    mosaic = True
    # 设置进行马赛克数据增强的概率
    mosaic_prob = 0.5
    # 是否使用mixup数据增强,且仅在mosaic=True时有效 但还未实现
    mixup = True
    # 设置进行mixup数据增强的概率
    mixup_prob = 0.5
    # special_aug_ratio 保证图片不脱离自然分布
    special_aug_ratio = 0.7
    # 标签平滑 不使用
    label_smoothing = 0
    # Init_Epoch 模型当前开始的训练迭代数
    Init_Epoch = 0
    # Freeze_Epoch 冻结训练的迭代数 如果设置Freeze_Epoch为50 Init_Epoch为0 则 0-50的训练次数是冻结阶段训练
    Freeze_Epoch = 50
    # Freeze_batch_size 冻结阶段训练的小批量大小值
    Freeze_batch_size = 2
    # UnFreeze_Epoch 为不冻结训练阶段
    UnFreeze_Epoch = 100
    # UnFreeze_batch_size 为非冻结阶段的小批量大小值
    UnFreeze_batch_size = 2
    # Freeze_Train 是否进行冻结训练 默认先冻结训练
    Freeze_Train = True

    # 学习率,优化器,学习率下降等
    # Init_lr 模型莫大且最初的学习率 0.01
    Init_lr = 1e-2
    # 模型的最小下学习率
    Min_lr = Init_lr * 0.01
    # optimizer_type 优化器sgd
    optimizer_type = "sgd"
    # momentum 动量参数
    momentum = 0.937
    # weight_decay 衰减参数,防止过拟合
    weight_decay = 5e-4
    # lr_decay_type 设置学习率下降的方式
    lr_decay_type = "cos"
    # save_period 过多少个epoch保存一次权重
    save_period = 10
    # save_dir 权重与日志保存的文件夹
    save_dir = "logs"
    # eval_flag 是否在训练时进行评估,使用验证集进行评估
    eval_flag = True
    # eval_period 多少个epoch评估一次
    eval_period = 10
    # num_workers 多线程取得数据,电脑内存小就设置小点
    num_workers = 0
    # train_annotation_path 训练图片路径与标签
    train_annotation_path = '2007_train.txt'
    # val_annotation_path 验证图片路径与标签
    val_annotation_path = '2007_val.txt'
    # 设置用到的显卡
    # 查看有几个gpu
    ngpus_per_node = torch.cuda.device_count()
    # 分布式多卡未实现
    if distributed:
        pass
    # 单机单卡
    else:
        # 设置设备 有GPU就用,没有就用cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 方便后续操作设置的一些参数,如下:
        local_rank = 0
        rank = 0
    # 获取类名与数量,先验框大小与数量
    class_names,num_classes = get_classes(classes_path)
    anchors,num_anchors = get_classes(anchors_path)

    # 下栽预处理权重 pretrained = False 跳过
    if pretrained:
        download_weights(backbone,phi)

    # 创建YoloBody模型
    model = YoloBody(anchors_mask,num_classes,phi,backbone,pretrained=pretrained,input_shape=input_shape)
    # 模型参数初始化
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # 打印导入的weights
        if local_rank == 0:
            print("导入的权重{}.".format(model_path))

        # 更具预训练权重的key与模型的key进行加载
        # 模型序列化
        model_dict = model.state_dict()

        # 导入预训练模型
        pretrained_dict = torch.load(model_path,map_location = device)

        load_key,no_load_key,temp_dict = [],[],{}
        # 循提取预训练模型每层key与value
        for k,v in pretrained_dict.items():
            # 如果预训练的k在模型序列化里的keys里面,且预训练的k的value的shape 与模型的value的shape相等,则将预训练的k与v存入
            # temp_dict字典中 将k也导入load_key列表中
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            # 如果预训练没有被匹配到的k放入no_load_key列表中
            else:
                no_load_key.append(k)
        # 更新模型序列化
        model_dict.update(temp_dict)
        # 模型导入更新的模型序列化
        model.load_state_dict(model_dict)
        # 打印导入与没导入的键与数量
        if local_rank == 0:
            print("\n成功导入的键:",str(load_key)[:500],"成功导入的键数:",len(load_key))
            print("\n失败导入的键:", str(no_load_key)[:500], "失败导入的键数:", len(no_load_key))
    # 获取损失函数
    yolo_loss = YOLOLoss(anchors,num_classes,input_shape,Cuda,anchors_mask,label_smoothing)

    # 记录Loss
    if local_rank == 0:
        # 生成时间字符串 如:2022_11_22_13_36_55
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        # 生成log_dir文件夹路径
        log_dir = os.path.join(save_dir,"loss_" + str(time_str))
        # 实例化LossHistory为后面画图做准备
        loss_history = LossHistory(log_dir,model,input_shape=input_shape)
    else:
        loss_history = None

    # 判断是否使用混合精度训练,不用
    if fp16:
        pass
    else:
        scaler = None

    # 启动训练模式
    model_train = model.train()

    # 多卡同步bn 分布式多卡未实现
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("单卡与非分布式不支持多卡同步.")

    if Cuda:
        if distributed:# 分布式多卡未实现
            pass
        else:
            #
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            # 将模型放在cuda上
            model_train = model_train.cuda()

    # 权值平滑实例化 提升模型性能
    ema = ModelEMA(model_train)
    # 读取对应的数据集的txt
    # 打开文件
    with open(train_annotation_path,encoding='utf-8') as f:
        # 按行读取
        train_lines = f.readlines()
    with open(val_annotation_path,encoding='utf-8') as f:
        val_lines = f.readlines()
    # 记训练与验证集的数
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        pass
        # 查看所有配置
        # show_config(
        #     classes_path = classes_path,anchors_path = anchors_path,anchors_mask = anchors_mask,model_path = model_path,
        #     input_shape = input_shape,Init_Epoch = Init_Epoch,Freeze_Epoch = Freeze_Epoch,UnFreeze_Epoch = UnFreeze_Epoch,
        #     Freeze_batch_size = Freeze_batch_size,UnFreeze_batch_size = UnFreeze_batch_size,Freeze_Train=Freeze_Train,
        #     Init_lr = Init_lr,Min_lr = Min_lr,optimizer_type = optimizer_type,momentum = momentum,lr_decay_type = lr_decay_type,
        #     save_period = save_period,save_dir = save_dir,num_workers = num_workers,num_train = num_train,num_val = num_val
        # )

        # wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        # total_step = num_train // UnFreeze_batch_size * UnFreeze_Epoch
        # print(total_step)

    # 训练
    if True:
        UnFreeze_flag = False
        # 判断是否冻结训练
        if Freeze_Train:
            # 循环设置参数不计算梯度
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 如果冻结训练的话 batch_size设置为 Freeze_batch_size 否则设置 Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else UnFreeze_batch_size

        # 通过batch_size,自适应调整学习率
        nbs = 64
        # 学习率最大值与最小值与优化器种类有关
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr,lr_limit_min),lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr,lr_limit_min * 1e-2),lr_limit_max * 1e-2)

        # 根据optimizer_type选择优化器
        # 设置三个空列表
        pg0,pg1,pg2 = [],[],[]
        for k,v in model.named_modules():
            # 如果v对象有bias属性 存入pg2
            if hasattr(v,"bias") and isinstance(v.bias,nn.Parameter):
                pg2.append(v.bias)
            # 如果v对象有nn.BatchNorm2d 存入pg0
            if isinstance(v,nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            # 如果v对象有weight属性 存入pg1
            elif hasattr(v,"weight") and isinstance(v.weight,nn.Parameter):
                pg1.append(v.weight)
        # 设置两个优化器字典通过optimizer_type提取优化器
        # bn层的优化器
        optimizer = {
            'adam' : optim.Adam(pg0,Init_lr_fit,betas = (momentum,0.999)),
            'sgd' : optim.SGD(pg0,Init_lr_fit,momentum = momentum,nesterov=True)
        }[optimizer_type]
        # 对weight的优化器
        optimizer.add_param_group({"params" : pg1,"weight_decay" : weight_decay})
        # 对偏置的优化器
        optimizer.add_param_group({"params" : pg2})

        # 获取学习率下降的公式 学习率衰减之余弦退火
        lr_scheduler_func = get_lr_scheduler(lr_decay_type,Init_lr_fit,Min_lr_fit,UnFreeze_Epoch)

        # 计算训练集与验证集的每个epoch的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        # 判断训练集与测试集的epoch长度不可小于等于0
        if epoch_step == 0 and epoch_step_val == 0:
            raise ValueError("数据集过小,无法进行训练,请扩充。")

        # 权值平滑
        if ema:
            ema.updates = epoch_step * Init_Epoch

        # 构建数据集加载器
        train_dataset = YoloDataset(train_lines,input_shape,num_classes,anchors,anchors_mask,epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic,mixup=mixup,mosaic_prob=mosaic_prob,mixup_prob=mixup_prob,train=True,
                                    special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines,input_shape,num_classes,anchors,anchors_mask,epoch_length=UnFreeze_Epoch,
                                  mosaic=False,mixup=False,mosaic_prob=0,mixup_prob=0,train=False,special_aug_ratio=0)
        # 没用到分布式多卡
        if distributed:
            pass
        else:
            train_sample = None
            val_sample = None
            shuffle = True
        # 训练数据集加载器
        gen = DataLoader(train_dataset,shuffle = shuffle,batch_size = batch_size,num_workers = num_workers,pin_memory = True,
                         drop_last=True,collate_fn=yolo_dataset_collate,sampler=train_sample)
        # 验证
        gen_val = DataLoader(val_dataset,shuffle = shuffle,batch_size=batch_size,num_workers=num_workers,pin_memory=True,
                             drop_last=True,collate_fn=yolo_dataset_collate,sampler=val_sample)

        # 记录eval的map曲线
        if local_rank == 0:
            # 实例化EvalCallback
            eval_callback = EvalCallback(model,input_shape,anchors,anchors_mask,class_names,num_classes,val_lines,log_dir,
                                         Cuda,eval_flag=eval_flag,period=eval_period)
        else:
            eval_period = None

        # 开始模型训练
        for epoch in range(Init_Epoch,UnFreeze_Epoch):
            # 如果模型有冻结学习部分
            # 则解冻,并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Epoch:
                # 设置解冻阶段的batch_size
                batch_size = UnFreeze_batch_size

                # 判断当前batch_size,自自适应调整学习率
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_max = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr,lr_limit_min),lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr,lr_limit_min * 1e-2),lr_limit_max * 1e-2)

                # 获取学习率下降公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type,Init_lr_fit,Min_lr_fit,UnFreeze_Epoch)

                # 将主干网络参数设置为要计算梯度
                for param in model.backbone.parameters():
                    param.requires_grad = True

                # 计算解冻阶段训练与验证的一个epoch的小批量步长数
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                # 判断数据步长数,如果等于0,代表数据量小无法训练
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小,无法进行训练")
                # 权值平滑
                if ema:
                    ema.updates = epoch_step * epoch
                # 未实现分布式
                if distributed:
                    pass
                gen = DataLoader(train_dataset,shuffle = shuffle,batch_size = batch_size,num_workers = num_workers,pin_memory=True,
                                 drop_last=True,collate_fn=yolo_dataset_collate,sampler=train_sample)
                gen_val = DataLoader(val_dataset,shuffle = shuffle,batch_size = batch_size,num_workers = num_workers,pin_memory=True,
                                     drop_last=True,collate_fn=yolo_dataset_collate,sampler=val_sample)

                UnFreeze_Epoch = True
            # 设置当前的epoch
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            # 未实现分布式
            if distributed:
                pass
            # 设置学习率
            set_optimizer_lr(optimizer,lr_scheduler_func,epoch)

            # 一次epoch 训练
            fit_one_epoch(model_train,model,ema,yolo_loss,loss_history,eval_callback,optimizer,epoch,
                          epoch_step,epoch_step_val,gen,gen_val,UnFreeze_Epoch,Cuda,fp16,scaler,save_period,save_dir,local_rank)
        # 关闭loss_history
        if local_rank == 0:
            loss_history.writer.close()





















































