import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train,model,ema,yolo_loss,loss_history,eval_callback,optimizer,epoch,epoch_step,
                  epoch_step_val,gen,gen_val,Epoch,cuda,fp16,scaler,save_period,save_dir,local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print("Start Train")
        # 配置进度条 total 每个epoch迭代的次数,desc 添加的信息 mininterval更新进度条的间隔时间
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    # 启动训练模式
    model_train.train()
    # 循环提取小批量数据进行训练
    for iteration,batch in enumerate(gen):
        # 如果迭代书大于了一次epoch的批量数,结束本次epoch训练
        if iteration >= epoch_step:
            break
        # 提取预处理好了的图片数据,目标框数据,与网络应该训练成的模板
        images,targets,y_trues = batch[0],batch[1],batch[2]
        # 不计算张量梯度
        with torch.no_grad():
            # 判断是否使用GPU加速
            if cuda:
                # 将批量数据放入GPU上,进行加速
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]
        # 清除梯度,防止梯度残留
        optimizer.zero_grad()
        if not fp16:
            # 前向传播
            outputs = model_train(images)

            loss_value_all = 0
            # 逐层计算损失

            for l in range(len(outputs)):
                loss_item = yolo_loss(l,outputs[l],targets,y_trues[l])
                # 累加每层损失
                loss_value_all += loss_item

            loss_value = loss_value_all
            # 反向传播计算梯度
            loss_value.backward()
            # 更新参数
            optimizer.step()
        # 判断是否进行权值平滑
        if ema:
            ema.update(model_train)

        # 汇总损失
        loss += loss_value.item()

        # 判断local_rank 是否等于0
        if local_rank == 0:
            # 给进度条添加loss与学习率的信息
            pbar.set_postfix(**{'loss' : loss / (iteration + 1),'lr' : get_lr(optimizer)})
            # 小批量数据完了更新进度条
            pbar.update(1)
    if local_rank == 0:
        # 关闭训练的进度条
        pbar.close()
        print('一次epoch训练完成')
        print('验证阶段开始')
        # 设置验证进度条配置
        pbar = tqdm(total=epoch_step_val,desc=f'Epoch {epoch + 1} /{Epoch}',postfix=dict,mininterval=0.3)
    # 进行ema
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    # 循环提取验证集数据进行验证模型性能
    for iteration,batch in enumerate(gen_val):
        # 如果迭代数大于了一次验证的批量数,结束本次验证
        if iteration >= epoch_step_val:
            break
        # 提取对应的图片,框与网络最后应该预测的模板
        images,targets,y_trues = batch[0],batch[1],batch[2]
        # 不计算梯度
        with torch.no_grad():
            # 判断是否使用GPU
            if cuda:
                # 将数据放在GPU上加速
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]

            # 清除残余梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model_train_eval(images)

            loss_value_all = 0
            # 计算验证阶段的损失
            for l in range(len(outputs)):
                # 逐层计算loss,并累加loss
                loss_item = yolo_loss(l,outputs[l],targets,y_trues[l])
                loss_value_all += loss_item
            loss_value = loss_value_all
        val_loss += loss_value.item()
        # 判断local_rank 是否等于0
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss' : val_loss / (iteration + 1)})
            pbar.update(1)
    # 关闭验证进度条,保存loss,并保存本次epoch的训练与验证loss
    if local_rank == 0:
        pbar.close()
        print('验证阶段完成')
        loss_history.append_loss(epoch + 1,loss / epoch_step,val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1,model_train_eval)
        # 打印当前是第几轮epoch
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        # 打印当前轮的训练损失与验证损失
        print('Total Loss:%.3f || Val Loss: %.3f' % (loss / epoch_step,val_loss / epoch_step_val))

        # 保存权值
        # 生成模型有序字典
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        # 保存模型权重
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict,os.path.join(save_dir,"ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1,loss / epoch_step,val_loss / epoch_step_val)))
        # 保存最好的一个权重值
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print("保存最好的模型到 best_epoch_weights.pth")
            torch.save(save_state_dict,os.path.join(save_dir,"best_epoch_weights.pth"))
        # 保存上一个epoch的权重
        torch.save(save_state_dict,os.path.join(save_dir,"last_epoch_weights.pth"))





































