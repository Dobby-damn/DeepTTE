"""
该代码系8年前代码，极有可能基于python2开发，需考虑兼容
"""
import sys
from pathlib import Path

# 将当前目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent))
import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str, help = 'task, train or test') 
parser.add_argument('--batch_size', type = int, default = 64, help = 'the batch_size to train, default 400')
parser.add_argument('--epochs', type = int, default = 100, help = 'the epoch to train, default 100')

# evaluation args
parser.add_argument('--weight_file', type = str, help = 'the path of model weight')
parser.add_argument('--result_file', type = str, help = 'the path to save the result')

# cnn args
parser.add_argument('--kernel_size', type = int, help = 'the kernel size of Geo-Conv, only used when the model contains the Geo-conv part')

# rnn args attention池化模型可以自动关注轨迹中的拥堵路段或停留点，提升预测精度，mean池化模型序列中各时间步重要性均匀（可能忽略关键拥堵点）
parser.add_argument('--pooling_method', type = str, help = 'attention/mean') 

# multi-task args
parser.add_argument('--alpha', type = float, help='组合在多任务学习中的权重')

# log file name
parser.add_argument('--log_file', type = str, help='日志文件的路径')

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))

def train(model, elogger, train_set, eval_set):
    """训练模型的主函数。
    Args:
        model: 待训练的模型
        elogger: 日志记录器
        train_set: 训练集文件列表
        eval_set: 验证集文件列表
    """
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    # 设置模型为训练模式
    model.train()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # 使用 Adam 优化器 一种广泛使用的深度学习优化算法 旨在通过计算梯度的一阶矩估计和二阶矩估计来调整每个参数的学习率，从而实现更高效的网络训练。
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # 遍历每个 epoch
    for epoch in range(args.epochs):
        print ('Training on epoch {}'.format(epoch))
        # 遍历训练集中的每个文件
        for input_file in train_set:
            print ('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj获取数据加载器（返回属性 attr 和轨迹 traj）批处理
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0   # 累计损失

            # 遍历每个批次
            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable  将数据转换为 PyTorch Variable（兼容旧版本）
                # attr, traj = utils.to_var(attr), utils.to_var(traj)
                if use_cuda:
                    attr = {k: v.cuda() if torch.is_tensor(v) else v for k, v in attr.items()}
                    traj = {k: v.cuda() if torch.is_tensor(v) else v for k, v in traj.items()}
                if torch.__version__ < '0.4':
                    attr = {k: Variable(v) for k, v in attr.items()}
                    traj = {k: Variable(v) for k, v in traj.items()}

                # 前向传播并计算损失
                '''此处应是前向传播forward，_, loss = model.eval_on_batch(attr, traj, config)'''
                
                # print('traj:', traj['lens'])
                _, loss = model.eval_on_batch(attr, traj, config)
                '''traj: tensor([125, 122, 114, 111, 110, 108, 107, 104, 104, 103], device='cuda:0')
                Geo输出： torch.Size([10, 123, 33])
                属性张良形状 torch.Size([10, 1, 28])
                扩展属性张量形状 torch.Size([10, 123, 28])
                拼接后的卷积张量形状 torch.Size([10, 123, 61])
                '''
                

                # update the model
                # 反向传播和优化
                optimizer.zero_grad()
                '''loss.backward未定义'''
                loss.backward()
                optimizer.step()

                # 更新累计损失（注意：loss.data[0] 是旧写法，新版本应为 loss.item()）
                # 损失值获取兼容性
                if torch.__version__ < '0.4':
                    running_loss += loss.data[0]
                else:
                    running_loss += loss.item()

                # 打印训练进度
                print ('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0))),
            print
            # 记录当前文件的训练损失
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))

        # evaluate the model after each epoch 每个 epoch 结束后在验证集上评估
        evaluate(model, elogger, eval_set, save_result = True)

        # save the weight file after each epoch 保存模型权重（文件名包含时间戳）
        weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
        elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), './saved_weights/' + weight_name)

def write_result(fs, pred_dict, attr):
    """将预测结果写入文件。
    Args:
        fs: 文件句柄
        pred_dict: 包含预测值和标签的字典
        attr: 属性数据（如 dateID, timeID 等）
    """
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    """评估模型性能。
    Args:
        model: 待评估的模型
        elogger: 日志记录器
        files: 评估文件列表
        save_result: 是否保存预测结果
    """
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result: write_result(fs, pred_dict, attr)

            # running_loss += loss.data[0] #  旧版写法，已经失效
            if torch.__version__ < '0.4':
                running_loss += loss.data[0]
            else:
                running_loss += loss.item()

        print ('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    if save_result: fs.close()

def get_kwargs(model_class):
    """从命令行参数中提取模型需要的参数。
    Args:
        model_class: 模型类（如 DeepTTE.Net）
    Returns:
        过滤后的参数字典
    """
    
    model_args = list(inspect.signature(model_class.__init__).parameters.keys())

    # model_args = inspect.getargspec(model_class.__init__).args 已弃用
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        # 在测试集上评估并保存结果
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
   run()
