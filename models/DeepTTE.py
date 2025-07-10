import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# 将当前目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent))
import utils
import base

import numpy as np

from torch.autograd import Variable

EPS = 10  # 用于防止除零的小常数

class EntireEstimator(nn.Module):
    """全局估计器：预测整个轨迹的到达时间（总时间）"""
    def __init__(self, input_size, num_final_fcs, hidden_size = 128):
        super(EntireEstimator, self).__init__()

        # 输入层 -> 隐藏层
        self.input2hid = nn.Linear(input_size, hidden_size)

        # 残差连接的全连接层（增强特征提取能力）
        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        # 隐藏层 -> 输出层（单值预测）
        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        """前向传播
        Args:
            attr_t: 属性特征 [batch_size, attr_size]
            sptm_t: 时空特征 [batch_size, spatio_temporal_size]
        Returns:
            预测的总时间 [batch_size, 1]
        """
        inputs = torch.cat((attr_t, sptm_t), dim = 1)   # 拼接属性特征和时空特征

        hidden = F.leaky_relu(self.input2hid(inputs))   # LeakyReLU激活

        # 残差连接结构
        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual  # 残差连接

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std):
        """评估批次数据（反归一化 + 计算MAPE损失）
        Args:
            pred: 模型预测值（归一化后）
            label: 真实标签（归一化后）
            mean/std: 归一化时的均值和标准差
        Returns:
            pred_dict: 包含反归一化后的预测值和标签的字典
            loss: 平均绝对百分比误差（MAPE）
        """
        label = label.view(-1, 1)
        # 反归一化
        label = label * std + mean
        pred = pred * std + mean
        # 计算MAPE (|pred-truth|/truth)
        loss = torch.abs(pred - label) / label

        return {'label': label, 'pred': pred}, loss.mean()

class LocalEstimator(nn.Module):
    """局部估计器：预测轨迹片段的行驶时间（子路径时间）"""
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()
        # 简单的三层全连接网络

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        """前向传播
        Args:
            sptm_s: 时空序列特征 [total_seq_len, feature_size]
                   (total_seq_len是所有轨迹片段的总长度)
        """
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        """评估批次数据（处理变长序列 + 计算加权MAPE）
        Args:
            pred: 模型预测值 [total_seq_len, 1]
            lens: 每个轨迹的实际长度列表
            label: 真实标签（已填充的变长序列）
            mean/std: 归一化参数
        """
        # 移除填充部分（转换为PackedSequence）
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first = True)[0]
        label = label.view(-1, 1)
        # 反归一化
        label = label * std + mean
        pred = pred * std + mean
        # 计算加权MAPE（分母加EPS防止除零）
        loss = torch.abs(pred - label) / (label + EPS)

        return loss.mean()


class Net(nn.Module):
    """主网络：整合属性、时空特征，实现多任务学习（全局+局部时间预测）"""
    def __init__(self, kernel_size = 3, num_filter = 32, pooling_method = 'attention', num_final_fcs = 3, final_fc_size = 128, alpha = 0.3):
        super(Net, self).__init__()

        # parameter of attribute / spatio-temporal component
        # 时空卷积参数
        self.kernel_size = kernel_size  # 卷积核大小（局部路径长度）
        self.num_filter = num_filter    # 卷积通道数
        self.pooling_method = pooling_method  # 池化方法（attention/mean）

        # parameter of multi-task learning component
        # 多任务学习参数
        self.num_final_fcs = num_final_fcs  # 全局估计器的残差层数
        self.final_fc_size = final_fc_size  # 全局估计器的隐藏层大小
        self.alpha = alpha  # 局部损失权重系数（全局损失权重=1-alpha）

        self.build()  # 构建子模块
        self.init_weight()  # 初始化权重

    def init_weight(self):
        """权重初始化：偏置置零，权重Xavier均匀初始化"""
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def build(self):
        """构建子网络"""
        # attribute component
        # 属性特征提取网络（假设base.Attr.Net已定义）
        self.attr_net = base.Attr.Net()

        # spatio-temporal component
        # 时空特征提取网络（CNN+RNN+Pooling）
        self.spatio_temporal = base.SpatioTemporal.Net(attr_size = self.attr_net.out_size(), \
                                                       kernel_size = self.kernel_size, \
                                                       num_filter = self.num_filter, \
                                                       pooling_method = self.pooling_method
        )
        # 全局和局部估计器
        self.entire_estimate = EntireEstimator(input_size =  self.spatio_temporal.out_size() + self.attr_net.out_size(), num_final_fcs = self.num_final_fcs, hidden_size = self.final_fc_size)

        self.local_estimate = LocalEstimator(input_size = self.spatio_temporal.out_size())


    def forward(self, attr, traj, config):
        """前向传播
        Args:
            attr: 属性字典（如司机ID、天气等）
            traj: 轨迹数据（坐标序列）
            config: 配置文件参数
        Returns:
            训练模式: (全局预测结果, (局部预测结果, 各轨迹长度))
            测试模式: 全局预测结果
        """
        attr_t = self.attr_net(attr)  # 提取属性特征 [B, attr_size]

        # 提取时空特征
        # sptm_s: 序列特征（用于局部估计） [total_seq_len, feature_size]
        # sptm_l: 各轨迹实际长度列表
        # sptm_t: 池化后的聚合特征 [B, feature_size]
        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int); sptm_t: merged tensor after attention/mean pooling
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)

        # 全局时间预测
        entire_out = self.entire_estimate(attr_t, sptm_t)

        # 训练时额外返回局部预测
        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def eval_on_batch(self, attr, traj, config):
        """评估批次数据（计算多任务损失）
        Args:
            attr: 包含'time'字段（全局时间标签）
            traj: 包含'time_gap'字段（局部时间间隔标签）
        Returns:
            pred_dict: 全局预测结果（反归一化后）
            loss: 加权多任务损失（全局+局部）
        """
        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)
        print( 'Geo输出：', entire_out.shape)
        # 计算全局损失（MAPE）
        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['time'], config['time_mean'], config['time_std'])

        if self.training:
            # 计算局部损失（处理变长序列）
            # get the mean/std of each local path
            mean, std = (self.kernel_size - 1) * config['time_gap_mean'], (self.kernel_size - 1) * config['time_gap_std']

            # get ground truth of each local path
            local_label = utils.get_local_seq(traj['time_gap'], self.kernel_size, mean, std)
            local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)

            # 加权多任务损失
            return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss
        else:
            return pred_dict, entire_loss
