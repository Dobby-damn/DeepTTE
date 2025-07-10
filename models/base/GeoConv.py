import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    """轨迹特征提取网络，用于处理时空轨迹数据"""
    def __init__(self, kernel_size, num_filter):
        """
        Args:
            kernel_size (int): 卷积核大小（控制局部感受野）
            num_filter (int): 卷积通道数（特征图数量）
        """
        super(Net, self).__init__()

        # 网络参数
        self.kernel_size = kernel_size  # 卷积核大小（如3）
        self.num_filter = num_filter    # 卷积过滤器数量（如32）
        
        self.build()  # 构建网络层

    def build(self):
        """构建网络层结构"""
        # 状态嵌入层：将离散状态码映射为2维向量
        # 输入：状态码（0/1），输出：2维嵌入向量
        self.state_em = nn.Embedding(2, 2)  # 假设只有两种状态
        
        # 坐标处理层：将4维输入（经度+纬度+状态嵌入）映射到16维空间
        self.process_coords = nn.Linear(4, 16)
        
        # 1D卷积层：从16维特征提取局部时空模式
        # 输入通道：16，输出通道：num_filter，卷积核大小：kernel_size
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        """
        前向传播过程
        Args:
            traj (dict): 轨迹数据字典，包含：
                - lngs: 经度序列 [batch_size, seq_len]
                - lats: 纬度序列 [batch_size, seq_len]
                - states: 状态码序列 [batch_size, seq_len]
                - dist_gap: 相邻点距离间隔 [batch_size, seq_len-1]
            config (dict): 配置参数，包含：
                - dist_gap_mean: 距离间隔均值（用于归一化）
                - dist_gap_std: 距离间隔标准差
        Returns:
            conv_locs: 融合后的轨迹特征 [batch_size, seq_len-kernel_size+1, num_filter+1]
        """
        # 1.准备基础数据
        lngs = torch.unsqueeze(traj['lngs'], dim=2)  # [B, T, 1]
        lats = torch.unsqueeze(traj['lats'], dim=2)   # [B, T, 1]

        states = self.state_em(traj['states'].long())  # [B, T, 2]
        
        # 3. 拼接基础特征（经度+纬度+状态嵌入）
        locs = torch.cat((lngs, lats, states), dim=2)  # [B, T, 4]

        # 4. 坐标特征映射（4维->16维）
        # map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))

        # 5. 调整维度用于卷积（1D卷积需要在通道维度处理）
        locs = locs.permute(0, 2, 1)

        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'], config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim = 2)

        conv_locs = torch.cat((conv_locs, local_dist), dim = 2)

        return conv_locs

