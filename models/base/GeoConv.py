import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    TrajConv：保留时序维度，提取局部几何特征
    """
    def __init__(self, input_dim=4, kernel_size=3, num_filter=32):
        """
        Args:
            input_dim (int): 输入维度 (ex, ey, speed, acc) = 4. 如果你在 dataloader 加入了 timestamp，这里就是 3
            kernel_size (int): 卷积核大小
            num_filter (int): 输出通道数
        """
        super(Net, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter

        # 1. 初始特征映射 (Point-wise processing)
        # 将坐标点映射到更高维空间
        self.input_proj = nn.Conv1d(input_dim, 16, kernel_size=1) 

        # 2. 局部特征提取 (Local Feature Extraction)
        # 使用 padding='same' (PyTorch中通过 padding=kernel_size//2 实现) 保持序列长度不变
        self.conv1 = nn.Conv1d(16, num_filter, kernel_size, padding=kernel_size//2)
        # self.conv2 = nn.Conv1d(num_filter, num_filter, kernel_size, padding=kernel_size//2)
        
        # 激活函数
        self.act = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm1d(num_filter)
        self.bn2 = nn.BatchNorm1d(num_filter)

    def forward(self, traj):
        """
        Args:
            traj: dict containing 'ex' (B, T), 'ey' (B, T)
        Returns:
            tensor (B, T, num_filter)  <-- 注意这里保留了 T 维度
        """
        # 1. 数据准备
        ex = traj['ex'].unsqueeze(1) # (B, 1, T)
        ey = traj['ey'].unsqueeze(1) # (B, 1, T)
        sp = traj['speed'].unsqueeze(1) # 新增
        ac = traj['acc'].unsqueeze(1)   # 新增
        
        # 建议：如果 dataloader 里还没有计算速度，可以在这里简单计算一阶差分作为额外特征
        # 但为了保持兼容性，目前先只用坐标
        inputs = torch.cat((ex, ey, sp, ac), dim=1) # (B, 4, T)
        

        # 2. 网络前向传播
        x = self.input_proj(inputs)      # (B, 16, T)
        
        # 第一层卷积 + BN + 激活
        x = self.conv1(x)                # (B, num_filter, T)
        x = self.bn1(x)
        x = self.act(x)
        
        # 第二层卷积 (加深网络以提取更复杂的局部震颤特征)
        # residual = x
        # x = self.conv2(x)                # (B, num_filter, T)
        # x = self.bn2(x)
        # x += residual                    # 残差连接，防止梯度消失
        # x = self.act(x)

        # 3. 维度调整
        # Conv1d 输出是 (B, C, T)，LSTM 通常需要 (B, T, C)
        x = x.permute(0, 2, 1)           # (B, T, num_filter)
        
        return x
