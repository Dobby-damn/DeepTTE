import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    """轨迹特征提取网络，用于处理时空轨迹数据"""
    def __init__(self, kernel_size=5, num_filter=64):
        """
        Args:
            kernel_size (int): 卷积核大小（控制局部感受野）
            num_filter (int): 卷积通道数（特征图数量）
        """
        super(Net, self).__init__()

        # 网络参数
        self.kernel_size = kernel_size  # 卷积核大小（如5）
        self.num_filter = num_filter    # 卷积过滤器数量（如64）
        
        # 输入 ex, ey 共2个通道
        self.coord_fc = nn.Linear(2, 16)
        self.conv1 = nn.Conv1d(16, num_filter, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)  # 输出全局特征

    def forward(self, traj):
        """
        traj: dict
          {
            'ex': tensor(B, T),
            'ey': tensor(B, T)
          }
        """
        # 拼接 (ex, ey)
        ex = traj['ex'].unsqueeze(2)
        ey = traj['ey'].unsqueeze(2)
        coords = torch.cat((ex, ey), dim=2)  # (B, T, 2)

        # 投影到16维
        coords = F.relu(self.coord_fc(coords))  # (B, T, 16)
        coords = coords.permute(0, 2, 1)        # (B, 16, T)

        # 时序卷积
        conv_out = F.elu(self.conv1(coords))    # (B, num_filter, T')
        pooled = self.pool(conv_out).squeeze(2) # (B, num_filter)

        return pooled

