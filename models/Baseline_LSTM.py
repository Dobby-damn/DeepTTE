import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .base import Attr  # 复用你现有的属性处理模块

class Net(nn.Module):
    """
    基线模型 1：Vanilla BiLSTM + Simple Concat Fusion
    直接将 4 通道轨迹输入 BiLSTM，取最后时刻状态与属性特征拼接。
    """
    def __init__(self, 
                 num_classes=2, 
                 traj_input_dim=4,  # ex, ey, speed, acc
                 hidden_size=48, 
                 dropout_p=0.5):
        super(Net, self).__init__()

        # 1. 属性模块 (复用现有代码，保证公平)
        self.attr_net = Attr.Net()
        self.attr_dim = self.attr_net.out_size()

        # 2. 轨迹时序模块 (纯 BiLSTM，无 CNN)
        self.lstm = nn.LSTM(input_size=traj_input_dim, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.traj_dim = hidden_size * 2  # 双向

        # 3. 分类器 (简单拼接，无门控)
        fusion_dim = self.attr_dim + self.traj_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )

    def forward(self, attr, traj, config=None):
        # --- 1. 处理静态属性 ---
        attr_vec = self.attr_net(attr)  # (B, D_attr)

        # --- 2. 处理动态轨迹 ---
        # 组合 4 个通道: (B, T, 4)
        x = torch.stack([traj['ex'], traj['ey'], traj['speed'], traj['acc']], dim=-1)
        
        # 获取真实长度用于 Pack
        mask = traj['mask']
        lengths = mask.sum(dim=1).long().cpu()
        
        # 处理变长序列
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_x)
        
        # 取 BiLSTM 最后时刻的隐藏状态 (前向的最后一个 + 后向的第一个)
        # hn shape: (num_layers * num_directions, batch, hidden_size) = (2, B, H)
        lstm_out = torch.cat([hn[0], hn[1]], dim=-1)  # (B, 2*H)

        # --- 3. 简单拼接与分类 ---
        feat_fused = torch.cat([attr_vec, lstm_out], dim=1)
        logits = self.classifier(feat_fused)
        
        return logits