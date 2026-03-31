import torch
import torch.nn as nn
from .base import Attr

class SeparableConv1d(nn.Module):
    """LITEMV 的核心：深度可分离一维卷积，极大减少参数量，防过拟合"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        # 逐通道卷积 (Depthwise)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, padding=padding, bias=False)
        # 逐点卷积 (Pointwise)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class LITE_Block(nn.Module):
    """轻量级多尺度 Inception 模块"""
    def __init__(self, in_channels, out_channels=16):
        super().__init__()
        # 3 个不同感受野的分支 (模拟 LITEMV 的多尺度特性)
        self.branch1 = SeparableConv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.branch2 = SeparableConv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.branch3 = SeparableConv1d(in_channels, out_channels, kernel_size=39, padding=19)
        
        # 1 个 MaxPool 分支
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch4 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(self.pool(x))
        
        out = torch.cat([x1, x2, x3, x4], dim=1) # (B, out_channels*4, T)
        return self.relu(self.bn(out))

class Net(nn.Module):
    """
    基线模型 3：SOTA LITEMV (2025) + 简单拼接融合
    """
    def __init__(self, num_classes=2, traj_input_dim=4, dropout_p=0.5):
        super(Net, self).__init__()

        # 1. 属性模块
        self.attr_net = Attr.Net()
        self.attr_dim = self.attr_net.out_size()

        # 2. LITEMV 时序特征提取
        # 使用两层 LITE Block
        self.lite_layer1 = LITE_Block(in_channels=traj_input_dim, out_channels=16)
        self.lite_layer2 = LITE_Block(in_channels=64, out_channels=16) # 16*4 = 64
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)   
        self.traj_dim = 64 # 第二层输出通道数

        # 3. 分类器
        fusion_dim = self.attr_dim + self.traj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )

    def forward(self, attr, traj, config=None):
        attr_vec = self.attr_net(attr)  

        # 轨迹维度转换: (B, T, 4) -> (B, 4, T) 供 Conv1d 使用
        x = torch.stack([traj['ex'], traj['ey'], traj['speed'], traj['acc']], dim=1)
        
        # 提取时序特征
        x = self.lite_layer1(x)
        x = self.lite_layer2(x)
        
        # 屏蔽 Padding 区域
        mask = traj['mask'].unsqueeze(1) # (B, 1, T)
        x = x * mask
        
        # Global Average Pooling
        valid_lens = traj['mask'].sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)
        traj_vec = torch.sum(x, dim=2) / valid_lens # (B, 64)

        # 拼接并分类
        feat_fused = torch.cat([attr_vec, traj_vec], dim=1)
        logits = self.classifier(feat_fused)
        
        return logits