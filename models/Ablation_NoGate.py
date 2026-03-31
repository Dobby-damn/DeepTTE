import torch
import torch.nn as nn
from . import base

class Net(nn.Module):
    """
    消融实验变体：w/o Gated Fusion
    移除 Sigmoid 门控机制，退化为传统的特征直接拼接 (Simple Concatenation)
    """
    def __init__(self, 
                 num_classes=2,      
                 kernel_size=3, 
                 num_filter=32,       
                 pooling_method='attention', 
                 hidden_size=48,      
                 num_fc_layers=1,     
                 dropout_p=0.5):
        super(Net, self).__init__()

        # 1. 属性模块 (保持不变)
        self.attr_net = base.Attr.Net() 

        # 2. 时空特征模块 (保持包含 GeoConv 和 Attention 的完整版)
        self.spatio_temporal = base.SpatioTemporal.SpatioTemporalNet(
            attr_size=self.attr_net.out_size(),
            kernel_size=kernel_size,
            num_filter=num_filter,
            pooling=pooling_method
        )

        # 获取维度
        self.attr_dim = self.attr_net.out_size()         
        self.traj_dim = self.spatio_temporal.out_size() 

        # ==========================================
        # 🔴 修改 1：彻底删除 fusion_gate 的定义
        # self.fusion_gate = nn.Sequential(...)  <-- 删除了
        # ==========================================

        # 3. 分类器 (输入维度不变，依然是属性拼接轨迹)
        fusion_dim = self.attr_dim + self.traj_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )

    def forward(self, attr, traj, config=None):
        # 1. 提取两路特征
        attr_vec = self.attr_net(attr)                  # (B, attr_dim)
        sptm_vec = self.spatio_temporal(traj, attr_vec) # (B, traj_dim)
        
        # ==========================================
        # 🔴 修改 2：移除加权过程，直接进行暴力拼接
        # ==========================================
        # 以前是：
        # gate = self.fusion_gate(attr_vec)
        # sptm_vec_weighted = sptm_vec * gate
        # feat_fused = torch.cat([attr_vec, sptm_vec_weighted], dim=1)
        
        # 现在变成最朴素的多模态融合：
        feat_fused = torch.cat([attr_vec, sptm_vec], dim=1)
        
        # 2. 分类
        logits = self.classifier(feat_fused)
        return logits