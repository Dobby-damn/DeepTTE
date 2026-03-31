import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from . import base  # 包含你修改过的 Attr.py 和 SpatioTemporal.py

sys.path.append(str(Path(__file__).parent))

class Net(nn.Module):
    """
    DeepTTE 模型改造版 —— 用于二分类任务（label=0/1）
    复用了 DeepTTE 的 Attr 和 SpatioTemporal 模块，但将输出头改为分类器。
    """

    def __init__(self, 
                 num_classes=2,      # 二分类
                 kernel_size=3, 
                 num_filter=16, 
                 pooling_method='attention', 
                 hidden_size=32, 
                 num_fc_layers=1,
                 dropout_p=0.5):
        super(Net, self).__init__()

        # 属性模块（Attr.Net）
        self.attr_net = base.Attr.Net()

        # 时空特征模块（SpatioTemporal.Net）
        self.spatio_temporal = base.SpatioTemporal.SpatioTemporalNet(
            attr_size=self.attr_net.out_size(),
            kernel_size=kernel_size,
            num_filter=num_filter,
            pooling=pooling_method
        )

        # 获取维度
        self.attr_dim = self.attr_net.out_size()
        self.traj_dim = self.spatio_temporal.out_size()

        # 改进：使用 Gated Fusion 机制
        # 思想：利用静态特征生成一个“门控权重”，来加权时空特征
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.attr_dim, self.traj_dim),
            nn.BatchNorm1d(self.traj_dim),
            nn.ReLU(),
            nn.Linear(self.traj_dim, self.traj_dim),
            nn.Sigmoid()
        )
        # 初始时将门控层的偏置设置为较大值（如2.0），以便在训练初期更关注时空特征
        nn.init.constant_(self.fusion_gate[-2].bias, 2.0)
        # 融合后的维度
        fusion_dim = self.attr_dim + self.traj_dim
        
        # 分类器构建
        fc_layers = []
        curr_dim = fusion_dim

        # 添加多个隐藏层
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(curr_dim, hidden_size))
            fc_layers.append(nn.BatchNorm1d(hidden_size)) # 建议加入BN防止过拟合
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_p)) # 加入 Dropout 进一步防止过拟合
            curr_dim = hidden_size
        
        fc_layers.append(nn.Linear(curr_dim, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, attr, traj, config=None, return_attention=False):
        """
        参数：
            attr: dict，包括连续属性张量，如 {'time': ..., 'dist': ..., 'pause_count': ...}
            traj: dict，包括轨迹序列张量，如 {'ex': ..., 'ey': ...}
            config: 可选，用于兼容接口
        返回：
            logits: (B, num_classes)
        """
        # 提取属性向量
        attr_vec = self.attr_net(attr)  # (B, D_attr)

        # 提取时空轨迹特征
        sptm_vec, alpha = self.spatio_temporal(traj, attr_vec, config)  # (B, D_traj)
        # --- 改进的融合策略 ---
        # 简单的 Concat
        # concat_feat = torch.cat([attr_vec, sptm_vec], dim=1)
        
        # 计算门控 (非必须，但对于MCI任务通常有效)
        # 意思是：结合了年龄和轨迹后，决定哪些特征更重要
        z = self.fusion_gate(attr_vec)  # (B, D_traj)，每个维度一个权重
        
        # 加权时空特征 + 原始属性特征
        # 这里是一个简单的 Residual Gating 变体
        feat_fused = torch.cat([attr_vec, sptm_vec * z], dim=1)
        
        logits = self.classifier(feat_fused)
        if return_attention:
            return logits, alpha
        return logits

    def compute_loss(self, logits, labels):
        """
        分类损失函数封装
        """
        return F.cross_entropy(logits, labels)

    def predict(self, attr, traj, config=None):
        """
        推理接口：输出类别预测
        """
        logits = self.forward(attr, traj, config)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs
