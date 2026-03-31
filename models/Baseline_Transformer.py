import torch
import torch.nn as nn
import math
from .base import Attr  # 复用现有属性模块

class PositionalEncoding(nn.Module):
    """Transformer 必须的位置编码"""
    def __init__(self, d_model, max_len=15000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return x

class Net(nn.Module):
    """
    基线模型 2：Transformer Encoder + Mean Pooling + Simple Concat
    """
    def __init__(self, 
                 num_classes=2, 
                 traj_input_dim=4, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dropout_p=0.5):
        super(Net, self).__init__()

        # 1. 属性模块
        self.attr_net = Attr.Net()
        self.attr_dim = self.attr_net.out_size()

        # 2. 轨迹特征嵌入
        self.input_proj = nn.Linear(traj_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类器
        fusion_dim = self.attr_dim + d_model
        
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
        # (B, T, 4) -> (B, T, D_model)
        x = torch.stack([traj['ex'], traj['ey'], traj['speed'], traj['acc']], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # 生成 Padding Mask 给 Transformer (注意：Transformer 里 True 代表是被 pad 的位置，需要被忽略)
        # 你的 mask 里 1 是有效，0 是 pad，所以要取反
        src_key_padding_mask = (traj['mask'] == 0) # (B, T), bool tensor

        # Transformer 编码: (B, T, D_model)
        enc_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # --- 3. Mean Pooling 获取全序列特征 ---
        # 必须排除 padding 部分的干扰
        mask_expanded = traj['mask'].unsqueeze(-1)  # (B, T, 1)
        sum_hidden = torch.sum(enc_out * mask_expanded, dim=1) # (B, D_model)
        valid_lens = traj['mask'].sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)
        
        traj_vec = sum_hidden / valid_lens # 平均池化 (B, D_model)

        # --- 4. 拼接与分类 ---
        feat_fused = torch.cat([attr_vec, traj_vec], dim=1)
        logits = self.classifier(feat_fused)
        
        return logits