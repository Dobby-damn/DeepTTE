import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SpatioTemporalNet_NoGeoConv(nn.Module):
    """
    消融实验变体：w/o GeoConv
    移除 1D-CNN，直接将原始序列特征输入 BiLSTM
    """
    def __init__(self, attr_size, traj_input_dim=4, hidden_size=48, pooling='attention', rnn_type='lstm'):
        super(SpatioTemporalNet_NoGeoConv, self).__init__()

        # 🔴 修改 1：彻底移除 self.path_net = GeoConv.Net(...)

        # 🔴 修改 2：调整 RNN 的输入维度
        # 以前是 num_filter(32) + attr_size，现在直接是 原始特征(4) + attr_size
        self.rnn_input_size = traj_input_dim + attr_size 
        self.hidden_size = hidden_size

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.rnn_input_size, 
                               hidden_size=self.hidden_size, 
                               num_layers=1, 
                               batch_first=True,
                               bidirectional=True,   # 保持双向
                               dropout=0.5)
        
        self.pooling = pooling
        self.feature_dim = self.hidden_size * 2 
        
        if pooling == 'attention':
            self.att_fc = nn.Linear(attr_size, self.feature_dim)

    def out_size(self):
        return self.feature_dim
        
    def attention_pooling(self, hiddens, attr_vec, mask):
        att_query = torch.tanh(self.att_fc(attr_vec)).unsqueeze(2) 
        att_score = torch.bmm(hiddens, att_query).squeeze(2) 
        att_score = att_score.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(att_score, dim=1).unsqueeze(1) 
        context = torch.bmm(alpha, hiddens).squeeze(1) 
        return context

    def forward(self, traj, attr_vec, config=None):
        # 🔴 修改 3：不通过 path_net，直接手动堆叠原始输入特征
        # shape: (B, T, 4)
        raw_traj_feat = torch.stack([traj['ex'], traj['ey'], traj['speed'], traj['acc']], dim=-1)
        B, T, _ = raw_traj_feat.shape
        
        # 拼接属性特征 (Attribute Expansion)
        attr_expanded = attr_vec.unsqueeze(1).expand(B, T, attr_vec.size(1))
        
        # rnn_input shape: (B, T, 4 + attr_size)
        rnn_input = torch.cat([raw_traj_feat, attr_expanded], dim=2)
        
        # 处理变长序列
        mask = traj['mask']
        lengths = mask.sum(dim=1).long().cpu()
        packed_input = pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
        
        # RNN 前向传播
        packed_output, _ = self.rnn(packed_input)
        
        # 解包
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)
        
        # Pooling
        if self.pooling == 'attention':
            pooled = self.attention_pooling(rnn_output, attr_vec, mask)
        else:
            # fallback 为 mean pooling
            mask_expanded = mask.unsqueeze(-1)
            sum_hidden = torch.sum(rnn_output * mask_expanded, dim=1)
            valid_lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = sum_hidden / valid_lens

        return pooled