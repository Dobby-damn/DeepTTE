import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from . import GeoConv

class SpatioTemporalNet(nn.Module):
    """
    改进版时空特征提取模块：
    1. 支持 PackedSequence (忽略 Padding)
    2. 支持 Masked Attention (防止关注无效区域)
    """
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling='attention', rnn_type='lstm'):
        super(SpatioTemporalNet, self).__init__()

        # 1. 轨迹卷积 (输出 B, T, num_filter)
        self.path_net = GeoConv.Net(kernel_size=kernel_size, num_filter=num_filter)

        # 2. RNN 输入维度 = 卷积特征 + 属性特征
        # Attr 会被 copy 到每个时间步，这是 DeepTTE 的经典操作，保留
        self.rnn_input_size = num_filter + attr_size 
        self.hidden_size = 32

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.rnn_input_size, 
                               hidden_size=self.hidden_size, 
                               num_layers=1, 
                               batch_first=True,
                               bidirectional=True,
                               dropout=0.5) # 增加 dropout 防止过拟合
        else:
            self.rnn = nn.GRU(input_size=self.rnn_input_size, 
                              hidden_size=self.hidden_size, 
                              num_layers=1, 
                              batch_first=True,
                              bidirectional=True,
                              dropout=0.5)

        self.pooling = pooling
        
        # Attention 机制参数
        if pooling == 'attention':
            # 这里的 Attention 试图寻找与当前属性（如年龄）最相关的轨迹片段
            self.att_fc = nn.Linear(attr_size, self.hidden_size * 2)

    def out_size(self):
        # 必须返回真实的 hidden_size，否则维度对不上
        return self.hidden_size * 2

    def mean_pooling(self, hiddens, mask):
        """
        hiddens: (B, T, H)
        mask: (B, T) 0/1
        """
        # 将 mask 扩展到隐藏层维度 (B, T, H)
        mask_expanded = mask.unsqueeze(-1).expand_as(hiddens)
        
        # 只对有效区域求和
        sum_hidden = torch.sum(hiddens * mask_expanded, dim=1) # (B, H)
        
        # 计算有效长度
        lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1) 防止除0
        
        return sum_hidden / lens

    def attention_pooling(self, hiddens, attr_vec, mask):
        """
        hiddens: (B, T, H)
        attr_vec: (B, D_attr)
        mask: (B, T)
        """
        # 1. 计算 Query (基于属性生成的权重向量)
        att_query = torch.tanh(self.att_fc(attr_vec)).unsqueeze(2) # (B, H, 1)
        
        # 2. 计算 Score (Batch Matrix Multiplication)
        # (B, T, H) @ (B, H, 1) -> (B, T, 1)
        att_score = torch.bmm(hiddens, att_query).squeeze(2) # (B, T)
        
        # 3. Masking (关键步骤！)
        # 将 Padding 位置的分数设为极小值 (-1e9)，这样 Softmax 后概率为 0
        att_score = att_score.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax
        alpha = F.softmax(att_score, dim=1).unsqueeze(1) # (B, 1, T)
        
        # 5. 加权求和
        context = torch.bmm(alpha, hiddens).squeeze(1) # (B, H)
        return context, alpha.squeeze(1)

    def forward(self, traj, attr_vec, config=None):
        """
        traj: {'ex': (B,T), 'ey': (B,T), 'mask': (B,T)}
        attr_vec: (B, attr_size)
        """
        # 1. 获取卷积特征 (B, T, num_filter)
        path_feat_seq = self.path_net(traj) 
        
        B, T, D_conv = path_feat_seq.shape
        
        # 2. 拼接属性特征 (Attribute Expansion)
        # 将静态属性复制 T 份，拼接到每个时间步
        attr_expanded = attr_vec.unsqueeze(1).expand(B, T, attr_vec.size(1))
        
        # rnn_input: (B, T, num_filter + attr_size)
        rnn_input = torch.cat([path_feat_seq, attr_expanded], dim=2)
        
        # 3. 处理变长序列 (Pack Padded Sequence)
        # 必须从 mask 计算真实的长度
        mask = traj['mask'] # (B, T)
        lengths = mask.sum(dim=1).long().cpu() # pack_padded 需要 CPU tensor
        
        # 为了使用 pack_padded_sequence，序列长度最好是降序，但 enforce_sorted=False 允许乱序
        packed_input = pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
        
        # 4. RNN 前向传播
        packed_output, (h_n, c_n) = self.rnn(packed_input)
        
        # 5. 解包 (Unpack)
        # output: (B, T, hidden_size)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=T)
        
        # 6. Pooling
        if self.pooling == 'mean':
            pooled = self.mean_pooling(rnn_output, mask)
        elif self.pooling == 'attention':
            pooled, alpha = self.attention_pooling(rnn_output, attr_vec, mask)
        else:
            # 取最后一个有效时间步的 hidden state
            # 这是一个简单的写法，更严谨的写法应该根据 lengths 索引
            # pooled = h_n[-1] # 注意：这是双向或多层时的最后一层，若是多层需小心
            pooled = torch.mean(rnn_output, dim=1) 

        return pooled, alpha if self.pooling == 'attention' else None