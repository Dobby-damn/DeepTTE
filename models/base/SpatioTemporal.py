import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GeoConv

class SpatioTemporalNet(nn.Module):
    """
    轨迹时空特征提取模块，用于分类任务。
    """
    def __init__(self, attr_size, kernel_size=5, num_filter=64, pooling='mean', rnn_type='lstm'):
        super(SpatioTemporalNet, self).__init__()

        # 引入你自己修改的 PathNet（处理 ex, ey 序列）
        self.path_net = GeoConv.Net(kernel_size=kernel_size, num_filter=num_filter)

        input_size = num_filter + attr_size  # 不再拼接 +1 (dist_gap)，因为你数据中没有
        hidden_size = 128

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.pooling = pooling
        if pooling == 'attention':
            self.att_fc = nn.Linear(attr_size, hidden_size)

    def out_size(self):
        return 128  # 输出特征维度

    def mean_pooling(self, hiddens, lens):
        # hiddens: (B, T, H)
        mask = (torch.arange(hiddens.size(1))[None, :].to(hiddens.device) < lens[:, None]).float()
        hiddens_sum = torch.sum(hiddens * mask.unsqueeze(2), dim=1)
        return hiddens_sum / lens.unsqueeze(1)

    def attention_pooling(self, hiddens, attr_vec):
        att_weight = F.tanh(self.att_fc(attr_vec)).unsqueeze(2)  # (B, H, 1)
        att_score = torch.bmm(hiddens, att_weight).squeeze(2)    # (B, T)
        alpha = F.softmax(att_score, dim=1).unsqueeze(1)         # (B, 1, T)
        return torch.bmm(alpha, hiddens).squeeze(1)              # (B, H)

    def forward(self, traj, attr_vec, lens=None):
        """
        traj: {'ex': (B,T), 'ey': (B,T)}
        attr_vec: (B, attr_size)
        lens: tensor(B,)
        """
        path_feat_seq = self.path_net(traj)  # (B, num_filter)
        # 如果 path_net 已经输出全局向量，可以跳过 RNN 直接拼接 attr
        # 否则如果 path_net 输出时序 (B,T,D)，则拼接 attr 并送入 RNN

        if len(path_feat_seq.shape) == 2:
            # 已经是全局特征
            fused = torch.cat([path_feat_seq, attr_vec], dim=1)
            return fused  # (B, num_filter + attr_size)

        # 如果 path_feat_seq 是时序特征
        B, T, D = path_feat_seq.shape
        attr_expanded = attr_vec.unsqueeze(1).expand(B, T, attr_vec.size(1))
        x = torch.cat([path_feat_seq, attr_expanded], dim=2)

        packed, (h_n, c_n) = self.rnn(x)
        if self.pooling == 'mean':
            pooled = torch.mean(packed, dim=1)
        elif self.pooling == 'attention':
            pooled = self.attention_pooling(packed, attr_vec)
        else:
            pooled = h_n[-1]

        return pooled  # (B, 128)
