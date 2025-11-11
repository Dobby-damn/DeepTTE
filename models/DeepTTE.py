import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import base

# =============================
#   分类版本 EntireEstimator
# =============================
class EntireEstimator(nn.Module):
    """全局估计器：分类任务"""
    def __init__(self, input_size, num_final_fcs, hidden_size=128, num_classes=2):
        super().__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        self.residuals = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_final_fcs)])
        self.hid2out = nn.Linear(hidden_size, num_classes)   # 改为分类输出
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)
        hidden = F.leaky_relu(self.input2hid(inputs))
        for res in self.residuals:
            residual = F.leaky_relu(res(hidden))
            hidden = hidden + residual
        logits = self.hid2out(hidden)   # [B, num_classes]
        return logits

    def eval_on_batch(self, logits, label):
        """
        Args:
            logits: [B, num_classes]
            label: [B] LongTensor (0/1)
        """
        loss = self.criterion(logits, label)
        prob = torch.softmax(logits, dim=-1)[:, 1]   # 正类概率
        pred = torch.argmax(logits, dim=-1)
        pred_dict = {'logits': logits, 'prob': prob, 'pred': pred, 'label': label}
        return pred_dict, loss


# =============================
#   Net 主网络
# =============================
class Net(nn.Module):
    """DeepTTE 分类版"""
    def __init__(self, kernel_size=3, num_filter=32, pooling_method='attention',
                 num_final_fcs=3, final_fc_size=128, num_classes=2, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.num_classes = num_classes

        self.build()
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def build(self):
        # 属性子网
        self.attr_net = base.Attr.Net()
        # 时空子网
        self.spatio_temporal = base.SpatioTemporal.Net(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=self.num_filter,
            pooling_method=self.pooling_method
        )
        # 分类头
        self.entire_estimate = EntireEstimator(
            input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=self.num_final_fcs,
            hidden_size=self.final_fc_size,
            num_classes=self.num_classes
        )

    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr)
        _, _, sptm_t = self.spatio_temporal(traj, attr_t, config)
        logits = self.entire_estimate(attr_t, sptm_t)  # [B, num_classes]
        return logits

    def eval_on_batch(self, attr, traj, config):
        """
        Args:
            attr: 必须包含 'label' 字段 [B]
        """
        print('attr:', attr)
        logits = self(attr, traj, config)             # [B, num_classes]
        label = attr['label'].long().view(-1)         # [B]
        pred_dict, loss = self.entire_estimate.eval_on_batch(logits, label)
        return pred_dict, loss
