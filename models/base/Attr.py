import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Net(nn.Module):
    """
    driverID（离散）
    time（连续）
    dist（连续）
    pause_count（连续）
    mean_speed（连续）
    curvature_std（连续）
    处理全局属性特征：
    - driverID (embedding)
    - 连续数值特征: time, dist, pause_count, mean_speed, curvature_std,增加特征要修改 cont_fc 输入维度

    输出一个 (batch_size, attr_dim) 的向量，用于和 Path 模块输出拼接。
    """

    # 原始 driverID 最大值 (你需要确认实际范围)
    embed_dims = [('driverID', 9999999, 16)]  # 可根据实际被试数量修改，最多6W个被试，16维embedding

    def __init__(self):
        super(Net, self).__init__()
        self.build()

    def build(self):
        # 构建 driverID embedding
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

        # 处理连续特征的线性层
        # 5个连续特征 -> 16维表示
        self.cont_fc = nn.Linear(5, 16)

    def out_size(self):
        # 输出维度 = driverID embedding + 连续特征输出
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        sz += 16  # cont_fc输出维度
        return sz

    def forward(self, attr):
        """
        attr: dict
            {
              'driverID': tensor(B),
              'time': tensor(B),
              'dist': tensor(B),
              'pause_count': tensor(B),
              'mean_speed': tensor(B),
              'curvature_std': tensor(B)
            }
        """
        em_list = []

        # driverID embedding
        for name, dim_in, dim_out in Net.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)
            attr_t = torch.squeeze(embed(attr_t))
            em_list.append(attr_t)

        # 连续特征拼接
        cont = torch.stack([
            attr['time'],
            attr['dist'],
            attr['pause_count'],
            attr['mean_speed'],
            attr['curvature_std']
        ], dim=1)

        # 可选归一化
        # cont = utils.normalize(cont, 'attr_cont') if hasattr(utils, 'normalize') else cont

        cont_emb = F.relu(self.cont_fc(cont))
        em_list.append(cont_emb)

        return torch.cat(em_list, dim=1)  # shape (B, total_dim)

