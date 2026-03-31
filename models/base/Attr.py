import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    改进版Attr网络：
    1. 移除 driverID (防止数据泄漏)
    2. 加入 Gender Embedding (性别差异对运动控制有影响)
    3. 增强连续特征处理 (MLP 提取非线性关系)
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # 定义需要 Embedding 的离散特征
        # 假设 gender: 0=女, 1=男 (根据你的数据编码调整)
        self.embed_configs = {} 
        
        # 自动构建 Embedding 层
        for name, (n_cls, dim) in self.embed_configs.items():
            self.add_module(f'{name}_em', nn.Embedding(n_cls, dim))

        # 连续特征列表 (必须与 main.py 中 collate_fn 传进来的 key 一致)
        # 建议包含：age, edu_year, 以及过程统计特征
        self.cont_fields = ['Unnamed: 0', 'record_id', 'age', 'hand', 'edu_year',\
                            'edugrade', 'occupation', 'habit1', 'habit2', 'habit3', 'habit4',\
                            'habit5', 'habit6', 'ssoid', 'oppo_sub','hkbcscore', 'evaluation_type', \
                            'data_type', 'completed', 'correctConnections', 'incorrectConnections', 
                            'clickCount', 'noTouchCount', 'gameDuration', 'type', 'connTime', 'show_T', 
                            'show_D', 'ConnTime_ET1', 'ConnTime_T1', 'ConnTime_D', 'ConnTime_T', 
                            'education', 'mmse_s', 'adl_s', 'phq_s', 'sas_s', 'ab42', 'ab40', 'ab42_40', 
                            'pt217', 'pt217_ab42', 'Unnamed: 3', 'mean_speed', 'std_speed', 
                            'total_distance', 'total_time', 'pause_count', 'max_speed', 'min_speed', 
                            'speed_variation', 'point_count', 'complexity_ratio', 'direction_changes', 
                            'mean_acceleration', 'jerk_std', 'max_pause_duration', 'pause_time_ratio', 
                            'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points'
                            ]
        
        # 计算连续特征的总维度
        self.num_cont = len(self.cont_fields)
        
        # 使用 MLP 而不是单层 Linear 来处理连续特征
        # 这有助于捕捉年龄与速度之间的非线性关系
        self.cont_mlp = nn.Sequential(
            nn.Linear(self.num_cont, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 最终映射到 8 维
        )
        
    def out_size(self):
        # 输出维度 = 所有 Embedding 输出之和 + 连续特征 MLP 输出
        total_sz = 0
        for _, (_, dim) in self.embed_configs.items():
            total_sz += dim
        total_sz += 16 # cont_mlp 的输出维度
        return 16

    ''' # def forward(self, attr):
    #     """
    #     attr: dict, key 对应 self.cont_fields 和 self.embed_configs 中的名字
    #     """
        
    #     em_outputs = []

    #     # 1. 处理离散特征 (sex)
    #     for name, _ in self.embed_configs.items():
    #         if name in attr:
    #             print(name)
    #             val = attr[name].long().view(-1) # 确保是 LongTensor
    #             layer = getattr(self, f'{name}_em')
    #             em_outputs.append(layer(val))
    #         else:
    #             print(f"[Warning] 特征缺失: '{name}' 在输入 attr 中未找到！")

    #     # 🔴 检查一下实际收集到了几个
    #     if len(em_outputs) != self.num_cont:
    #         print(f"[Error] 模型定义了 {self.num_cont} 个特征, 但实际只找到 {len(em_outputs)} 个。")
            
    #         print(f"      Attr.py 定义列表: {self.cont_fields}")
    #         print(f"      DataLoader 提供列表: {list(attr.keys())}")

    #     # 2. 处理连续特征
    #     cont_vals = []
    #     for name in self.cont_fields:
    #         if name in attr:
    #             # 确保维度是 (B, 1)
    #             val = attr[name].view(-1, 1).float() 
    #             cont_vals.append(val)
    #         else:
    #             # 如果某个特征缺失，给予警告或补0 (建议在DataLoader处理好)
    #             pass 
                
    #     if cont_vals:
    #         cont_tensor = torch.cat(cont_vals, dim=1) # (B, num_cont)
    #         cont_emb = self.cont_mlp(cont_tensor)     # (B, 16)
    #         em_outputs.append(cont_emb)

    #     # 3. 拼接所有特征
    #     return torch.cat(em_outputs, dim=1) # (B, total_dim)
    '''
    
    def forward(self, attr):
        
        em_outputs = []

        # ========= 连续特征 =========
        cont_values = []
        for name in self.cont_fields:
            if name not in attr:
                raise KeyError(f"[AttrNet] 缺失连续特征: {name}")
            v = attr[name]

            # shape 统一
            if v.dim() > 1:
                v = v.view(v.size(0))

            cont_values.append(v)
            #cont_values.append(attr[name].float())

        cont_tensor = torch.stack(cont_values, dim=1)  # (B, num_cont)
        cont_tensor = torch.nan_to_num(cont_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        cont_tensor = torch.clamp(cont_tensor, -1e6, 1e6)
        if torch.isnan(cont_tensor).any():
            raise RuntimeError("[AttrNet] cont_tensor contains NaN")

        if torch.isinf(cont_tensor).any():
            raise RuntimeError("[AttrNet] cont_tensor contains Inf")
        
        cont_emb = self.cont_mlp(cont_tensor)           # (B, 16)

        # ========= 拼接 =========
        outputs = em_outputs + [cont_emb]
        return torch.cat(outputs, dim=1)
