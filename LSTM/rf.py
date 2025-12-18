"""train_lstm.py
================
Process variable‑length eye‑tracking trajectories and train a 5‑layer LSTM classifier.

File structure expected (relative to this script):
└── LSTM/
    ├── iddiag.csv                # evaluation_id, diagnose
    ├── 连线测试轨迹(1).csv         # evaluation_id, ex, ey
    └── 连线测试轨迹(2).xlsx        # evaluation_id, ex, ey

Outputs
-------
* checkpoints/best_model_epoch*.pt – best model per validation accuracy
* loss_curve.png, accuracy_curve.png – training curves


"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from data_get import TrajectoryFeatureExtractor


# print("GPU是否可用：", torch.cuda.is_available())
# print("当前GPU名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")


class ShapWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, full_input):
        """
        full_input: (batch, seq_len + static_dim)
        你需要将其拆分回 x, lengths, static_feats
        """
        batch = full_input.shape[0]

        seq_len = 2048 * 2    # 每个点两个维度
        static_dim = 68      # 根据你的数据而定

        # 1. 还原轨迹
        x = full_input[:, :seq_len]
        x = x.view(batch, 2048, 2)

        # 2. 还原静态特征
        static_feats = full_input[:, seq_len: seq_len + static_dim]

        # 3. 你需要提供长度（用真实的）
        lengths = torch.full((batch,), 2048, dtype=torch.int64).to(x.device)

        return self.model(x, lengths, static_feats)


class TrajectoryDataset(Dataset):
    """Dataset that groups (ex, ey) clicks per evaluation_id and attaches a label."""

    def __init__(
        self,
        iddiag_path: Path,
        traj_csv_path: Path,
        traj_xlsx_path: Path,
        traj_path_new: Path,
        X: pd.DataFrame = None, # 传递计算得到的属性特征
    ) -> None:
        # Load diagnostic labels
        iddiag = pd.read_parquet('data.parquet')
        # 输出diagnose列的类型  
        

        
        #筛选出诊断标签为0和1的样本 个，正常280，异常416？为什么还少了
        iddiag = iddiag[iddiag["diagnose"].isin([0, 1])].copy()   # 只保留标签为0和1的样本
        self.label_encoder = LabelEncoder()      
        iddiag["diagnose_encoded"] = self.label_encoder.fit_transform(iddiag["diagnose"])

        label_map = dict(zip(iddiag["evaluation_id"], iddiag["diagnose_encoded"]))  # id与label的映射关系 690条
        print("[Info] 二分类标签分布：\n", iddiag["diagnose_encoded"].value_counts())
        
        # ==== 对静态特征做归一化 ====
        
        self.eval_ids = X["evaluation_id"].values
        static_features = X.drop(columns=["evaluation_id"]).values     
        self.scaler = StandardScaler()
        # print(X.columns)
        # print(X.dtypes)
        # print("=== Inspect X columns ===")
        # for col in X.columns:
        #     print(col, X[col].dtype)
        self.static_features_scaled = self.scaler.fit_transform(static_features)

        # 保证 evaluation_id 和归一化后的特征对应
        self.static_map = {
            eid: feat for eid, feat in zip(self.eval_ids, self.static_features_scaled)
        }

        traj1 = pd.read_csv(traj_csv_path, engine="python")
        traj2 = pd.read_excel(traj_xlsx_path)
        traj3 = pd.read_csv(traj_path_new)
        traj = pd.concat([traj1, traj2, traj3], ignore_index=True)   #长度：1462741
        k = 0

        groups = traj.groupby("evaluation_id")   #790 groups 
        # ====== 收集样本 ======
        self.samples: List[Tuple[torch.Tensor, int]] = []
        missing_label = 0
        all_coords = []  # 用于存储所有坐标

        # 此处筛选出100个在轨迹中有而信息学中没有的被试
        for  eval_id, df in groups:
            eval_id = float(eval_id)
            if eval_id not in label_map:
                k += 1
                continue  # 跳过没有有效标签（或非 0/1 类）的受试者
            label = label_map.get(eval_id)
            if label is None:
                missing_label += 1
                continue  # skip participants without a label

            coords = df[["ex", "ey"]].values  
            if coords.size == 0:
                # skip empty trajectories
                k += 1
                continue
            all_coords.append(coords)
        print(f"[Info] 共跳过 {k} 个在轨迹数据中有但在诊断数据中没有的被试。")  #all_coords格式：list，长度690，每个元素是一个(n,2)的numpy数组，n为该被试的轨迹点数。
        
        # ====== 对所有轨迹整体归一化 ======
        if len(all_coords) == 0:
            # no trajectories found -> create dummy mean/std
            coord_mean = np.array([0.0, 0.0], dtype=np.float32)
            coord_std = np.array([1.0, 1.0], dtype=np.float32)
        else:
            all_coords = np.vstack(all_coords)  # 拼接成大矩阵
            coord_mean = all_coords.mean(axis=0)
            coord_std = all_coords.std(axis=0) + 1e-6  # 防止除0

        # ====== 重新构建样本 ======
        for eval_id, df in groups:
            if eval_id not in label_map:
                continue
            label = label_map.get(eval_id)
            if label is None:
                continue

            coords = df[["ex", "ey"]].values[:2048].astype(np.float32)    #截断到2048个点
            coords = (coords - coord_mean) / coord_std
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            coords = torch.tensor(coords, dtype=torch.float32)

            static_feat = torch.tensor(self.static_map[eval_id], dtype=torch.float32)
            static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=0.0, neginf=0.0)

            self.samples.append((coords, static_feat, int(label)))
          
        if missing_label:
            print(f"[Info] Skipped {missing_label} participants without labels.")

    def __len__(self) -> int:  # noqa: D401
        """Return number of participants available."""
        return len(self.samples)


    def __getitem__(self, idx: int):  # noqa: D401, D403
        def compute_motion_features(coords):
            # coords: (seq_len, 2)  ex, ey
            x = coords[:, 0]
            y = coords[:, 1]

            # 一阶差分（速度）
            dx = torch.diff(x, prepend=x[:1])
            dy = torch.diff(y, prepend=y[:1])
            speed = torch.sqrt(dx**2 + dy**2)

            # 二阶差分（加速度）
            ax = torch.diff(dx, prepend=dx[:1])
            ay = torch.diff(dy, prepend=dy[:1])

            # 方向 & 转向角速度
            heading = torch.atan2(dy, dx)
            turn_rate = torch.diff(heading, prepend=heading[:1])

            return torch.stack([dx, dy, speed, ax, ay, heading, turn_rate], dim=1)
        # return self.samples[idx]
        coords, static_feat, label = self.samples[idx]
        motion_feat = compute_motion_features(coords)  # [seq, 7]
        coords = torch.cat([coords, motion_feat], dim=1)
        return coords, static_feat, label


def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
    """Custom collate_fn to pad variable‑length sequences within each mini‑batch."""

    # sequences, labels = zip(*batch)
    # lengths = torch.tensor([len(seq) for seq in sequences])
    # padded_sequences = pad_sequence(sequences, batch_first=True)  # zero‑pad shorter seqs
    # return padded_sequences, lengths, torch.tensor(labels)
    coords, static_feats, labels = zip(*batch)
    lengths = torch.tensor([len(c) for c in coords], dtype=torch.long)

    coords_padded = nn.utils.rnn.pad_sequence(coords, batch_first=True)
    static_feats = torch.stack(static_feats)
    labels = torch.tensor(labels, dtype=torch.long)
    return coords_padded, lengths, static_feats, labels

# 构造bucket，样本按照序列长度分组
class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 记录每个样本的长度
        self.lengths = [len(data_source.dataset.samples[i][0]) for i in data_source.indices]
        # self.lengths = [len(seq) for seq, _ in data_source.samples]

        # 排序并分桶
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        self.buckets = [
            sorted_indices[i:i + batch_size] 
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)
        for bucket in self.buckets:
            yield bucket

    def __len__(self):
        return len(self.buckets)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """实现带热重启的余弦退火学习率"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        
        T_cur = self.last_epoch % self.T_0
        if self.T_mult != 1:
            T_cur = self.last_epoch % (self.T_0 * (self.T_mult ** (self.last_epoch // self.T_0)))
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]


class TrajectoryRFClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        包装 sklearn 的 RandomForest，使其适配 PyTorch DataLoader 的输入格式
        """
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # 并行计算
            class_weight='balanced' # 处理样本不平衡
        )

    def extract_features(self, x, lengths, static_feats):
        """
        将时序数据转换为统计特征向量
        Input:
            x: (Batch, Seq_Len, 9) - Tensor or Numpy
            lengths: (Batch,)
            static_feats: (Batch, 67)
        Output:
            features: (Batch, Total_Dim) - Numpy Array
        """
        # 确保转为 numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().numpy()
        if isinstance(static_feats, torch.Tensor):
            static_feats = static_feats.cpu().numpy()

        batch_size = x.shape[0]
        num_channels = x.shape[2] # 9个特征通道 (ex, ey, vel, ...)
        
        traj_features_list = []

        for i in range(batch_size):
            # 1. 截取有效序列 (去除 Padding)
            # 这一点非常关键！不能把填充的0算进平均值里
            seq_len = lengths[i]
            valid_seq = x[i, :seq_len, :] # shape: (Real_Len, 9)

            if seq_len == 0:
                # 极其罕见的情况，给全0
                stats = np.zeros(num_channels * 5)
            else:
                # 2. 计算统计特征 (针对每个通道计算)
                # 包括: 均值, 标准差, 最大值, 最小值, 最后一个时间点的值(终点状态)
                feat_mean = np.mean(valid_seq, axis=0) # (9,)
                feat_std  = np.std(valid_seq, axis=0)  # (9,)
                feat_max  = np.max(valid_seq, axis=0)  # (9,)
                feat_min  = np.min(valid_seq, axis=0)  # (9,)
                # 某些诊断可能看重最后一个点的状态（比如是否最后停顿了）
                feat_last = valid_seq[-1, :]           # (9,)

                # 拼接该样本的所有时序统计特征 -> 9 * 5 = 45 维
                stats = np.concatenate([feat_mean, feat_std, feat_max, feat_min, feat_last])
            
            traj_features_list.append(stats)

        # (Batch, 45)
        traj_features = np.array(traj_features_list)
        
        # 3. 与静态特征拼接
        # 最终维度: 45 (轨迹统计) + 67 (静态) = 112 维
        combined_features = np.concatenate([traj_features, static_feats], axis=1)
        
        return combined_features

    def prepare_data_from_loader(self, dataloader):
        """
        遍历整个 DataLoader，收集所有数据并转换为 RF 可用的 Numpy 矩阵
        """
        all_feats = []
        all_labels = []

        print("Extracting features for Random Forest...")
        for batch in tqdm(dataloader):
            # 解包你的 DataLoader (假设顺序是 x, lengths, static, label)
            x, lengths, static_feats, labels = batch
            
            # 提取特征
            feats = self.extract_features(x, lengths, static_feats)
            
            all_feats.append(feats)
            all_labels.append(labels.cpu().numpy())

        # 堆叠成大矩阵
        X_full = np.vstack(all_feats) # (Total_Samples, 112)
        y_full = np.concatenate(all_labels) # (Total_Samples,)
        
        return X_full, y_full

    def fit(self, train_loader):
        X_train, y_train = self.prepare_data_from_loader(train_loader)
        print(f"Start Training Random Forest with input shape: {X_train.shape}")
        self.clf.fit(X_train, y_train)
        print("RF Training Done.")

    def evaluate(self, test_loader):
        X_test, y_test = self.prepare_data_from_loader(test_loader)
        
        # 预测
        y_pred = self.clf.predict(X_test)
        
        # 评估
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n=== Random Forest Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return acc


def main():  
    # ------------------------------ Paths & Hyper‑parameters -----------------------------
    data_dir = Path("LSTM")
    id_path = r"D:/Code/DeepTTE/data/iddiag.csv"    # 身份id"video",age,sex,,edu_year,habit,label
    iddiag_path = r"D:/Code/DeepTTE/data/连线测试.csv"       # 其他的一些特征
    traj_csv_path = r"D:/Code/DeepTTE/data/连线测试轨迹(1).csv"    #轨迹1
    traj_xlsx_path = r"D:/Code/DeepTTE/data/连线测试轨迹(2).xlsx"   #轨迹2
     # 新增的诊断文件和轨迹文件
    id_path_new = r"D:/Code/DeepTTE/data/体检id.xlsx"       # o列是ID F列是MoCA分数
    iddiag_path_new = r"D:/Code/DeepTTE/data/连线测试体检.csv"      # 其他的一些特征
    traj_path_new = r"D:/Code/DeepTTE/data/连线测试轨迹体检.csv"    # 轨迹特征

    # # 获取属性特征，edu_year, sex, age ,mean_speed etlc.    我的目标：将所有的特征整合为一个打的dataframe变量。（储存到csv中？）
    # extractor = TrajectoryFeatureExtractor(id_path, iddiag_path, traj_csv_path, traj_xlsx_path, id_path_new, iddiag_path_new, traj_path_new)
    # X, y = extractor.get_feature_matrix_and_target() #这里返回的是DataFrame格式

    # print(X.head())
    df = pd.read_parquet('data.parquet')
    # features = [
    #     'age', 'edu_year',  # 人口统计'edu_year','sex'
    #     'mean_speed', 'std_speed', 'total_distance',  # 轨迹特征
    #     'total_time', 'pause_count', 'speed_variation', 'point_count',
    #     'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points',  # 曲率特征
    #     'complexity_ratio', 'direction_changes', 'mean_acceleration',
    #     'jerk_std', 'max_pause_duration', 'pause_time_ratio', 'evaluation_id',
    # ]

    x = df.drop(columns=['video','hkbcscore','moca_s','moca_score','diagnose','id','birthdate','game_code','save_time','create_time','update_time','touchDuration','numberInterval','name','Unnamed: 2'])  # 删除无关列
    print("初始特征",x.columns.tolist())
    # Ensure we do NOT leak labels in the static features
    label_cols = [c for c in ['diagnose', 'diagnose_encoded', 'label'] if c in x.columns]
    if len(label_cols) > 0:
        print(f"[Warning] Dropping label columns from static features: {label_cols}")
        x = x.drop(columns=label_cols)

    feature_names = x.columns.tolist()
    print("初始特征列表：", feature_names)
    # for feature_name in feature_names:
        # assert feature_name in x.columns, f"{feature_name} 不在特征列表中!"
    X = x#.drop(columns=['Unnamed: 0']).copy()
        # print(X.columns.tolist())
        


    batch_size = 32

    # ----------------------------------- Dataset ---------------------------------------
    dataset = TrajectoryDataset('data.parquet', traj_csv_path, traj_xlsx_path, traj_path_new, X)
    
    labels = [lbl for *_, lbl in dataset.samples]
    # 0.7/0.15/0.15 划分训练/验证/测试集
    train_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.3,
        random_state=10,
        stratify=labels             # 按标签比例分层抽样
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=10,
        stratify=[labels[i] for i in temp_idx]
    )
    train_sampler = BucketBatchSampler(torch.utils.data.Subset(dataset, train_idx), batch_size) #按照长度分组

    train_loader = DataLoader(
        dataset,
        batch_sampler = train_sampler,
        # batch_size = batch_size,
        # shuffle = True,
        collate_fn = collate_fn,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_fn,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # 1. 初始化模型
    rf_model = TrajectoryRFClassifier(n_estimators=200, max_depth=20)

    # 2. 训练
    # 注意：RF 是一次性训练，不需要 epoch 循环
    rf_model.fit(train_loader)

    # 3. 测试
    rf_model.evaluate(test_loader)
    del train_loader
    del test_loader
    del val_loader
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
