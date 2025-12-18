from __future__ import annotations

import os
import json
import time
import math
import torch
import utils
import models
import random
import logger
import inspect
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from datetime import timedelta
from torch.optim import Adam
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

def read_json_array(file_path):
    """按元素读取标准JSON数组文件（每个元素对应原dataset中的字典）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 整体加载JSON数组
        data_list = f.readlines()#json.load(f)
    
    # 按行（元素）遍历
    for idx, item in enumerate(data_list):
        print(f"第{idx+1}条数据：{item}")
        break


#############################################dataloader部分#############################################
class MySet(Dataset):
    def __init__(self, input_file):
        self.data = []
        # 读取数据文件（每行是一个JSON格式的轨迹记录）
        # self.content = open('./data/' + input_file, 'r').readlines() 
        #这里，我的文件并不是每行一个被试，他是一个被试好几行
        with open(input_file, 'r') as f:
            content = json.load(f)
        # 解析JSON数据 
        driver_ids = [item['driverID'] for item in content]
        # print("driverID 最小值:", min(driver_ids))
        # print("driverID 最大值:", max(driver_ids))
        # print("driverID 去重数量:", len(set(driver_ids)))
        # self.content = list(map(lambda x: json.loads(x), self.content))
        # 计算每条轨迹的长度（坐标点数量）
        self.lengths = list(map(lambda x: len(x['ex']), content))
        for item in content:
            attr = {
                'driverID': item['driverID'],
                'time': item['time'],
                'dist': item['dist'],
                'pause_count': item['pause_count'],
                'mean_speed': item['mean_speed'],
                'curvature_std': item['curvature_std'],
                # 'num_features': [
                #     item['time'],
                #     item['dist'],
                #     item['pause_count'],
                #     item['mean_speed'],
                #     item['curvature_std']
                # ],
            }
            # ---- 轨迹特征（序列部分）----
            ex = item['ex']
            ey = item['ey']

            traj = {
                'ex': ex,   # 经度序列
                'ey': ey,   # 纬度序列
                'lens': len(ex),
            }
            label = item['label']
            driver_id = item['driverID']
            self.data.append({'attr': attr, 'traj': traj, 'label': label, 'driverID': driver_id})
          


    def __getitem__(self, idx):
        """获取单条轨迹数据"""
        return self.data[idx]


    def __len__(self):
        """返回数据集总大小"""
        return len(self.data)

def collate_fn(batch):
    """
    将 batch 样本打包为模型可接受的格式
    输出:
        attr: dict{key: (B, 1) Tensor}
        traj: dict{'ex': (B,T,1), 'ey': (B,T,1), 'mask': (B,T)}
        labels: (B,)
    """
    # === 属性部分 ===
    attr_keys = batch[0]['attr'].keys()
    # print("attr_keys:", attr_keys)
    # attr_keys包括[int,[]]
    attr = {}
    for k in batch[0]['attr'].keys():
        if k == 'driverID':
            # ⚠️ Embedding 层必须使用 long 类型索引
            attr[k] = torch.tensor([b['attr'][k] for b in batch], dtype=torch.long)
        else:
            attr[k] = torch.tensor([b['attr'][k] for b in batch], dtype=torch.float32)

    # === 轨迹部分 ===
    ex_list = [torch.tensor(b['traj']['ex'], dtype=torch.float32) for b in batch]
    ey_list = [torch.tensor(b['traj']['ey'], dtype=torch.float32) for b in batch]

    ex_pad = pad_sequence(ex_list, batch_first=True)  # (B, T)
    ey_pad = pad_sequence(ey_list, batch_first=True)  # (B, T)

    lengths = [len(b['traj']['ex']) for b in batch]
    max_len = max(lengths)
    mask = torch.zeros(len(batch), max_len)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    traj = {'ex': ex_pad, 'ey': ey_pad, 'mask': mask}

    # === driverID（如果需要）===
    driver_ids = torch.tensor([b['driverID'] for b in batch], dtype=torch.long)

    # === 标签 ===
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    # print(456)
    # # print(attr['driverID'].shape, traj['ex'].shape, labels.shape, driver_ids.shape)
    # for k,v in attr.items():
    #     print(k, v.shape, v.dtype)

    return attr, traj, labels, driver_ids


def get_loader(file, batch_size, test_ratio= 0.15, val_ratio=0.15, num_workers=0, seed=42):
    """创建数据加载器
    Args:
        file: 文件数据路径
        batch_size: 批次大小
        test_ratio: 测试集比例 (默认0.15)
        val_ratio: 验证集比例 (默认 0.15)
        num_workers: DataLoader 的子进程数量
        seed: 随机种子，保证划分可复现
    Returns:
        配置好的DataLoader实例
    """

    dataset = MySet(input_file=file)


    # 按比例划分
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_sixe = int(total_size * test_ratio)
    train_size = total_size - val_size - test_sixe
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_sixe], generator=torch.Generator().manual_seed(seed))


    # batch_sampler = BatchSampler(dataset, batch_size)

    # 关键配置说明：
    # - batch_size=1 因为批处理已在collate_fn中实现
    # - num_workers=4 使用4个子进程加载数据
    # - pin_memory=True 加速GPU数据传输
    #
    # print("训练集分割")
    train_loader = DataLoader(
        dataset = train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # print("验证集分割")
    # 验证集 DataLoader
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        #batch_sampler=BatchSampler(val_set, batch_size),
        pin_memory=True
    )
    # print("测试集分割")
    # 测试集 DataLoader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        #batch_sampler=BatchSampler(test_set, batch_size),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
#############################################dataloader部分#############################################


class TrajectoryDataset(Dataset):
    """Dataset that groups (ex, ey) clicks per evaluation_id and attaches a label."""

    def __init__(
        self,
        iddiag_path: Path,
        traj_csv_path: Path,
        traj_xlsx_path: Path,
        X: pd.DataFrame = None, # 传递计算得到的属性特征
        
    ) -> None:
        # Load diagnostic labels
        iddiag = pd.read_csv(iddiag_path)
        
        iddiag = iddiag[iddiag["diagnose"].isin([0, 1])].copy()
        self.label_encoder = LabelEncoder()
        iddiag["diagnose_encoded"] = self.label_encoder.fit_transform(iddiag["diagnose"])
    
        label_map = dict(zip(iddiag["evaluation_id"], iddiag["diagnose_encoded"]))
        print("[Info] 二分类标签分布：\n", iddiag["diagnose_encoded"].value_counts())
        
        # ==== 对静态特征做归一化 ====
        self.eval_ids = X["evaluation_id"].values
        static_features = X.drop(columns=["evaluation_id"]).values
        self.scaler = StandardScaler()
        self.static_features_scaled = self.scaler.fit_transform(static_features)

        # 保证 evaluation_id 和归一化后的特征对应
        self.static_map = {
            eid: feat for eid, feat in zip(self.eval_ids, self.static_features_scaled)
        }

        # Load trajectories from two files and concatenate
        traj1 = pd.read_csv(traj_csv_path, engine="python")
        traj2 = pd.read_excel(traj_xlsx_path)
        traj = pd.concat([traj1, traj2], ignore_index=True)

        # Group rows by participant
        groups = traj.groupby("evaluation_id")
        
        # ====== 收集样本 ======
        self.samples: List[Tuple[torch.Tensor, int]] = []
        missing_label = 0
        all_coords = []  # 用于存储所有坐标

        for  eval_id, df in groups:
            if eval_id not in label_map:
                continue  # 跳过没有有效标签（或非 0/1 类）的受试者
            label = label_map.get(eval_id)
            if label is None:
                missing_label += 1
                continue  # skip participants without a label
            coords = df[["ex", "ey"]].values[:2048]  # 截断到最大长度
            all_coords.append(coords)
        # ====== 对所有轨迹整体归一化 ======
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

            coords = df[["ex", "ey"]].values[:2048].astype(np.float32)

            # --- 修改1：归一化处理 ---
            coords = (coords - coord_mean) / coord_std

            # --- 修改2：NaN/Inf 检查 ---
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

            coords = torch.tensor(coords, dtype=torch.float32)
            static_feat = torch.tensor(self.static_map[eval_id], dtype=torch.float32)

            # --- 修改3：静态特征 NaN/Inf 检查 ---
            static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=0.0, neginf=0.0)

            # self.samples.append((coords, static_feat, int(label)))
            sample = {
                'dis': static_feat.tolist()[5],       #total_distance
                'time': static_feat.tolist()[6],      #total_time
                "driverID": int(eval_id),             #evaluation_id
                'pause_count':static_feat.tolist()[7],
                'mean_speed':static_feat.tolist()[3],
                'curvature_std':static_feat.tolist()[12],
                "ex": coords[:, 0].tolist(),
                "ey": coords[:, 1].tolist(),
                "label": int(label),
            }
            self.samples.append(sample)
            
            '''ex ey没有问题，如何将 satic_feat 分开'''
            # coords = torch.tensor(coords, dtype=torch.float32)
            # static_feat = torch.tensor(self.static_map[eval_id], dtype=torch.float32)
            # self.samples.append((coords, static_feat, int(label)))
        if missing_label:
            print(f"[Info] Skipped {missing_label} participants without labels.")

    def __len__(self) -> int:  # noqa: D401
        """Return number of participants available."""
        return len(self.samples)

    def __getitem__(self, idx: int):  # noqa: D401, D403
        return self.samples[idx]


class TrajectoryFeatureExtractor:
    def __init__(self, iddiag_path, traj_csv_path, traj_xlsx_path):
        # 文件路径
        self.iddiag_path = iddiag_path
        self.traj_csv_path = traj_csv_path
        self.traj_xlsx_path = traj_xlsx_path

        # 加载并处理数据
        self.df_demo = self._load_demo_data()
        self.df_traj = self._load_and_merge_trajectory_data()

    def _load_demo_data(self):
        """加载被试基本信息数据"""
        print("正在加载被试基本信息数据...")
        return pd.read_csv(self.iddiag_path)

    def _process_time_column(self, df):
        """处理时间列，统一两种时间格式"""
        try:
            df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S')
        except:
            print("第一种时间格式解析失败，尝试第二种格式")
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S:%f')

        def redistribute_time(group):
            time_counts = group['time'].value_counts()
            duplicate_times = time_counts[time_counts > 1].index

            for time_val in duplicate_times:
                dup_mask = group['time'] == time_val
                n_duplicates = sum(dup_mask)

                if n_duplicates > 1:
                    time_increment = timedelta(seconds=1) / n_duplicates
                    new_times = [time_val + i * time_increment for i in range(n_duplicates)]
                    group.loc[dup_mask, 'time'] = new_times

            return group
        
        return df.groupby('evaluation_id', group_keys=False).apply(redistribute_time)

    def _process_duplicate_times(self, group):
        """处理重复时间戳"""
        time_counts = group['time'].value_counts()
        duplicate_times = time_counts[time_counts > 1].index

        for time_val in duplicate_times:
            dup_mask = group['time'] == time_val
            n_duplicates = sum(dup_mask)

            if n_duplicates > 1:
                time_increment = pd.Timedelta(minutes=1) / n_duplicates
                new_times = [time_val + i * time_increment for i in range(n_duplicates)]
                group.loc[dup_mask, 'time'] = new_times

        return group

    def _load_and_merge_trajectory_data(self):
        """加载并合并轨迹数据"""
        print("正在加载轨迹数据...")
        
        # 加载CSV轨迹数据
        if os.path.exists(self.traj_csv_path):
            df_traj1 = pd.read_csv(self.traj_csv_path, engine='python')
            df_traj1 = df_traj1.drop('create_time', axis=1) 
            df_traj1['time'] = pd.to_datetime(df_traj1['time'], format='%Y/%m/%d %H:%M')
            df_traj1 = df_traj1.groupby('evaluation_id', group_keys=False).apply(self._process_duplicate_times)
            df_traj1['timestamp'] = df_traj1['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_csv_path}")

        # 加载XLSX轨迹数据
        if os.path.exists(self.traj_xlsx_path):
            df_traj2 = pd.read_excel(self.traj_xlsx_path)
            df_traj2 = df_traj2.drop('create_time', axis=1) 
            df_traj2 = self._process_time_column(df_traj2)
            df_traj2['timestamp'] = df_traj2['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_xlsx_path}")

        # 合并轨迹数据
        df_traj = pd.concat([df_traj1, df_traj2], ignore_index=True)
        print(f"合并后的轨迹数据量: {len(df_traj)}条记录")

        # 预处理时间列
        df_traj['time'] = pd.to_numeric(df_traj['time'], errors='coerce')
        df_traj = df_traj.dropna(subset=['time'])  # 删除无效时间记录
        return df_traj

    def _calculate_curvature(self, group):
        """计算曲率相关特征"""
        group = group.sort_values('timestamp')
        
        x = group['ex'].values
        y = group['ey'].values
        t = group['timestamp'].values
        
        dx_dt = np.gradient(x, t)
        dy_dt = np.gradient(y, t)
        
        d2x_dt2 = np.gradient(dx_dt, t)
        d2y_dt2 = np.gradient(dy_dt, t)
        
        eps = 1e-10
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt**2 + dy_dt**2)**1.5 + eps)
        
        return pd.Series({
            'mean_curvature': float(np.nanmean(curvature)),
            'max_curvature': float(np.nanmax(curvature)),
            'curvature_std': np.nanstd(curvature),
            'high_curvature_points': (curvature > 0.5).sum()
        })

    def _calculate_trajectory_features(self, group):
        """计算轨迹特征"""
        group = group.sort_values('time')
        
        dx = group['ex'].diff()
        dy = group['ey'].diff()
        
        distance = np.sqrt(dx**2 + dy**2)
        time_diff = group['time'].diff()
        speed = distance / (time_diff.replace(0, np.nan) + 1e-6)
        
        curvature_features = self._calculate_curvature(group)
        l = len(group)
        
        if len(group) > 2:
            acceleration = np.diff(speed) / (time_diff[1:] + 1e-6)
            jerk = np.diff(acceleration) / (time_diff[2:] + 1e-6)
            directions = np.arctan2(dy, dx)
            direction_changes = np.sum(np.abs(np.diff(directions)) > np.pi/4)
        else:
            acceleration = jerk = np.array([0])
            direction_changes = 0
        
        total_distance = np.sum(distance)
        straight_distance = np.sqrt((group['ex'].iloc[-1] - group['ex'].iloc[0])**2 + 
                                   (group['ey'].iloc[-1] - group['ey'].iloc[0])**2)
        
        pause_mask = (speed < 0.1) & (distance < 1)
        pause_durations = time_diff[pause_mask]

        base_features = pd.Series({
            'mean_speed': float(np.nanmean(speed)),
            'std_speed': float(np.nanstd(speed)),
            'total_distance': float(np.nansum(distance)),
            'total_time': float(np.nansum(time_diff)),
            'pause_count': int((speed < 0.1).sum()),
            'max_speed': float(np.nanmax(speed)),
            'min_speed': float(np.nanmin(speed)),
            'speed_variation': float(np.nanstd(speed) / (np.nanmean(speed) + 1e-6)),
            'point_count': int(l),
            'complexity_ratio': total_distance / (straight_distance + 1e-6),
            'direction_changes': direction_changes,
            'mean_acceleration': np.nanmean(acceleration) if len(acceleration) > 0 else 0,
            'jerk_std': np.nanstd(jerk) if len(jerk) > 0 else 0,
            'max_pause_duration': np.max(pause_durations) if len(pause_durations) > 0 else 0,
            'pause_time_ratio': np.sum(pause_durations) / np.sum(time_diff) if np.sum(time_diff) > 0 else 0,
            # 'evaluation_id': group['evaluation_id'].iloc[0],
        })
        return pd.concat([base_features, curvature_features])

    def _extract_features(self):
        """从轨迹数据计算特征"""
        traj_features = self.df_traj.groupby('evaluation_id').apply(self._calculate_trajectory_features)
        traj_features = traj_features.reset_index()
        return traj_features

    def get_feature_matrix_and_target(self):
        """提取特征矩阵和目标变量"""
        # 先重置索引，防止evaluation_id作为索引

        if 'evaluation_id' in self.df_demo.index.names:
            self.df_demo = self.df_demo.reset_index(drop=True)  # 重设索引


        print("\n正在合并数据集...")
        traj_features = self._extract_features()

            # 重设索引，防止冲突
        if 'evaluation_id' in traj_features.index.names:
            traj_features = traj_features.reset_index(drop=True)
        df = pd.merge(self.df_demo, traj_features, left_on='evaluation_id', right_on='evaluation_id')
        print(df.head())

        # 定义目标变量
        df['cognitive_impairment'] = (df['moca_score'] <= 25).astype(int)

        # 处理point_count列
        df['point_count'] = pd.to_numeric(df['point_count'], errors='coerce').fillna(0).astype('int64')

        # 选择特征列
        features = [
            'age', 'edu_year', 'habit1',  # 人口统计'edu_year','sex'
            'mean_speed', 'std_speed', 'total_distance',  # 轨迹特征
            'total_time', 'pause_count', 'speed_variation', 'point_count',
            'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points',  # 曲率特征
            'complexity_ratio', 'direction_changes', 'mean_acceleration',
            'jerk_std', 'max_pause_duration', 'pause_time_ratio', 'evaluation_id',
        ]

        X = df[features]
        y = df['cognitive_impairment']
        
        return X, y

def train(model, elogger, train_loader, val_loader, test_loader, epochs, batch_size, lr=1e-3, device=None):
    """
    训练二分类 DeepTTE 模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training", leave=False)
        for attr, traj, labels, driver_ids in pbar:
            attr = {k: v.to(device) for k, v in attr.items()}
            traj = {k: v.to(device) for k, v in traj.items()}
            driver_ids = driver_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            logits = model(attr, traj, driver_ids)
            loss = criterion(logits, labels)
            # backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / len(train_loader)

        # ===== 验证阶段 =====
        val_acc, val_f1 = evaluate(model, val_loader, device)

        elogger.log(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            elogger.log(" Saved new best model")

    # ===== 测试阶段 =====
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc, test_f1 = evaluate(model, test_loader, device)
    elogger.log(f"Test Accuracy={test_acc:.4f}, Test F1={test_f1:.4f}")
    print(f"\n Final Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}, Best Val Accuracy: {best_val_acc:.4f}")
    elogger.close()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for attr, traj, labels, driver_ids in loader:
        attr = {k: v.to(device) for k, v in attr.items()}
        traj = {k: v.to(device) for k, v in traj.items()}
        driver_ids = driver_ids.to(device)
        labels = labels.to(device)

        logits = model(attr, traj, driver_ids)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1


def tuple_to_json(tuple_data, output_file):
    """
    将元组数据转换为JSON格式并写入文件
    
    Args:
        tuple_data: 待写入的元组（支持嵌套元组，如 (1, "a", (3.14, True))）
        output_file: 输出JSON文件路径（如 "output.json"）
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 1. 递归将元组转换为列表（JSON不支持元组，需统一为数组类型）
        def convert_tuple_to_list(data):
            if isinstance(data, tuple):
                # 若为元组，递归转换每个元素后转为列表
                return [convert_tuple_to_list(item) for item in data]
            elif isinstance(data, list):
                # 若为列表，直接递归转换内部元素（处理嵌套元组）
                return [convert_tuple_to_list(item) for item in data]
            elif isinstance(data, dict):
                # 若为字典，递归转换值（键通常为字符串，无需处理）
                return {key: convert_tuple_to_list(value) for key, value in data.items()}
            else:
                # 其他基础类型（int/str/float/bool/None）直接返回
                return data
        
        # 2. 执行转换：元组 → 列表（JSON兼容格式）
        json_compatible_data = convert_tuple_to_list(tuple_data)
        
        # 3. 写入JSON文件（ensure_ascii=False支持中文，indent=2格式化输出）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_compatible_data, f, ensure_ascii=False, indent=2)
        
        print(f"元组已成功写入JSON文件：{output_file}")
        return True
    
    except FileNotFoundError:
        print(f"错误：输出路径不存在 → {output_file}")
        return False
    except PermissionError:
        print(f"错误：无权限写入文件 → {output_file}")
        return False
    except Exception as e:
        print(f"写入JSON时发生未知错误：{str(e)}")
        return False

def convert_tuple_to_list(data):
    if isinstance(data, tuple):
        # 若为元组，递归转换每个元素后转为列表
        return [convert_tuple_to_list(item) for item in data]
    elif isinstance(data, list):
        # 若为列表，直接递归转换内部元素（处理嵌套元组）
        return [convert_tuple_to_list(item) for item in data]
    elif isinstance(data, dict):
        # 若为字典，递归转换值（键通常为字符串，无需处理）
        return {key: convert_tuple_to_list(value) for key, value in data.items()}
    else:
        # 其他基础类型（int/str/float/bool/None）直接返回
        return data
    
def write_dataset_to_json(dataset, file_path):
    """
    新建或重建JSON文件，并将dataset中的所有字典元素写入（格式为JSON数组）
    
    Args:
        dataset: 可迭代对象，每个元素为字典
        file_path: 目标JSON文件路径
    """
    # 收集所有数据到列表（确保JSON格式正确）
    all_data = []
    for item in dataset:
        # 验证元素是否为字典（非字典元素会被过滤，可选操作）
        if isinstance(item, dict):
            all_data.append(item)
        else:
            print(f"警告：跳过非字典元素 {item}")
    
    # 写入文件（若文件存在则覆盖重建）
    with open(file_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保留中文，indent=2 格式化输出
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"已成功将 {len(all_data)} 条数据写入 {file_path}（文件已重建）")

def main():

    kernel_size = 3
    num_filter = 64     #控制 卷积特征提取层（GeoConv） 的通道数增大 num_filter → 模型容量变强，能提取更复杂的轨迹特征，但显著增加显存和训练时间；减小 num_filter → 模型更轻，速度快但特征表达能力变弱。
    pooling_method = 'attention'
    num_fc_layers = 2  #控制 分类器（最终全连接部分） 的层数 增加 分类器表达力增强，可以更好地处理非线性关系；但参数量变多，容易过拟合；减少模型更简单，泛化更稳，但可能欠拟合。
    hidden_size = 128
    alpha = 0.3
    epochs = 100
    batch_size = 32
    file_path="data/dataset_output.json"  # 目标文件路径

    train_loader, val_loader, test_loader = get_loader(file_path, batch_size)   #输出符合预期

    model = models.DeepTTE.Net(
        num_classes = 2, 
        kernel_size = kernel_size, 
        num_filter = num_filter, 
        pooling_method = pooling_method, 
        hidden_size = hidden_size, 
        num_fc_layers= num_fc_layers
    )
   
    elogger = logger.Logger("run_log")
    
    print("模型创建成功,开始训练...")
    train(model, elogger, train_loader, val_loader, test_loader, epochs, batch_size)   #elogger是干什么的？日志文件路径  在这里需要将
    return

if __name__ == "__main__":
    main()