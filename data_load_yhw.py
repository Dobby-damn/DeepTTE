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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from datetime import timedelta

import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
#import data_loader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
config = json.load(open('./config.json', 'r'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

import json

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
        # self.content = list(map(lambda x: json.loads(x), self.content))
        # 计算每条轨迹的长度（坐标点数量）
        self.lengths = list(map(lambda x: len(x['ex']), content))
        for item in content:
            attr = {
                'driverID': torch.tensor(item['driverID'], dtype=torch.long),
                'num_features': torch.tensor([
                    item['time'],
                    item['dist'],
                    item['pause_count'],
                    item['mean_speed'],
                    item['curvature_std']
                ], dtype=torch.float32)
            }
            # ---- 轨迹特征（序列部分）----
            ex = torch.tensor(item['ex'], dtype=torch.float32)
            ey = torch.tensor(item['ey'], dtype=torch.float32)

            traj = {
                'lngs': ex,   # 经度序列
                'lats': ey,   # 纬度序列
                'lens': torch.tensor(len(ex), dtype=torch.long)
            }
            label = torch.tensor(item['label'], dtype=torch.long)
            driver_id = item['driverID']
            self.data.append({'attr': attr, 'traj': traj, 'label': label, 'driver_id': driver_id})
          


    def __getitem__(self, idx):
        """获取单条轨迹数据"""
        print("这里是getitem函数" + "-----------------")
        return self.data[idx]
        item = self.data[idx]

        # ---- 属性特征（非序列部分）----
        attr = {
            'driverID': torch.tensor(item['driverID'], dtype=torch.long),
            'num_features': torch.tensor([
                item['time'],
                item['dist'],
                item['pause_count'],
                item['mean_speed'],
                item['curvature_std']
            ], dtype=torch.float32)
        }
        # ---- 轨迹特征（序列部分）----
        ex = torch.tensor(item['ex'], dtype=torch.float32)
        ey = torch.tensor(item['ey'], dtype=torch.float32)

        traj = {
            'lngs': ex,   # 经度序列
            'lats': ey,   # 纬度序列
            'lens': torch.tensor(len(ex), dtype=torch.long)
        }
        label = torch.tensor(item['label'], dtype=torch.long)
        driver_id = item['driverID']

        return {'attr': attr, 'traj': traj, 'label': label, 'driver_id': driver_id}

    def __len__(self):
        """返回数据集总大小"""
        return len(self.data)
'''
# def collate_fn(data):
#     """自定义批处理函数，用于填充变长序列和归一化数据
#     Args:
#         data: 一个batch的原始数据列表（每个元素是MySet返回的一条轨迹）
#     Returns:
#         attr: 静态属性字典（标准化后的数值特征和类别ID）
#         traj: 轨迹数据字典（填充后的序列和原始长度）
#     """
#     # 需要统计归一化的静态属性
#     stat_attrs = ['dist', 'time']   # 总距离和总时间
#     # 类别型ID属性
#     info_attrs = ['driverID', 'mean_speed', 'pause_count', 'curvature_std']
#     # 轨迹序列属性
#     traj_attrs = ['ex', 'ey']

#     attr, traj = {}, {}
#     # 获取当前batch中各轨迹的实际长度
#     lens = np.asarray([len(item['ex']) for item in data])
#     # 处理静态数值属性（归一化）
#     for key in stat_attrs:
#         x = torch.FloatTensor([item[key] for item in data])
#         attr[key] = utils.normalize(x, key)   # 均值方差归一化
    
#     # 处理类别型ID属性将其变为长张量 info_attrs=['driverID', 'mean_speed', 'pause-count', 'curvature_std']
#     for key in info_attrs:
#         attr[key] = torch.LongTensor([item[key] for item in data])

#     # 处理轨迹序列数据（填充变长序列）
#     for key in traj_attrs:
#         # pad to the max length 创建填充矩阵（batch_size x max_len）
#         # seqs = np.asarray([item[key] for item in data])
#         # mask = np.arange(lens.max()) < lens[:, None]   # 生成掩码矩阵（标记有效数据位置）
#         # padded = np.zeros(mask.shape, dtype = np.float32)   # 填充实际数据
#         # padded[mask] = np.concatenate(seqs)   
#         # python2/3兼容性问题，直接构建填充后的张量
#         seqs = [item[key] for item in data]
#         max_len = max(len(seq) for seq in seqs)
#         padded = torch.zeros(len(data), max_len, dtype=torch.float32)
#         for i, seq in enumerate(seqs):
#             padded[i, :len(seq)] = torch.FloatTensor(seq)
#         # 对数值型序列进行归一化
#         if key in ['ex', 'ey']:
#             padded = utils.normalize(padded, key)

#         # padded = torch.from_numpy(padded).float()
#         traj[key] = padded
#     # 保存原始长度信息（用于后续处理）
#     lens = [len(item['ex']) for item in data]
#     assert all(l > 0 for l in lens), "存在零长度序列"
#     # lens = lens.tolist()
#     traj['lens'] = torch.tensor(lens, dtype=torch.int64)  # 确保是int64张量

#     return attr, traj
'''
def collate_fn(batch):
    print('这里是批处理函数'+'-----------------')
    """
    将 batch 样本打包为模型可接受的格式
    输出:
        attr: dict{key: (B, 1) Tensor}
        traj: dict{'ex': (B,T,1), 'ey': (B,T,1), 'mask': (B,T)}
        labels: (B,)
    """
    # === 属性部分 ===
    attr_keys = batch[0]['attr'].keys()
    print("attr_keys:", attr_keys)
    attr = {
        k: torch.tensor([b['attr'][k] for b in batch], dtype=torch.float32).unsqueeze(1)
        for k in attr_keys
    }

    # === 轨迹部分 ===
    ex_list = [torch.tensor(b['traj']['ex'], dtype=torch.float32).unsqueeze(-1) for b in batch]
    ey_list = [torch.tensor(b['traj']['ey'], dtype=torch.float32).unsqueeze(-1) for b in batch]

    ex_pad = pad_sequence(ex_list, batch_first=True)  # (B, T, 1)
    ey_pad = pad_sequence(ey_list, batch_first=True)  # (B, T, 1)

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
    print(456)
    print(attr['driverID'].shape, traj['ex'].shape, labels.shape, driver_ids.shape)
    print(789)

    return attr, traj, labels, driver_ids
class BatchSampler:
    """自定义批次采样器，优化变长序列的批处理效率"""
    def __init__(self, dataset, batch_size):
        """生成批次索引，按长度排序减少填充量"""
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.dataset.lengths
        self.indices = range(self.count)

    def __iter__(self):
        '''
        1. 将数据分成大块（每块大小=batch_size*100）
        2. 每块内按轨迹长度降序排序
        3. 从排序后的块中按批次取索引
        '''
        # 步骤1：将索引转为numpy数组（方便后续打乱和切片），创建副本避免修改原数据
        indices = np.array(self.indices)  
        # 步骤2：全局打乱索引（保证随机性，避免按原始顺序采样的偏差）
        np.random.shuffle(indices)

        # 步骤3：定义“大块”大小（batch_size*100，平衡随机性和排序效果）
        chunk_size = self.batch_size * 100
        # 计算总共有多少个大块（向上取整，避免遗漏最后几个样本）
        chunks = (self.count + chunk_size - 1) // chunk_size

        # 步骤4：每个大块内按轨迹长度降序排序（减少同批次内的长度差异）
        for i in range(chunks):
            # 提取当前大块的索引（切片：从i*chunk_size到(i+1)*chunk_size）
            partial_indices = indices[i * chunk_size: (i + 1) * chunk_size]
            # 按轨迹长度排序：key=lambda x: self.lengths[x]（x是样本索引，取该样本的长度）
            # reverse=True：降序（长轨迹和长轨迹一组，短轨迹和短轨迹一组）
            partial_indices = sorted(partial_indices, 
                        key=lambda x: self.lengths[x], 
                        reverse=True)
            # 将排序后的索引放回原indices数组
            indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # 步骤5：按批次分割排序后的索引，生成最终批次
        # 计算总批次数（向上取整）
        batches = (self.count - 1 + self.batch_size) // self.batch_size
        # 遍历每个批次，返回该批次的索引列表
        for i in range(batches):
            yield indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

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
    print("即将执行myset函数")
    dataset = MySet(input_file=file)
    print(dataset.__getitem__(0))
    print("myset函数执行完毕")

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
    print("训练集分割")
    train_loader = DataLoader(
        dataset = train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=batch_size,  # 实际批次大小由batch_sampler控制
    #     collate_fn=collate_fn,  # 使用自定义批处理函数
    #     num_workers=4,
    #     #batch_sampler=BatchSampler(train_set, batch_size),
    #     pin_memory=True
    # )
    print("验证集分割")
    # 验证集 DataLoader
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        #batch_sampler=BatchSampler(val_set, batch_size),
        pin_memory=True
    )
    print("测试集分割")
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
        # 保留 diagnose_encoded == 0 或 1 的记录
        #iddiag = iddiag[iddiag["diagnose_encoded"].isin([0, 1])].copy()
        # self.label_encoder = LabelEncoder()
        # iddiag["diagnose_encoded"] = self.label_encoder.fit_transform(iddiag["diagnose"])
    
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
        dis, time, driverID, pause_count, mean_speed, curvature_std, ex, ey, label = (self.samples[idx]['dis'],
                                                                                   self.samples[idx]['time'],
                                                                                   self.samples[idx]['driverID'],
                                                                                   self.samples[idx]['pause_count'],
                                                                                   self.samples[idx]['mean_speed'],
                                                                                   self.samples[idx]['curvature_std'],
                                                                                   self.samples[idx]['ex'],
                                                                                   self.samples[idx]['ey'],
                                                                                   self.samples[idx]['label'])
        #coords, static_feat, label = self.samples[idx]
        #return dis, time, driverID, pause_count, mean_speed, curvature_std, ex, ey, label


# def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
#     """Custom collate_fn to pad variable‑length sequences within each mini‑batch."""

#     # sequences, labels = zip(*batch)
#     # lengths = torch.tensor([len(seq) for seq in sequences])
#     # padded_sequences = pad_sequence(sequences, batch_first=True)  # zero‑pad shorter seqs
#     # return padded_sequences, lengths, torch.tensor(labels)
#     coords, static_feats, labels = zip(*batch)
#     lengths = torch.tensor([len(c) for c in coords], dtype=torch.long)

#     coords_padded = nn.utils.rnn.pad_sequence(coords, batch_first=True)
#     static_feats = torch.stack(static_feats)
#     labels = torch.tensor(labels, dtype=torch.long)
#     return coords_padded, lengths, static_feats, labels


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
            elogger.log("✅ Saved new best model")

    # ===== 测试阶段 =====
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc, test_f1 = evaluate(model, test_loader, device)
    elogger.log(f"Test Accuracy={test_acc:.4f}, Test F1={test_f1:.4f}")
    print(f"\n📊 Final Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
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
# def train(model, elogger, train_loader, val_loader, test_loader, epochs=10, batch_size=32):
#     """训练模型的主函数。
#     Args:
#         model: 待训练的模型
#         elogger: 日志记录器
#         train_loader: 训练集
#         val_loader: 验证集
#         test_loader: 测试集
#         epochs: 训练轮数
        
#     """
#     # record the experiment setting
#     elogger.log(str(model))  #记录模型结构
#     # elogger.log(str(args._get_kwargs())) # 记录使用的超参数

#     # 设置模型为训练模式
#     model.train()

#     use_cuda = torch.cuda.is_available()
#     if use_cuda:
#         print("使用 GPU 进行训练")
#         model.cuda()

#     # 使用 Adam 优化器 一种广泛使用的深度学习优化算法 旨在通过计算梯度的一阶矩估计和二阶矩估计来调整每个参数的学习率，从而实现更高效的网络训练。
#     optimizer = optim.Adam(model.parameters(), lr = 1e-3)

#     # 遍历每个 epoch
#     for epoch in range(epochs):
#         print ('Training on epoch {}'.format(epoch))
#         # 遍历训练集中的每个文件
        
#         # print ('Train on file {}'.format(input_file))

#         # data loader, return two dictionaries, attr and traj获取数据加载器（返回属性 attr 和轨迹 traj）批处理
#         #data_iter = get_loader(file, batch_size)

#         running_loss = 0.0   # 累计损失

#         # 遍历每个批次
#         for idx, (attr, traj) in enumerate(train_loader):  #此处要求dataset为字典格式。dataloader格式
#             # transform the input to pytorch variable  将数据转换为 PyTorch Variable（兼容旧版本）
#             # attr, traj = utils.to_var(attr), utils.to_var(traj)
#             if use_cuda:
#                 attr = {k: v.cuda() if torch.is_tensor(v) else v for k, v in attr.items()}
#                 traj = {k: v.cuda() if torch.is_tensor(v) else v for k, v in traj.items()}
#             if torch.__version__ < '0.4':
#                 attr = {k: Variable(v) for k, v in attr.items()}
#                 traj = {k: Variable(v) for k, v in traj.items()}

#             # 前向传播并计算损失
#             '''此处应是前向传播forward_, loss = model.eval_on_batch(attr, traj, config)'''
            
#             # print('traj:', traj['lens'])
#             print("attr.keys:{}".format(attr.keys()))
#             print("config:{}".format(config))
#             _, loss = model.eval_on_batch(attr, traj, config)
#             '''traj: tensor([125, 122, 114, 111, 110, 108, 107, 104, 104, 103], device='cuda:0')
#             Geo输出： torch.Size([10, 123, 33])
#             属性张良形状 torch.Size([10, 1, 28])
#             扩展属性张量形状 torch.Size([10, 123, 28])
#             拼接后的卷积张量形状 torch.Size([10, 123, 61])
#             '''
            

#             # update the model
#             # 反向传播和优化
#             optimizer.zero_grad()
#             '''loss.backward未定义'''
#             loss.backward()
#             optimizer.step()

#             # 更新累计损失（注意：loss.data[0] 是旧写法，新版本应为 loss.item()）
#             # 损失值获取兼容性
#             if torch.__version__ < '0.4':
#                 running_loss += loss.data[0]
#             else:
#                 running_loss += loss.item()

#             # 打印训练进度
#             print ('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(train_loader), running_loss / (idx + 1.0))),
#             print
#             # 记录当前文件的训练损失
#             elogger.log('Training Epoch {}, Loss {}'.format(epoch, running_loss / (idx + 1.0)))

#         # evaluate the model after each epoch 每个 epoch 结束后在验证集上评估
#         evaluate(model, elogger, val_loader, save_result = True)

#         # save the weight file after each epoch 保存模型权重（文件名包含时间戳）
#         weight_name = '{}_{}'.format('./logs/run_log.log', str(datetime.datetime.now()))
#         elogger.log('Save weight file {}'.format(weight_name))
#         torch.save(model.state_dict(), './saved_weights/' + weight_name)

def write_result(fs, pred_dict, attr):
    """将预测结果写入文件。
    Args:
        fs: 文件句柄
        pred_dict: 包含预测值和标签的字典
        attr: 属性数据（如 dateID, timeID 等）
    """
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]

def evaluate(model, elogger, val_loader, save_result = False):
    """评估模型性能。
    Args:
        model: 待评估的模型
        elogger: 日志记录器
        files: 评估文件列表
        save_result: 是否保存预测结果
    """
    model.eval()
    if save_result:
        fs = open('./result/deeptte.res', 'w')

    
    running_loss = 0.0
    # data_iter = data_loader.get_loader(input_file, args.batch_size)


    for idx, (attr, traj) in enumerate(val_loader):
        attr, traj = utils.to_var(attr), utils.to_var(traj)

        pred_dict, loss = model.eval_on_batch(attr, traj, config)
        
        if save_result: write_result(fs, pred_dict, attr)

        # running_loss += loss.data[0] #  旧版写法，已经失效
        if torch.__version__ < '0.4':
            running_loss += loss.data[0]
        else:
            running_loss += loss.item()

    print ('Evaluate on loss {}'.format(running_loss / (idx + 1.0)))
    elogger.log('Evaluate, Loss {}'.format(running_loss / (idx + 1.0)))

    if save_result: fs.close()


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
    '''
    iddiag_path = r"D:/Code/DeepTTE/LSTM/iddiag.csv"
    traj_csv_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(1).csv"
    traj_xlsx_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(2).xlsx"

    kernel_size = 3
    num_filter = 32
    pooling_method = 'attention'
    num_final_fcs = 3
    final_fc_size = 128
    alpha = 0.3
    epochs = 10
    batch_size = 32

    # 获取属性特征，edu_year, sex, age ,mean_speed etlc.
    extractor = TrajectoryFeatureExtractor(iddiag_path, traj_csv_path, traj_xlsx_path)
    X, y = extractor.get_feature_matrix_and_target()
    
    print(y.head())
    #现在，我们的X当中包含着可以用做attr的属性,我需要把他变成字典，下面的dataset也是

    # ----------------------------------- Dataset ---------------------------------------
    dataset = TrajectoryDataset(iddiag_path, traj_csv_path, traj_xlsx_path, X)
    print(dataset[0]) # 过来的是一个字典
    print(11)

    # 调用函数：重建文件并写入所有元素
    write_dataset_to_json(
        dataset=dataset,
        file_path="data/dataset_output.json"  # 目标文件路径
    )'''

    kernel_size = 3
    num_filter = 64     #控制 卷积特征提取层（GeoConv） 的通道数增大 num_filter → 模型容量变强，能提取更复杂的轨迹特征，但显著增加显存和训练时间；减小 num_filter → 模型更轻，速度快但特征表达能力变弱。
    pooling_method = 'attention'
    num_fc_layers = 2  #控制 分类器（最终全连接部分） 的层数 增加 分类器表达力增强，可以更好地处理非线性关系；但参数量变多，容易过拟合；减少模型更简单，泛化更稳，但可能欠拟合。
    hidden_size = 128
    alpha = 0.3
    epochs = 10
    batch_size = 32
    file_path="data/dataset_output.json"  # 目标文件路径

    # print(dataset.__len__())
    # read_json_array(file_path)  # 通常用.jsonl作为扩展名
    # return


    train_loader, val_loader, test_loader = get_loader(file_path, batch_size)   #输出符合预期
    #原来的代码种有训练测试与验证

    #first_batch = next(iter(train_loader))   # File "d:\Code\DeepTTE\utils.py", line 24, in normalize    mean = config[key + '_mean']    KeyError: 'dis_mean'
    # 打印第一个批次的内容（根据你的数据结构，可能是元组、字典等）
    #print("first_batch:{}".format(first_batch))


   
    # kernel_size = 3, num_filter = 32, pooling_method = 'attention', num_final_fcs = 3, final_fc_size = 128, alpha = 0.3
    #model = models.DeepTTE.Net(kernel_size = kernel_size, num_filter = num_filter, pooling_method = pooling_method, num_final_fcs = num_final_fcs, final_fc_size = final_fc_size, alpha = alpha)

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
    train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    # elif args.task == 'test':
    #     # load the saved weight file
    #     model.load_state_dict(torch.load(args.weight_file))
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     # 在测试集上评估并保存结果
    #     evaluate(model, elogger, config['test_set'], save_result = True)
    
    labels = [lbl for *_, lbl in dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(
            len(dataset)), 
            test_size = 0.2, 
            random_state = 5, 
            stratify = labels
    )
    '''现在，dataset中包含着一切数据，需要配知道它什么样子，拆除分割将他变为Deep TTE中的attr/traj格式
    在这里dataset是一个类对象，'''
    print (dataset.head())

if __name__ == "__main__":
    main()