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
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler, random_split, Subset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from datetime import timedelta
from torch.optim import Adam
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from evaluation.metrics import compute_binary_metrics

#############################################dataloader部分#############################################
class ParquetDataset(Dataset):
    META_FIELDS = [
        'evaluation_id', 'driverID', 'record_id', 'diagnose', 'age', 'sex', 'hand',
        'edu_year', 'education', 'edugrade', 'occupation', 'hospital', 'center', 'site',
        'gameDuration', 'clickCount', 'correctConnections', 'incorrectConnections',
        'noTouchCount', 'connTime', 'total_time', 'total_distance', 'mean_speed',
        'std_speed', 'pause_count', 'pause_time_ratio'
    ]

    def __init__(self, file_path=None, normalize=True, dataframe=None, normalizer=None, fit_normalizer=True):
        """
        Args:
            file_path: Parquet 文件路径
            mode: 'train' (开启数据增强) 或 'test'/'val' (关闭数据增强)
            normalize: 是否执行 Z-score 归一化
        读取 Parquet 文件并构建数据集
        """
   
        # 1. 读取 Parquet
        # engine='pyarrow' 或 'fastparquet'，通常默认即可
        if dataframe is not None:
            source_df = dataframe.copy()
            source_name = "<dataframe>"
        else:
            source_df = pd.read_parquet(file_path)
            source_name = file_path

        # Keep raw demographics for subgroup analysis. The model-facing copy may
        # be normalized below, so it cannot be used for clinically meaningful bins.
        self.raw_df = source_df.copy()
        self.df = source_df.drop(columns=['moca_score'], errors='ignore').copy()
        
        print(f"Loaded dataset from {source_name}: {len(self.df)} samples")
        print(f"Columns: {list(self.df.columns)}")
        
        # 2. 预处理
        # 确保 ex, ey 是 list 类型 (有些 parquet 存的是 string 或 numpy array)
        # 如果存的是 numpy array，tolist() 是必要的
        # 如果存的是 string (如 "[1.1, 2.2]"), 需要 json.loads 解码 (视你存的方式而定)
        self.exclude_cols = ['ex', 'ey', 'label', 'driverID', 'evaluation_id', 'diagnose', 
                             'sex', 'hand'] 
        self.cont_cols = [c for c in self.df.columns if c not in self.exclude_cols and pd.api.types.is_numeric_dtype(self.df[c])]
        print(f"Detected {len(self.cont_cols)} continuous columns for normalization: {self.cont_cols[:5]}...")
        self.normalize = normalize
        if self.normalize:
            self._apply_normalization(normalizer=normalizer, fit=fit_normalizer)
            
    def _apply_normalization(self, normalizer=None, fit=True):
        print("Applying normalization (Z-score)...")

        # Static attributes: fit on the training fold, then transform validation/test folds.
        self.df[self.cont_cols] = self.df[self.cont_cols].fillna(0)

        if fit or normalizer is None:
            scaler = StandardScaler()
            if self.cont_cols:
                self.df[self.cont_cols] = scaler.fit_transform(self.df[self.cont_cols])
            normalizer = {"static_scaler": scaler, "cont_cols": list(self.cont_cols)}
        else:
            scaler = normalizer["static_scaler"]
            expected_cols = normalizer.get("cont_cols", self.cont_cols)
            if list(expected_cols) != list(self.cont_cols):
                raise ValueError(f"Normalizer columns do not match dataset columns: {expected_cols} != {self.cont_cols}")
            if self.cont_cols:
                self.df[self.cont_cols] = scaler.transform(self.df[self.cont_cols])
        print("Static attribute normalization complete.")

        # 2. 轨迹坐标归一化 (计算全局均值和标准差)
        # 注意：ex 和 ey 是列表，不能直接用 scaler。我们需要展平所有轨迹点来计算。
        # 这里为了效率，我们先随机采样一部分数据估算，或者全量计算（取决于数据量）
        
        if fit or normalizer is None or "ex_mean" not in normalizer:
            # Fit coordinate normalization on the current dataframe, normally the training fold.
            all_ex = np.concatenate(self.df['ex'].values)
            all_ey = np.concatenate(self.df['ey'].values)

            self.ex_mean = all_ex.mean()
            self.ex_std = all_ex.std() + 1e-6

            self.ey_mean = all_ey.mean()
            self.ey_std = all_ey.std() + 1e-6
            normalizer.update(
                {
                    "ex_mean": self.ex_mean,
                    "ex_std": self.ex_std,
                    "ey_mean": self.ey_mean,
                    "ey_std": self.ey_std,
                }
            )
            del all_ex, all_ey
        else:
            self.ex_mean = normalizer["ex_mean"]
            self.ex_std = normalizer["ex_std"]
            self.ey_mean = normalizer["ey_mean"]
            self.ey_std = normalizer["ey_std"]
        self.normalizer = normalizer

        print("Trajectory normalization parameters ready:")
        print(f"   ex: mean={self.ex_mean:.4f}, std={self.ex_std:.4f}")
        print(f"   ey: mean={self.ey_mean:.4f}, std={self.ey_std:.4f}")

    def get_normalizer(self):
        return getattr(self, "normalizer", None)

    def _metadata_from_row(self, row):
        meta = {}
        for field in self.META_FIELDS:
            if field not in row:
                continue
            value = row[field]
            try:
                if pd.isna(value):
                    value = None
            except ValueError:
                continue
            if hasattr(value, "item"):
                value = value.item()
            meta[field] = value
        if "evaluation_id" not in meta and "driverID" in meta:
            meta["evaluation_id"] = meta["driverID"]
        return meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取 DataFrame 的一行 (Series 对象)
        row = self.df.iloc[idx]
        raw_row = self.raw_df.iloc[idx]
        
        # === 1. 构建 Attr 字典 === 
        attr_dict = {}
        for col_name in self.cont_cols:
            # 这里的 row[col_name] 已经是归一化后的数据了
            attr_dict[col_name] = float(row[col_name])
        # 处理离散特征（不归一化的）
        if 'hand' in row:
            val = row['hand']
            # 判断是否为 NaN (Pandas中 NaN != NaN) 或者使用 pd.isna(val)
            if pd.isna(val):
                attr_dict['hand'] = 0  # <--- 遇到空值，填充默认值 0
            else:
                attr_dict['hand'] = int(val)
        
        # 对 sex 也做同样的保护，防止报错
        if 'sex' in row:
            val = row['sex']
            if pd.isna(val):
                attr_dict['sex'] = 0   # <--- 遇到空值，填充默认值 0
            else:
                attr_dict['sex'] = int(val)
        # === 2. 构建 Traj 字典 ===
        # 获取原始列表
        raw_ex = row['ex'].tolist() if hasattr(row['ex'], 'tolist') else list(row['ex'])
        raw_ey = row['ey'].tolist() if hasattr(row['ey'], 'tolist') else list(row['ey'])

        # === 计算运动学特征 (关键步骤) ===
        # 速度 (一阶差分)
        # np.diff 后长度会少 1，我们需要补一个 0 保持长度一致
        vel_x = np.diff(raw_ex, prepend=raw_ex[0])
        vel_y = np.diff(raw_ey, prepend=raw_ey[0])
        speed = np.sqrt(vel_x**2 + vel_y**2)
        # 加速度 (速度的差分)
        acc_x = np.diff(vel_x, prepend=vel_x[0])
        acc_y = np.diff(vel_y, prepend=vel_y[0])
        acc_mag = np.sqrt(acc_x**2 + acc_y**2) # 加速度大小
        speed = np.log1p(speed) 
        acc_mag = np.log1p(acc_mag)

        # 转换为 numpy 以便进行数学运算
        np_ex = np.array(raw_ex, dtype=np.float32)
        np_ey = np.array(raw_ey, dtype=np.float32)
        # *** 实时应用轨迹归一化 ***
        if self.normalize:
            np_ex = (np_ex - self.ex_mean) / self.ex_std
            np_ey = (np_ey - self.ey_mean) / self.ey_std
            speed = (speed - speed.mean()) / (speed.std() + 1e-6)
            acc_mag = (acc_mag - acc_mag.mean()) / (acc_mag.std() + 1e-6)
            
        # 处理 NaN (轨迹中可能存在无效点)
        np_ex = np.nan_to_num(np_ex, nan=0.0)
        np_ey = np.nan_to_num(np_ey, nan=0.0)
        traj_dict = {
            'ex': np_ex.tolist(),
            'ey': np_ey.tolist(),
            'speed': speed.tolist(),       # 新增
            'acc': acc_mag.tolist()        # 新增
        }
        # === 3. ID 和 Label ===
        # 兼容 driverID 或 evaluation_id
        driver_id = int(row.get('driverID', row.get('evaluation_id', 0)))
        label = int(row['diagnose'])
        
        return {
            'attr': attr_dict,
            'traj': traj_dict,
            'label': label,
            'driverID': driver_id,
            'meta': self._metadata_from_row(raw_row)
        }


def collate_fn(batch):
    """
    将 batch 样本打包为模型可接受的格式
    输出:
        attr: dict{key: (B, 1) Tensor}
        traj: dict{'ex': (B,T,1), 'ey': (B,T,1), 'mask': (B,T)}
        labels: int
        driver_id: int 
    """
    # === 属性部分 ===
    attr_keys = batch[0]['attr'].keys()
    #print("attr_keys:", attr_keys)
    # attr_keys包括[int,[]]
    attr = {}
    for k in attr_keys:
        # 彻底移除 driverID 进网络的可能性
        if k in ( 'driverID', 'ex', 'ey', 'label'): 
            continue 
        attr[k] = torch.tensor([b['attr'][k] for b in batch], dtype=torch.float32)

    # === 轨迹部分 ===
    ex_list = [torch.tensor(b['traj']['ex'], dtype=torch.float32) for b in batch]
    ey_list = [torch.tensor(b['traj']['ey'], dtype=torch.float32) for b in batch]
    speed_list = [torch.tensor(b['traj']['speed'], dtype=torch.float32) for b in batch]
    acc_list = [torch.tensor(b['traj']['acc'], dtype=torch.float32) for b in batch]
    
    # Pad
    speed_pad = pad_sequence(speed_list, batch_first=True)
    acc_pad = pad_sequence(acc_list, batch_first=True)
    ex_pad = pad_sequence(ex_list, batch_first=True)  # (B, T)
    ey_pad = pad_sequence(ey_list, batch_first=True)  # (B, T)

    # Mask
    lengths = [len(b['traj']['ex']) for b in batch]
    max_len = max(lengths)
    mask = torch.zeros(len(batch), max_len)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    traj = {'ex': ex_pad, 'ey': ey_pad, 'speed': speed_pad, 'acc': acc_pad, 'mask': mask}
   

    # === driverID（如果需要）===
    driver_ids = torch.tensor([b['driverID'] for b in batch], dtype=torch.long)

    # === 标签 ===
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    # print(456)
    # # print(attr['driverID'].shape, traj['ex'].shape, labels.shape, driver_ids.shape)
    # for k,v in attr.items():
    #     print(k, v.shape, v.dtype)

    # 仅为了 debug 或 logging 保留 ID，不传入 forward
    meta_ids = [b.get('meta', {'evaluation_id': b['driverID'], 'driverID': b['driverID']}) for b in batch]
    
    return attr, traj, labels, meta_ids

# 构造bucket，样本按照序列长度分组
class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 记录每个样本的长度
        self.lengths = []
        if hasattr(self.data_source, "indices") and hasattr(self.data_source, "dataset"):
            subset_indices = self.data_source.indices
            original_dataset = self.data_source.dataset
        else:
            subset_indices = range(len(self.data_source))
            original_dataset = self.data_source
        for original_idx in subset_indices:
            # 直接访问 DataFrame 获取长度，速度最快
            row = original_dataset.df.iloc[original_idx]
            ex_data = row['ex']
            # 处理可能的类型差异 (list vs numpy vs string)
            # 假设你的 Parquet 读出来 ex 已经是 list 或者 numpy array
            length = len(ex_data) if hasattr(ex_data, '__len__') else 0 
            self.lengths.append(length)
        # self.lengths = [len(seq) for seq, _ in data_source.samples]
        # 2. 生成排序后的【局部索引】 (0 到 len(subset)-1)
        # 关键修改：我们要排序的是 range(len(subset))，而不是原始索引
        local_indices = range(len(self.lengths))
        self.sorted_local_indices = sorted(local_indices, key=lambda i: self.lengths[i])
        # 3. 分桶
        self.buckets = [
            self.sorted_local_indices[i:i + batch_size] 
            for i in range(0, len(self.sorted_local_indices), batch_size)
        ]

        if len(self.buckets) > 0 and len(self.buckets[-1]) < 2:
            # print("⚠️ 警告: 丢弃最后一个大小为 1 的 Batch，防止 BatchNorm 报错")
            self.buckets.pop()  # 移除最后一个桶


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.buckets)
        for bucket in self.buckets:
            yield bucket

    def __len__(self):
        return len(self.buckets)

# 修改 get_loader 函数，确保按 ID 划分
def get_loader(file_path, batch_size, test_ratio=0.2, val_ratio=0.2, num_workers=0, seed=8): # 加载所有数据
    content = ParquetDataset(file_path=file_path, normalize=False)
    raw_df = content.raw_df.copy()


    # 提取所有唯一的 driverID (Subject ID) 
    # 假设数据中每个 dict 都有 'driverID' 字段 
    all_subject_ids = raw_df['evaluation_id'].unique().tolist()
    all_subject_ids.sort() # 保证顺序固定 
    id_labels =[]
    for subj_id in all_subject_ids:
        # 找到这个病人，取他的一条标签
        label = raw_df[raw_df['evaluation_id'] == subj_id]['diagnose'].iloc[0]
        id_labels.append(label)
    # 按 Subject ID 划分 
    train_ids, test_ids = train_test_split(all_subject_ids, test_size=test_ratio, random_state=seed, stratify=id_labels) 
    # 从训练集中再分出验证集 
    train_labels = [id_labels[all_subject_ids.index(i)] for i in train_ids]
    train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio/(1-test_ratio), random_state=seed, stratify=train_labels) 
    print(f"Split Info: Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}") 
    # 根据 ID 构建数据集，并仅使用训练集拟合归一化参数。
    train_df = raw_df[raw_df['evaluation_id'].isin(train_ids)].copy()
    val_df = raw_df[raw_df['evaluation_id'].isin(val_ids)].copy()
    test_df = raw_df[raw_df['evaluation_id'].isin(test_ids)].copy()
    train_set = ParquetDataset(dataframe=train_df, normalize=True, fit_normalizer=True)
    normalizer = train_set.get_normalizer()
    val_set = ParquetDataset(dataframe=val_df, normalize=True, normalizer=normalizer, fit_normalizer=False)
    test_set = ParquetDataset(dataframe=test_df, normalize=True, normalizer=normalizer, fit_normalizer=False)
    # DataLoader 
    train_sampler = BucketBatchSampler(train_set, batch_size) #按照长度分组
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers) 

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_loader, test_loader


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


def train(
    model,
    elogger,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    batch_size,
    lr=1e-3,
    device=None,
    checkpoint_path="best_model.pth",
    evaluate_test=True,
):
    """
    训练二分类 DeepTTE 模型
    """
    patience = 15      # 容忍多少个 epoch 验证集不提升
    patience_counter = 0
    best_val_acc = 0.0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # ===== 统计训练集类别分布（只做一次）=====
    label_counter = Counter()

    for _, _, labels, _ in train_loader:
        label_counter.update(labels.tolist())

    print("Train label distribution:", label_counter)

    # 假设是二分类 0/1
    num_class0 = label_counter[0]
    #print(f"Class 0 samples: {num_class0}") 167
    num_class1 = label_counter[1]
    #print(f"Class 1 samples: {num_class1}") 249

    # 反比权重（常用）
    weights = torch.tensor(
        [1.2, 1.0],
        dtype=torch.float32
    )

    #model.attr_net = model.attr_net.cpu()

    # 损失函数与优化器 加权loss 类别不平衡
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion = FocalLoss(gamma=2)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',        # 因为我们看的是 val_acc
        factor=0.8,        # 每次衰减为原来的 0.8
        patience=10,        # 5 个 epoch 不提升就降 LR
        min_lr=1e-5        # 最小学习率
    )
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        #pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training", leave=False)
        print(f"Epoch {epoch}/{epochs} - Training...")
        for attr, traj, labels, driver_ids in train_loader:
            attr = {k: v.to(device) for k, v in attr.items()}
            traj = {k: v.to(device) for k, v in traj.items()}
           
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            logits = model(attr, traj)

            if torch.isnan(logits).any():
                raise RuntimeError("Logits contain NaN")

            loss = criterion(logits, labels)
            # backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #print("labels unique:", torch.unique(labels))
            #print("labels dtype:", labels.dtype)

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / len(train_loader)

        # ===== 验证阶段 =====
        val_acc, val_f1, val_auc, val_sensitivity, val_specificity = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        elogger.log(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}, Val Sensitivity={val_sensitivity:.4f}, Val Specificity={val_specificity:.4f}")

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            elogger.log(f"Saved new best model @ Epoch {epoch}")
        else:
            patience_counter += 1
            elogger.log(f"No improvement. Patience {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            elogger.log(f"Early stopping at epoch {epoch}")
            break
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")

    if evaluate_test:
        # ===== 测试阶段 =====
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_acc, test_f1, test_auc, test_sensitivity, test_specificity = evaluate(model, test_loader, device)
        elogger.log(f"Test Accuracy={test_acc:.4f}, Test F1={test_f1:.4f}, Test AUC={test_auc:.4f}, Test Sensitivity={test_sensitivity:.4f}, Test Specificity={test_specificity:.4f}")
        print(f"\n Final Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}, AUC: {test_auc:.4f}, Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}, Best Val Accuracy: {best_val_acc:.4f}")
    elogger.close()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    records = []

    for attr, traj, labels, metadata in loader:
        attr = {k: v.to(device) for k, v in attr.items()}
        traj = {k: v.to(device) for k, v in traj.items()}

        labels = labels.to(device)

        logits = model(attr, traj)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        labels_cpu = labels.detach().cpu().numpy()
        probs_cpu = probs.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()

        for index in range(len(labels_cpu)):
            meta = metadata[index] if index < len(metadata) else {}
            if isinstance(meta, dict):
                record = dict(meta)
            else:
                record = {"evaluation_id": meta}
            record.update(
                {
                    "y_true": int(labels_cpu[index]),
                    "y_prob": float(probs_cpu[index]),
                    "y_pred": int(preds_cpu[index]),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, return_details=False):
    predictions = collect_predictions(model, loader, device)
    metrics = compute_binary_metrics(
        predictions["y_true"],
        predictions["y_prob"],
        threshold=threshold,
    )

    acc = metrics["accuracy"]
    f1 = metrics["f1_macro"]
    auc = metrics["roc_auc"]
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]

    print("\n--- 分类报告 ---")
    print(classification_report(predictions["y_true"], predictions["y_pred"], zero_division=0))
    print("混淆矩阵:")
    print(confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=[0, 1]))
    print(
        "Clinical metrics: "
        f"PPV={metrics['ppv']:.4f}, NPV={metrics['npv']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, Brier={metrics['brier_score']:.4f}"
    )
    if return_details:
        return metrics, predictions
    return acc, f1, auc, sensitivity, specificity


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
    #num_filter = 64     #控制 卷积特征提取层（GeoConv） 的通道数增大 num_filter → 模型容量变强，能提取更复杂的轨迹特征，但显著增加显存和训练时间；减小 num_filter → 模型更轻，速度快但特征表达能力变弱。
    pooling_method = 'attention'
    # pooling_method = 'mean'  
    #num_fc_layers = 2  #控制 分类器（最终全连接部分） 的层数 增加 分类器表达力增强，可以更好地处理非线性关系；但参数量变多，容易过拟合；减少模型更简单，泛化更稳，但可能欠拟合。
    #hidden_size = 128
    num_filter = 32
    num_fc_layers = 1
    hidden_size = 48
    epochs = 50
    batch_size = 32
    file_path="data2.parquet"  # 目标文件路径

    train_loader, val_loader, test_loader = get_loader(file_path, batch_size, seed=10)   #输出符合预期

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
    train(model, elogger, train_loader, val_loader, test_loader, epochs, batch_size) 
    return

if __name__ == "__main__":
    main()
