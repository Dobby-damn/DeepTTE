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
问题：
1.标准化/类权重在全体数据上计算 → 数据泄漏
在 TrajectoryDataset 里 StandardScaler().fit_transform(X)，分割 train/val 之前就做了；类权重也在全体样本上算。泄漏会抬高验证指标并影响训练动态。
2.10 层 LSTM 太深
深层 LSTM 训练困难、梯度消失更明显；曲线显示学习缓慢、上限不高，属于欠拟合/优化难度大。
3.阈值固定为 0.5
类不均衡时，0.5 不是最佳点，导致 F1(0类) 比 F1(1类) 低。
4.静态特征融合方式有限
目前是“时序注意力池化 + 静态 MLP 拼接”，可以更强：让静态特征调制时序表征（FiLM/gating），或做跨注意力。
5.学习率与调度
Cosine + Adam 没问题，但起始 LR=1e-4 可能有点小；深 LSTM 更难收敛。



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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
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
        print(traj1.shape, traj2.shape, traj.shape)

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
            # traj_coords = df[["ex", "ey"]].values[:2048]  # 取前2048个坐标
            # traj_features = X.loc[X["evaluation_id"] == eval_id].drop(columns=["evaluation_id"]).values[0] # 获取当前eval_id的特征
            # combined_features = np.hstack([traj_coords, np.tile(traj_features, (traj_coords.shape[0], 1))])  # 合并特征和轨迹坐标 将一个被试的一份属性特征复制{坐标点数}次，从两列变为2+20
            # # coords = torch.tensor(df[["ex", "ey"]].values[:2048], dtype=torch.float32)
            # self.samples.append((torch.tensor(combined_features, dtype=torch.float32), int(label)))
            coords = df[["ex", "ey"]].values  # 截断到最大长度
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

            self.samples.append((coords, static_feat, int(label)))
            # coords = torch.tensor(coords, dtype=torch.float32)
            # static_feat = torch.tensor(self.static_map[eval_id], dtype=torch.float32)
            # self.samples.append((coords, static_feat, int(label)))
        if missing_label:
            print(f"[Info] Skipped {missing_label} participants without labels.")

    def __len__(self) -> int:  # noqa: D401
        """Return number of participants available."""
        return len(self.samples)

    def __getitem__(self, idx: int):  # noqa: D401, D403
        # return self.samples[idx]
        coords, static_feat, label = self.samples[idx]
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


class LSTMClassifier(nn.Module):
    def __init__(
        self,       
        input_size: int = 2,    # 轨迹点 (ex, ey)
        static_dim: int = 20,   # 属性特征维度
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 时序 LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention 层
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # 静态特征 MLP
        self.fc_static = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 分类层：拼接时序和静态 embedding
        combined_dim = hidden_size * self.num_directions + hidden_size
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )


        # Final classifier
        # self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size * self.num_directions, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size // 2, num_classes)
        # )


    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static_feat):
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Attention: compute weights for each time step
        attn_weights = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        # Mask attention weights to ignore padded time steps
        mask = torch.arange(lstm_out.size(1), device=lengths.device)[None, :] < lengths[:, None]
        attn_weights[~mask] = -1e9  # mask out padding
        attn_weights = attn_weights - attn_weights.max(dim=1, keepdim=True)[0]  # 防溢出
        attn_scores = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)

        # Compute context vector (weighted sum of lstm outputs)
        context = torch.sum(lstm_out * attn_scores.unsqueeze(-1), dim=1)  # (batch, hidden*2)
        
        # 静态特征编码
        static_vec = self.fc_static(static_feat)

        # 拼接
        combined = torch.cat([context, static_vec], dim=-1)
        logits = self.fc(combined)  # (batch, num_classes)
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = correct = total = 0
    for x, lengths, static_feats, y in loader:
        x, lengths, static_feats, y = (
            x.to(device), 
            lengths.to(device),
            static_feats.to(device), 
            y.to(device)
        )
        optimizer.zero_grad()
        logits = model(x, lengths, static_feats)  # 注意这里传入了静态特征
        loss = criterion(logits, y)
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        # print("x device:", x.device)  # 应输出 cuda:0
        # print("model device:", next(model.parameters()).device)  # 应输出 cuda:0
    return epoch_loss / total, correct / total


@torch.no_grad()

# def eval_epoch(model, loader, criterion, device):
#     model.eval()
#     epoch_loss = correct = total = 0
#     for x, lengths, y in loader:
#         x, lengths, y = x.to(device), lengths.to(device), y.to(device)
#         logits = model(x, lengths)
#         loss = criterion(logits, y)
#         epoch_loss += loss.item() * y.size(0)
#         preds = logits.argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)
#     return epoch_loss / total, correct / total
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_probs = []
    epoch_loss = correct = total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, lengths, static_feats, y in loader:
            x, lengths, static_feats, y = (
                x.to(device), 
                lengths.to(device), 
                static_feats.to(device),
                y.to(device)
            )

            logits = model(x, lengths, static_feats)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]
            
            epoch_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # 收集所有预测和标签
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # 计算各项指标
    avg_loss = epoch_loss / total
    accuracy = correct / total
    #f1 = f1_score(all_targets, all_preds, average='binary')  # 默认计算类别1的F1
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    f1_class1 = f1_score(all_targets, all_preds, pos_label=1)
    f1_class0 = f1_score(all_targets, all_preds, pos_label=0)  # 专门计算类别0的F1
    auc = roc_auc_score(all_targets, all_probs)
    return  avg_loss, accuracy, f1_class1, f1_class0, auc, macro_f1, all_probs, all_targets # 正常类（类别0）的F1


def find_best_threshold(probs, y_true, metric="f1"):  # metric: 'f1' 以正类F1最大为准
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    if len(thr) == 0:
        return 0.5, 0.0
    best_idx = np.argmax(f1s[:-1])  # 跳过最后一个点
    return float(thr[best_idx]), float(f1s[best_idx])


def main():  # noqa: D401
    # ------------------------------ Paths & Hyper‑parameters -----------------------------
    # data_dir = Path("LSTM")
    iddiag_path = r"D:/Code/DeepTTE/LSTM/iddiag.csv"
    traj_csv_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(1).csv"
    traj_xlsx_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(2).xlsx"

    # 获取属性特征，edu_year, sex, age ,mean_speed etlc.
    extractor = TrajectoryFeatureExtractor(iddiag_path, traj_csv_path, traj_xlsx_path)
    X, y = extractor.get_feature_matrix_and_target()
    # print(X.head())
    print(y.head())

    batch_size = 64
    hidden_size = 64
    num_epochs = 1000
    learning_rate = 3e-4
    dropout = 0.2
    input_size = 2  # 2（ex, ey）
    num_layers = 3 #LSTM 层数

    # ----------------------------------- Dataset ---------------------------------------
    dataset = TrajectoryDataset(iddiag_path, traj_csv_path, traj_xlsx_path, X)
    
    labels = [lbl for *_, lbl in dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(
            len(dataset)), 
            test_size = 0.2, 
            random_state = 5, 
            stratify = labels
    )
    train_sampler = BucketBatchSampler(torch.utils.data.Subset(dataset, train_idx), batch_size)
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

    # ---------------------------------- Model & Optim ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_encoder.classes_)
    model = LSTMClassifier(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        num_classes = num_classes,
        dropout = dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 初始学习率

    # 余弦退火参数设置
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = 30,          # 初始周期长度（epoch数）
        T_mult = 2,        # 每次重启后周期倍增
        eta_min = 1e-5     # 最小学习率
    )
        

    #  为 CrossEntropyLoss 添加类别权重，这能显著抑制模型倾向于预测“异常”，从而提高对“正常”的识别率。##################################################
    class_counts = np.bincount(labels)  # [236, 447]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()  # 归一化为 1
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    ######################################################################################################
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------- Training Loop -------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1, f1_class0, auc, *_ = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history.setdefault("f1_score", []).append(f1)
        history.setdefault("f1_score_class0", []).append(f1_class0)

        scheduler.step()  # 每个epoch更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, LR: {current_lr:.6f}")

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Incremental save when validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / f"best_model_epoch{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "label_classes": dataset.label_encoder.classes_.tolist(),
            }, ckpt_path)
            print(f"[Checkpoint] Saved improved model to {ckpt_path}")
    epochs = range(1, num_epochs + 1)

    _, _, _, _, auc, _, val_probs, val_targets = eval_epoch(model, val_loader, criterion, device)
    best_thr, best_f1 = find_best_threshold(val_probs, val_targets, metric="f1")
    print(f"[Val] AUC={auc:.3f} | Best F1(1)={best_f1:.3f} at threshold={best_thr:.3f}")

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Training acc")
    plt.plot(epochs, history["val_acc"], label="Validation acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")

    plt.figure()
    plt.plot(epochs, history["f1_score"], label="F1 score")
    plt.plot(epochs, history["f1_score_class0"], label="F1 score normal")
    plt.xlabel("Epoch")
    plt.ylabel("f1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("f1_score_curve.png")

    print("Training finished. Curves saved as loss_curve.png & accuracy_curve.png")


if __name__ == "__main__":
    main()
