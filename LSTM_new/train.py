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

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


# print("GPU是否可用：", torch.cuda.is_available())
# print("当前GPU名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")


class ShapWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, full_input):
        """
        full_input: (batch, seq_len + static_dim)
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
        X: pd.DataFrame = None, # 传递计算得到的属性特征
    ) -> None:
        # Load diagnostic labels
        iddiag = pd.read_parquet('data_new.parquet')
        
        iddiag = iddiag[iddiag["diagnose"].isin([0, 1])].copy()   # 只保留标签为0和1的样本
        self.label_encoder = LabelEncoder()      
        iddiag["diagnose_encoded"] = self.label_encoder.fit_transform(iddiag["diagnose"])

        label_map = dict(zip(iddiag["cmbctid"], iddiag["diagnose_encoded"]))  # id与label的映射关系 
        print("[Info] 二分类标签分布：\n", iddiag["diagnose_encoded"].value_counts())
        
        # ==== 对静态特征做归一化 ====
        
        self.eval_ids = X["cmbctid"].values
        static_features = X.drop(columns=["cmbctid", "ex", "ey"]).values     
        self.scaler = StandardScaler()
        self.static_features_scaled = self.scaler.fit_transform(static_features)

        # 保证 evaluation_id 和归一化后的特征对应
        self.static_map = {
            eid: feat for eid, feat in zip(self.eval_ids, self.static_features_scaled)
        }

        traj = X[['ex', 'ey', 'cmbctid']]   
        k = 0

        
        # ====== 收集样本 ======
        self.samples: List[Tuple[torch.Tensor, int]] = []
        missing_label = 0
        all_coords = []  # 用于存储所有坐标

        for _, row in X.iterrows():
            ex = np.asarray(row["ex"], dtype=np.float32)
            ey = np.asarray(row["ey"], dtype=np.float32)

            if ex.ndim != 1 or ey.ndim != 1:
                continue
            if len(ex) == 0 or len(ey) == 0:
                continue

            coords = np.column_stack([ex, ey])  # (T, 2)
            all_coords.append(coords)
        # ====== 对所有轨迹整体归一化 ======
        if len(all_coords) == 0:
            # no trajectories found -> create dummy mean/std
            coord_mean = np.array([0.0, 0.0], dtype=np.float32)
            coord_std = np.array([1.0, 1.0], dtype=np.float32)
        else:
            all_coords = np.vstack(all_coords)  # 拼接成大矩阵
            assert all_coords.ndim == 2 and all_coords.shape[1] == 2, all_coords.shape
            coord_mean = all_coords.mean(axis=0)
            coord_std = all_coords.std(axis=0) + 1e-6  # 防止除0

        # ====== 重新构建样本 ======
        i = 0
        for _, row in X.iterrows():
            eval_id = row["cmbctid"]

            ex = np.asarray(row["ex"], dtype=np.float32)
            ey = np.asarray(row["ey"], dtype=np.float32)

            if eval_id not in label_map:
                continue
            label = label_map.get(eval_id)
            if label is None:
                continue

            coords = np.column_stack([ex, ey])[:2048]    #截断到2048个点
            coords = (coords - coord_mean) / coord_std
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            coords = torch.tensor(coords, dtype=torch.float32)

            static_feat = torch.tensor(self.static_map[eval_id], dtype=torch.float32)
            static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=0.0, neginf=0.0)

            self.samples.append((coords, static_feat, int(label)))
          
        if missing_label:
            print(f"[Info] Skipped {missing_label} participants without labels.")

    def __len__(self) -> int:  
        """Return number of participants available."""
        return len(self.samples)


    def __getitem__(self, idx: int):  
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


class EarlyStopping:
    """Simple EarlyStopping helper.

    Args:
        patience: epochs to wait after last improvement
        min_delta: minimum change to qualify as improvement
        mode: 'min' to monitor loss, 'max' to monitor accuracy/metric
        restore_best: whether to restore best model state when stopping
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min', restore_best=True, ckpt_path=None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best = restore_best
        self.ckpt_path = ckpt_path

        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")

        self.best = None
        self.num_bad_epochs = 0
        self.is_better = (lambda a, b: a < b - self.min_delta) if mode == 'min' else (lambda a, b: a > b + self.min_delta)

    def step(self, metric, model=None, optimizer=None, epoch=None):
        """Return True when should stop."""
        stop = False
        if self.best is None:
            self.best = metric
            self.num_bad_epochs = 0
            if self.ckpt_path and model is not None:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, self.ckpt_path)
            return False

        if self.is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
            if self.ckpt_path and model is not None:
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, self.ckpt_path)
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            stop = True

        return stop

    def restore_checkpoint(self, model, device=None):
        if not self.ckpt_path:
            return
        if not os.path.exists(self.ckpt_path):
            return
        map_loc = (device if device is not None else None)
        ck = torch.load(self.ckpt_path, map_location=map_loc)
        model.load_state_dict(ck['model_state_dict'])


class LSTMClassifier(nn.Module):
    def __init__(
        self,       
        input_size: int = 9,    # 轨迹点 (ex, ey......)
        static_dim: int = 27,   # 属性特征维度
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
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
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )

        # 静态特征 MLP
        self.fc_static = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类层：拼接时序和静态 embedding
        combined_dim = hidden_size * self.num_directions + hidden_size
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward_static_only(self, static_feats):
        """
        仅用静态特征进行推理（用于 SHAP）
        """
        # 1. 静态特征 MLP
        static_out = self.fc_static(static_feats)   # [N, 64]
        # 2. Dummy 序列向量（全部为零），保持原维度一致
        dummy_seq = torch.zeros(static_feats.size(0), 128, device=static_feats.device)  # [N, 128]
        # 3. 拼接 (静态64 + 序列128 = 192)
        fused = torch.cat([static_out, dummy_seq], dim=1)
        # 4. 全连接层（模型原始的分类头）
        logits = self.fc(fused)

        return logits
        

    def forward(self, x, lengths, static_feats):
        # x: (batch, seq, 2)
        # ----- 1. pack -----
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        # ----- 2. unpack -----
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (batch, seq, 2*hidden)
        # ----- 3. Multi-Head Attention -----
        # Q = K = V = lstm_out
        max_len = lstm_out.size(1)
        mask = (
            torch.arange(max_len, device=lengths.device)[None, :]
            < lengths[:, None]
        )  # (batch, seq)

        key_padding_mask = ~mask  # True = padding
        key_padding_mask = ~mask.squeeze(-1)  # True = padding
        attn_out, _ = self.attn(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )
        # (batch, seq, 2*hidden)
        # ----- 4. 序列池化（取平均 or 最大）-----
        mask = mask.unsqueeze(-1).float()
        

        attn_out = attn_out * mask
        seq_repr = attn_out.sum(dim=1) / lengths.unsqueeze(1)

  
        # ----- 5. 静态特征 MLP -----
        static_vec = self.fc_static(static_feats)  # (batch, 32)
        # ----- 6. 融合 -----
        fused = torch.cat([seq_repr, static_vec], dim=1)
        # ----- 7. 分类 -----
        logits = self.fc(fused)

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


def main():  
    # print(X.head())
    df = pd.read_parquet('data_new.parquet')

    x = df.drop(columns=['mmse_s', 'moca_s', 'evaluation_id'])  # 删除无关列
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
    hidden_size = 64
    num_epochs = 200
    learning_rate = 3e-4
    dropout = 0.3
    input_size = 9  # 2（ex, ey）
    num_layers = 3 #LSTM 层数

    # ----------------------------------- Dataset ---------------------------------------
    dataset = TrajectoryDataset(X)
    
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

    # ---------------------------------- Model & Optim ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_encoder.classes_)
    torch.backends.cudnn.enabled = False
    # infer static feature dimension from dataset (prevents mismatch / leakage)
    inferred_static_dim = int(dataset.static_features_scaled.shape[1])
    print(f"[Info] Inferred static feature dim: {inferred_static_dim}")
    model = LSTMClassifier(
        input_size = input_size,
        static_dim = inferred_static_dim,
        hidden_size = hidden_size,
        num_layers = num_layers,
        num_classes = num_classes,
        dropout = dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)  # 初始学习率 L2正则化

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    class_counts = torch.tensor([157, 170], dtype=torch.float32)
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------- Training Loop -------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1, f1_class0, auc, *_ = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history.setdefault("f1_score", []).append(f1)
        history.setdefault("f1_score_class0", []).append(f1_class0)

        scheduler.step(val_loss)  # 每个epoch更新学习率
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
        
        # --- Early stopping check ---
        # Create early_stopper on first epoch
        if epoch == 1:
            early_stop_ckpt = ckpt_dir / "best_model_earlystop.pt"
            early_stopper = EarlyStopping(patience=10, min_delta=1e-4, mode='min', restore_best=True, ckpt_path=str(early_stop_ckpt))

        # monitor validation loss (mode='min')
        should_stop = early_stopper.step(val_loss, model=model, optimizer=optimizer, epoch=epoch)
        if should_stop:
            print(f"[EarlyStopping] no improvement for {early_stopper.patience} epochs. Stopping at epoch {epoch}.")
            if early_stopper.restore_best:
                print("[EarlyStopping] Restoring best model from checkpoint.")
                early_stopper.restore_checkpoint(model, device=device)
            break
    epochs = range(1, epoch + 1)

    # _, _, _, _, auc, _, val_probs, val_targets = eval_epoch(model, val_loader, criterion, device)
    print("\n===== Running final test evaluation =====")
    test_loss, test_acc, test_f1, test_f1_class0, test_auc, _, val_probs, val_targets = eval_epoch(model, test_loader, criterion, device)
    best_thr, best_f1 = find_best_threshold(val_probs, val_targets, metric="f1")
    print(f"Best F1(1)={best_f1:.3f} at threshold={best_thr:.3f}")
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, "f"f1={test_f1:.4f}, auc={test_auc:.4f}")

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
    plt.xlabel(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, "f"f1={test_f1:.4f}, auc={test_auc:.4f}")
    plt.ylabel("f1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("f1_score_curve.png")

    print("Training finished. Curves saved as loss_curve.png & accuracy_curve.png")
    del model
    del optimizer
    del train_loader
    del test_loader
    del val_loader
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
