# incremental_training.py

"""
增量训练 5 层 LSTM 脚本
======================

数据：
- D:/Code/DeepTTE/LSTM/轨迹测试数据(1).csv    (~1 000 000 行)
- D:/Code/DeepTTE/LSTM/轨迹测试数据(2).xlsx  (~20 000 行)
- D:/Code/DeepTTE/LSTM/iddiag.csv           (evaluation_id, diagnose)

功能：
1. 按 chunk 逐块读入大 CSV，合并诊断标签，训练后保存 checkpoint
2. 把 XLSX 文件当验证集，一次性加载并评估
3. 每个 chunk 训练后：计算 train loss、val loss、val acc；保存到 history
4. 训练结束自动输出两张曲线图（loss & accuracy）
5. 支持从某个 checkpoint 恢复训练
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ──────────────── CONFIG ───────────────────────────────────────────────────

CSV_PATH      = r"D:/Code/DeepTTE/LSTM/轨迹测试数据(1).csv"
XLSX_PATH     = r"D:/Code/DeepTTE/LSTM/轨迹测试数据(2).xlsx"
IDDIAG_PATH   = r"D:/Code/DeepTTE/LSTM/iddiag.csv"

CHUNK_SIZE    = 100_000    # CSV 按行数分块
MAX_SEQ_LEN   = 150        # 每条轨迹 pad/truncate 长度
BATCH_SIZE    = 64
LR            = 1e-3
HIDDEN_SIZE   = 64
NUM_LAYERS    = 5

CHECKPOINT_DIR = "checkpoints"
HISTORY_PLOT   = "training_history.png"
RESUME_PATH    = None       # e.g. "checkpoints/ckpt_chunk3.pt" 来恢复

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────── 工具函数 & 类 ─────────────────────────────────────────────

def pad_sequence(arr: np.ndarray, length: int = MAX_SEQ_LEN) -> np.ndarray:
    """Pad or truncate (N,2) array to (length,2)."""
    if arr.shape[0] >= length:
        return arr[:length]
    pad = np.zeros((length - arr.shape[0], 2), dtype=arr.dtype)
    return np.vstack([arr, pad])

def build_sequences(df: pd.DataFrame, id_col: str = "evaluation_id"): #-> tuple[list[np.ndarray], list[int]]:
    """
    按受试者分组，返回 list of (x,y)-ndarrays 和 int labels。
    需 df 包含列：[id_col, "x","y","label"]。
    """
    seqs, lbls = [], []
    for pid, grp in df.groupby(id_col, sort=False):
        lbl = grp["label"].iloc[0]
        # 如果 label 丢失 (NaN)，跳过此受试者
        if pd.isna(lbl):
            continue
        xy = grp[["x","y"]].values.astype(np.float32)
        seqs.append(pad_sequence(xy, MAX_SEQ_LEN))
        lbls.append(int(lbl))
    return seqs, lbls

class TouchDataset(Dataset):
    """Wrap sequences & labels into PyTorch Dataset."""
    def __init__(self, seqs: list[np.ndarray], labels: list[int]):
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    """5-layer LSTM + 1-layer FC."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: (B, T, 2)
        out, _ = self.lstm(x)            # (B, T, hidden)
        last = out[:, -1, :]             # (B, hidden)
        return self.fc(last)             # (B, num_classes)

def evaluate(model: nn.Module, loader: DataLoader, criterion) :#-> tuple[float, float]:
    """返回 (avg_loss, accuracy%) on loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(loader), 100 * correct / total

def save_checkpoint(model, optim, chunk_idx: int, history: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "history":     history,
        "chunk_idx":   chunk_idx
    }
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_chunk{chunk_idx}.pt")
    torch.save(ckpt, path)
    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def plot_history(history: dict, out_fn: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Val loss")
    plt.xlabel("Chunk #")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fn)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["val_acc"], label="Val acc (%)")
    plt.xlabel("Chunk #")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_fn.replace(".png","_acc.png"))
    plt.close()

# ──────────────── 主流程 ────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1️⃣ 读取诊断标签，构建 LabelEncoder
    diag_df = pd.read_csv(IDDIAG_PATH, usecols=["evaluation_id","diagnose"])
    label_encoder = LabelEncoder().fit([0,1])  # 0=正常,1=异常
    num_classes = len(label_encoder.classes_)
    print(f"检测到 {num_classes} 类标签：{label_encoder.classes_}")

    # 2️⃣ 构建验证集（XLSX）
    df_val = pd.read_excel(XLSX_PATH, engine="openpyxl") \
               .rename(columns={"ex":"x","ey":"y"}) \
               .merge(diag_df, on="evaluation_id", how="inner") \
               .rename(columns={"diagnose":"label"})
    val_seqs, val_lbls_raw = build_sequences(df_val, id_col="evaluation_id")
    val_lbls = label_encoder.transform(val_lbls_raw)
    val_loader = DataLoader(TouchDataset(val_seqs, val_lbls), batch_size=BATCH_SIZE)
    print(f"验证集：{len(val_seqs)} 条序列")

    # 3️⃣ 模型 & 优化器 & 损失 & history
    model     = LSTMClassifier(input_size=2,
                               hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS,
                               num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history   = defaultdict(list)
    start_chunk = 0

    # 3b️⃣ 如果要恢复训练
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        history.update(ckpt["history"])
        start_chunk = ckpt["chunk_idx"]
        print(f"从 {RESUME_PATH} 恢复，已训练到 chunk {start_chunk}")

    # 4️⃣ 增量训练：逐 chunk 读 CSV
    for chunk_idx, chunk in enumerate(pd.read_csv(
                                        CSV_PATH,
                                        usecols=["ex","ey","evaluation_id"],
                                        chunksize=CHUNK_SIZE,
                                        engine="python"), start=1):
        if chunk_idx <= start_chunk:
            continue

        # 重命名列 & 合并标签
        chunk = (chunk
                 .rename(columns={"ex":"x","ey":"y"})
                 .merge(diag_df, on="evaluation_id", how="inner")
                 .rename(columns={"diagnose":"label"}))
        if chunk.empty:
            print(f"Chunk {chunk_idx}：无有效带标签轨迹，跳过")
            continue

        print(f"\n🔹 Chunk {chunk_idx}: {len(chunk)} 行 → {chunk['evaluation_id'].nunique()} 条序列")
        train_seqs, train_lbls_raw = build_sequences(chunk, id_col="evaluation_id")
        train_lbls = label_encoder.transform(train_lbls_raw)
        train_loader = DataLoader(TouchDataset(train_seqs, train_lbls),
                                  batch_size=BATCH_SIZE, shuffle=True)

        # —— 训练一个 mini‑epoch
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # —— 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"  ↳ train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")

        # —— 记录 & checkpoint
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        save_checkpoint(model, optimizer, chunk_idx, history)

    # 5️⃣ 完成后绘图
    plot_history(history, HISTORY_PLOT)
    print("训练完成。曲线已保存：", HISTORY_PLOT,
          HISTORY_PLOT.replace(".png","_acc.png"))

if __name__ == "__main__":
    main()
