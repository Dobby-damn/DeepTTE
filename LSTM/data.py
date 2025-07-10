# incremental_training.py

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
from typing import List, Tuple

# ──────────────── CONFIG ───────────────────────────────────────────────────

CSV_PATH      = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(1).csv"
XLSX_PATH     = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(2).xlsx"
IDDIAG_PATH   = r"D:/Code/DeepTTE/LSTM/iddiag.csv"

NUM_EPOCHS_PER_CHUNK = 50
CHUNK_SIZE    = 100_000
MAX_SEQ_LEN   = 150
BATCH_SIZE    = 64
LR            = 1e-3
HIDDEN_SIZE   = 64
NUM_LAYERS    = 10

CHECKPOINT_DIR = "checkpoints"
HISTORY_PLOT   = "training_history.png"
RESUME_PATH    = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────── 工具函数 & 类 ─────────────────────────────────────────────

def pad_sequence(arr: np.ndarray, length: int = MAX_SEQ_LEN) -> np.ndarray:
    if arr.shape[0] >= length:
        return arr[:length]
    pad = np.zeros((length - arr.shape[0], 2), dtype=arr.dtype)
    return np.vstack([arr, pad])

def build_sequences(df: pd.DataFrame, id_col: str = "evaluation_id") -> Tuple[List[np.ndarray], List[int]]:
    seqs = []
    lbls = []
    for pid, grp in df.groupby(id_col, sort=False):
        lbl = grp["label"].iloc[0]
        if pd.isna(lbl):
            continue
        xy = grp[["x","y"]].values.astype(np.float32)
        seqs.append(pad_sequence(xy, MAX_SEQ_LEN))
        lbls.append(int(lbl))
    return seqs, lbls

class TouchDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], labels: List[int]):
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def evaluate(model, loader, criterion) -> Tuple[float, float]:
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

def save_checkpoint(model, optim, chunk_idx, history):
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

def plot_history(history, out_fn):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"],   label="Val loss")
    plt.xlabel("Chunk #"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_fn); plt.close()

    plt.figure()
    plt.plot(epochs, history["val_acc"], label="Val acc (%)")
    plt.xlabel("Chunk #"); plt.ylabel("Accuracy (%)"); plt.ylim(0,100); plt.tight_layout()
    plt.savefig(out_fn.replace(".png","_acc.png")); plt.close()

# ──────────────── Main ─────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. 读诊断标签
    diag_df = pd.read_csv(IDDIAG_PATH, usecols=["evaluation_id","diagnose"])
    # print(diag_df.head())
    label_encoder = LabelEncoder().fit([0,1])
    # print("label_ancoder:", label_encoder)   # 31516.0       1.0形式
    num_classes = len(label_encoder.classes_)
    # print("num_classes:", num_classes)
    print(f"标签类别：{label_encoder.classes_}")

    # 2. 验证集
    # print(pd.read_excel(XLSX_PATH).head())
    df_val = (pd.read_excel(XLSX_PATH)
                .rename(columns={"ex":"x","ey":"y"})
                .merge(diag_df, on="evaluation_id", how="inner")
                .rename(columns={"diagnose":"label"}))
    # print("135 df_val:", df_val.head())
    val_seqs, val_lbls_raw = build_sequences(df_val)
    # print("val_seqs, val_lbls_raw:139",val_seqs[:2], val_lbls_raw[:2])
    val_lbls = label_encoder.transform(val_lbls_raw)
    val_loader = DataLoader(TouchDataset(val_seqs, val_lbls), batch_size=BATCH_SIZE)
    print(f"验证集序列数：{len(val_seqs)}")

    # 3. 模型与训练设置
    model     = LSTMClassifier(2, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history   = defaultdict(list)
    start_chunk = 0

    # 恢复训练（可选）
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        history.update(ckpt["history"])
        start_chunk = ckpt["chunk_idx"]
        print(f"恢复到 chunk {start_chunk}")

    # 4. 增量训练
    for chunk_idx, chunk in enumerate(pd.read_csv(
                        CSV_PATH,
                        usecols=["ex","ey","evaluation_id"],
                        chunksize=CHUNK_SIZE,
                        engine="python"), start=1):
        if chunk_idx <= start_chunk:
            continue
        # 重命名 & 合并标签
        chunk = (chunk.rename(columns={"ex":"x","ey":"y"})
                      .merge(diag_df, on="evaluation_id", how="inner")
                      .rename(columns={"diagnose":"label"}))
        # print("chunk:",chunk.head()) # 这里的chunk 是【x,y,label,】而不是[x1,y1,x2,y2],lable的形式
        
        if chunk.empty:
            print(f"Chunk {chunk_idx} 空，跳过"); continue

        print(f"\nChunk {chunk_idx}: {chunk['evaluation_id'].nunique()} 条序列")
        train_seqs, train_lbls_raw = build_sequences(chunk)
        train_lbls = label_encoder.transform(train_lbls_raw)
        train_loader = DataLoader(TouchDataset(train_seqs, train_lbls),
                                  batch_size=BATCH_SIZE, shuffle=True)

        # 训练
        model.train()
        total_loss = 0.0
        for epoch in range(NUM_EPOCHS_PER_CHUNK):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"  ↪ Epoch {epoch+1}/{NUM_EPOCHS_PER_CHUNK}: train_loss={avg_loss:.4f}")
            # train_loss = total_loss / len(train_loader)
            total_loss += avg_loss
        train_loss = total_loss / NUM_EPOCHS_PER_CHUNK

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f" ↳ train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")

        # 记录 & 保存
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        save_checkpoint(model, optimizer, chunk_idx, history)

    # 5. 绘图保存
    plot_history(history, HISTORY_PLOT)
    print("完成，曲线保存在：", HISTORY_PLOT, HISTORY_PLOT.replace(".png","_acc.png"))

if __name__ == "__main__":
    main()
