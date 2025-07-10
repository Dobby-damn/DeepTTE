import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# # ------------------------ 配置 ------------------------
# file1 = r"D:\Code\DeepTTE\LSTM\连线测试轨迹(1).csv"

# file2 = r"D:\Code\DeepTTE\LSTM\连线测试轨迹(2).xlsx"
CSV_PATH    = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(1).csv"
XLSX_PATH   = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(2).xlsx"
IDDIAG_PATH = r"D:/Code/DeepTTE/LSTM/iddiag.csv"   # 新增
# ── 全部标签预扫描 ────────────────────────────────────────────────────────
# 轨迹文件里没有 label，直接从 iddiag.csv 里读
diag_df = pd.read_csv(IDDIAG_PATH, usecols=["video", "diagnose"])
# 诊断标签只有 0/1 两类
label_encoder = LabelEncoder().fit([0,1])
num_classes = 2
print(f"诊断类别：{label_encoder.classes_}")

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

max_seq_len = 100
chunk_size = 20000
batch_size = 64
num_epochs_per_chunk = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ 模型定义 ------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=5, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ------------------------ 数据集 ------------------------
class TouchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(np.stack(sequences), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), 2))
        return np.vstack([seq, padding])
    else:
        return seq[:max_len]

# ── build_sequences 函数改动 ────────────────────────────────────────────
def build_sequences(df: pd.DataFrame,
                    id_col: str = "evaluation_id") -> tuple[list[np.ndarray], list[int]]:
    """Group a DataFrame by id_col → list of (x,y)-arrays and integer labels."""
    seqs, lbls = [], []
    # 要求 DataFrame 有这几列： [id_col, "x","y","label"]
    for pid, grp in df.groupby(id_col, sort=False):
        # 忽略没有标签的数据
        if grp["label"].isnull().all():
            continue
        xy = grp[["x","y"]].values.astype(np.float32)
        seqs.append(pad_sequence(xy))
        lbls.append(int(grp["label"].iloc[0]))
    return seqs, lbls

# ── 构建验证集 ───────────────────────────────────────────────────────────
df_val_traj = pd.read_excel(XLSX_PATH)  # 包含 ex,ey,evaluation_id
df_val = (df_val_traj
          .merge(diag_df, on="evaluation_id", how="inner")
          .rename(columns={"ex":"x","ey":"y","diagnose":"label"}))
print(df_val.head())
exit(0)
val_seqs, val_lbls_raw = build_sequences(df_val, id_col="evaluation_id")
val_lbls = label_encoder.transform(val_lbls_raw)
val_loader = DataLoader(TouchDataset(val_seqs, val_lbls), batch_size=batch_size)



# ------------------------ 读取验证集 ------------------------
# val_df = pd.read_excel(file2)
# val_sequences = []
# val_labels = []

# for pid, group in val_df.groupby('evaluation_id'):
#     xy = group[['ex', 'ey']].values
#     label = group['evaluation_type'].iloc[0]
#     val_sequences.append(pad_sequence(xy, max_seq_len))
#     val_labels.append(label)

# # 标签编码
# label_encoder = LabelEncoder()
# val_labels_encoded = label_encoder.fit_transform(val_labels)
# val_dataset = TouchDataset(val_sequences, val_labels_encoded)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ------------------------ 初始化模型 ------------------------
model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=5, num_classes=len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
val_accuracies = []

# ------------------------ 分块训练 ------------------------
chunk_num = 0
for chunk_idx in pd.read_csv(CSV_PATH, usecols=["ex","ey","evaluation_id"], chunksize=chunk_size, state = 1):
    # 把列重命名为 x,y,id，然后与诊断标签合并
    chunk = (chunk
             .rename(columns={"ex":"x","ey":"y"})
             .merge(diag_df, on="evaluation_id", how="inner")
             .rename(columns={"diagnose":"label"}))
    if chunk.empty:
        print(f"Chunk {chunk_idx} 全部无标签 → skip")
        continue

    print(f"Chunk {chunk_idx}: 有效轨迹 {chunk['evaluation_id'].nunique()} 个")
    train_seqs, train_lbls_raw = build_sequences(chunk, id_col="evaluation_id")
    train_lbls = label_encoder.transform(train_lbls_raw)

    train_loader = DataLoader(TouchDataset(train_seqs, train_lbls),
                              batch_size=batch_size, shuffle=True)
    # chunk_num += 1
    # print(f"\n📦 第 {chunk_num} 块，共 {len(chunk)} 行")

    # sequences, labels = [], []
    # for pid, group in chunk.groupby('evaluation_id'):
    #     if len(group) < 5: continue
    #     xy = group[['ex', 'ey']].values
    #     label = group['evaluation_type'].iloc[0]
    #     sequences.append(pad_sequence(xy, max_seq_len))
    #     labels.append(label)

    # if len(sequences) < 10:
    #     print("数据不足，跳过此块")
    #     continue

    # labels_encoded = label_encoder.transform(labels)
    # train_dataset = TouchDataset(sequences, labels_encoded)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs_per_chunk):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"🧠 Train Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # ----------- 验证评估 -----------
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            pred = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(pred)
            val_targets.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"✅ Val Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

    # ----------- 保存模型 -----------
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_chunk{chunk_num}.pt"))

# ------------------------ 可视化 ------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Chunk')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss 曲线")

plt.subplot(1,2,2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Chunk')
plt.ylabel('Accuracy')
plt.legend()
plt.title("验证准确率")

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
