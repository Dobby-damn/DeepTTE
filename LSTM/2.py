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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TrajectoryDataset(Dataset):
    """Dataset that groups (ex, ey) clicks per evaluation_id and attaches a label."""

    def __init__(
        self,
        iddiag_path: Path,
        traj_csv_path: Path,
        traj_xlsx_path: Path,
    ) -> None:
        # Load diagnostic labels
        iddiag = pd.read_csv(iddiag_path)
        self.label_encoder = LabelEncoder()
        iddiag["diagnose_encoded"] = self.label_encoder.fit_transform(iddiag["diagnose"])
        label_map = dict(zip(iddiag["evaluation_id"], iddiag["diagnose_encoded"]))

        # Load trajectories from two files and concatenate
        traj1 = pd.read_csv(traj_csv_path)
        traj2 = pd.read_excel(traj_xlsx_path)
        traj = pd.concat([traj1, traj2], ignore_index=True)

        # Group rows by participant
        groups = traj.groupby("evaluation_id")
        self.samples: List[Tuple[torch.Tensor, int]] = []
        missing_label = 0
        for eval_id, df in groups:
            label = label_map.get(eval_id)
            if label is None:
                missing_label += 1
                continue  # skip participants without a label
            coords = torch.tensor(df[["ex", "ey"]].values, dtype=torch.float32)
            self.samples.append((coords, int(label)))
        if missing_label:
            print(f"[Info] Skipped {missing_label} participants without labels.")

    def __len__(self) -> int:  # noqa: D401
        """Return number of participants available."""
        return len(self.samples)

    def __getitem__(self, idx: int):  # noqa: D401, D403
        return self.samples[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
    """Custom collate_fn to pad variable‑length sequences within each mini‑batch."""

    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True)  # zero‑pad shorter seqs
    return padded_sequences, lengths, torch.tensor(labels)


class LSTMClassifier(nn.Module):
    """5‑layer LSTM followed by a linear classifier."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 5,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):  # type: ignore[override]
        # Pack the padded batch so the LSTM ignores padded timesteps
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        logits = self.fc(hn[-1])  # last layer’s hidden state
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = correct = total = 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return epoch_loss / total, correct / total


@torch.no_grad()

def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = correct = total = 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        epoch_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return epoch_loss / total, correct / total


def main():  # noqa: D401
    # ------------------------------ Paths & Hyper‑parameters -----------------------------
    data_dir = Path("LSTM")
    iddiag_path = data_dir / "iddiag.csv"
    traj_csv_path = data_dir / "连线测试轨迹(1).csv"
    traj_xlsx_path = data_dir / "连线测试轨迹(2).xlsx"

    batch_size = 64
    hidden_size = 128
    num_epochs = 30
    learning_rate = 1e-3
    dropout = 0.2

    # ----------------------------------- Dataset ---------------------------------------
    dataset = TrajectoryDataset(iddiag_path, traj_csv_path, traj_xlsx_path)
    labels = [lbl for _, lbl in dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=labels
    )
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---------------------------------- Model & Optim ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_encoder.classes_)
    model = LSTMClassifier(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=5,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------- Training Loop -------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

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

    # --------------------------------- Visualization ----------------------------------
    epochs = range(1, num_epochs + 1)
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

    print("Training finished. Curves saved as loss_curve.png & accuracy_curve.png")


if __name__ == "__main__":
    main()
