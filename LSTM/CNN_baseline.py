from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from train import BucketBatchSampler, TrajectoryDataset, collate_fn


SEED = 10
BATCH_SIZE = 64
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.30
CNN_CHANNELS = (32, 64, 128)
KERNEL_SIZE = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNBaseline(nn.Module):
    def __init__(
        self,
        input_size: int,
        static_dim: int,
        num_classes: int = 2,
        channels: tuple[int, ...] = CNN_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = input_size
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Average and max pooling each emit channels[-1] features.
        fused_dim = channels[-1] * 2 + 128
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def masked_pool(feature_map: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = feature_map.shape
        positions = torch.arange(seq_len, device=feature_map.device)
        mask = positions.unsqueeze(0) < lengths.view(batch_size, 1)
        mask = mask.unsqueeze(1)

        valid = mask.to(feature_map.dtype)
        denominator = lengths.clamp(min=1).to(feature_map.dtype).view(batch_size, 1)
        average = (feature_map * valid).sum(dim=2) / denominator

        masked_for_max = feature_map.masked_fill(~mask, torch.finfo(feature_map.dtype).min)
        maximum = masked_for_max.max(dim=2).values
        return torch.cat([average, maximum], dim=1)

    def forward(
        self,
        trajectories: torch.Tensor,
        lengths: torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        feature_map = self.cnn(trajectories.transpose(1, 2))
        sequence_vector = self.masked_pool(feature_map, lengths)
        static_vector = self.static_mlp(static_features)
        return self.classifier(torch.cat([sequence_vector, static_vector], dim=1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[float] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for trajectories, lengths, static_features, targets in loader:
            trajectories = trajectories.to(device)
            lengths = lengths.to(device)
            static_features = static_features.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(trajectories, lengths, static_features)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_probabilities.extend(probabilities.detach().cpu().tolist())

    metrics = {
        "loss": total_loss / len(all_targets),
        "accuracy": float(np.mean(np.equal(all_targets, all_predictions))),
        "macro_f1": f1_score(
            all_targets, all_predictions, average="macro", zero_division=0
        ),
        "f1_class1": f1_score(
            all_targets, all_predictions, pos_label=1, zero_division=0
        ),
        "f1_class0": f1_score(
            all_targets, all_predictions, pos_label=0, zero_division=0
        ),
        "auc": roc_auc_score(all_targets, all_probabilities),
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }
    return metrics


def build_static_features() -> pd.DataFrame:
    dataframe = pd.read_parquet("data.parquet")
    drop_columns = [
        "video",
        "hkbcscore",
        "moca_s",
        "moca_score",
        "diagnose",
        "id",
        "birthdate",
        "game_code",
        "save_time",
        "create_time",
        "update_time",
        "touchDuration",
        "numberInterval",
        "name",
        "Unnamed: 2",
    ]
    features = dataframe.drop(columns=drop_columns)
    label_columns = [
        column
        for column in ("diagnose", "diagnose_encoded", "label")
        if column in features.columns
    ]
    return features.drop(columns=label_columns)


def save_curves(history: list[dict], output_dir: Path) -> None:
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)


def main() -> None:
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")

    device = torch.device("cuda")
    output_dir = Path("result") / "cnn_baseline"
    checkpoint_dir = Path("checkpoints") / "cnn_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    static_features = build_static_features()
    dataset = TrajectoryDataset(
        "data.parquet",
        Path("data") / "连线测试轨迹(1).csv",
        Path("data") / "连线测试轨迹(2).xlsx",
        Path("data") / "连线测试轨迹体检.csv",
        static_features,
    )
    labels = np.asarray([label for *_, label in dataset.samples])

    train_indices, temporary_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=SEED,
        stratify=labels[temporary_indices],
    )

    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)
    test_subset = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_subset,
        batch_sampler=BucketBatchSampler(train_subset, BATCH_SIZE),
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CNNBaseline(
        input_size=9,
        static_dim=int(dataset.static_features_scaled.shape[1]),
        num_classes=len(dataset.label_encoder.classes_),
    ).to(device)

    train_counts = np.bincount(labels[train_indices])
    class_weights = 1.0 / torch.tensor(train_counts, dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started_at = time.perf_counter()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        f"Samples: train={len(train_subset)}, val={len(validation_subset)}, "
        f"test={len(test_subset)}"
    )
    print(
        f"CNN: channels={CNN_CHANNELS}, kernel={KERNEL_SIZE}, "
        f"batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": validation_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": validation_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": validation_metrics["macro_f1"],
                "val_auc": validation_metrics["auc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_f1={validation_metrics['macro_f1']:.4f} "
            f"val_auc={validation_metrics['auc']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "val_accuracy": validation_metrics["accuracy"],
                    "config": {
                        "channels": list(CNN_CHANNELS),
                        "kernel_size": KERNEL_SIZE,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "dropout": DROPOUT,
                    },
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    save_curves(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)

    matrix = confusion_matrix(
        test_metrics["targets"], test_metrics["predictions"], labels=[0, 1]
    )
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    predictions_frame = pd.DataFrame(
        {
            "target": test_metrics["targets"],
            "prediction": test_metrics["predictions"],
            "probability_class1": test_metrics["probabilities"],
        }
    )
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    result = {
        "experiment": "CNN baseline",
        "entry_script": "LSTM/CNN_baseline.py",
        "device": torch.cuda.get_device_name(0),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_accuracy": checkpoint["val_accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_f1_class1": test_metrics["f1_class1"],
        "test_f1_class0": test_metrics["f1_class0"],
        "test_auc": test_metrics["auc"],
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "confusion_matrix": matrix.tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - started_at,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        classification_report(
            test_metrics["targets"],
            test_metrics["predictions"],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
