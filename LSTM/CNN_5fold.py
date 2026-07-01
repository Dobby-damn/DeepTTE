from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from CNN_baseline import (
    BATCH_SIZE,
    CNN_CHANNELS,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    KERNEL_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    SEED,
    WEIGHT_DECAY,
    CNNBaseline,
    build_static_features,
    run_epoch,
    set_seed,
)
from train import BucketBatchSampler, TrajectoryDataset, collate_fn


N_SPLITS = 5


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")

    device = torch.device("cuda")
    output_dir = Path("result") / "cnn_5fold"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path("best_model_c5.pt")

    static_features = build_static_features()
    dataset = TrajectoryDataset(
        "data.parquet",
        Path("data") / "连线测试轨迹(1).csv",
        Path("data") / "连线测试轨迹(2).xlsx",
        Path("data") / "连线测试轨迹体检.csv",
        static_features,
    )
    labels = np.asarray([label for *_, label in dataset.samples], dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results: list[dict] = []
    histories: list[dict] = []
    oof_rows: list[dict] = []
    aggregate_matrix = np.zeros((2, 2), dtype=np.int64)
    global_best_macro_f1 = -1.0
    global_best_fold = 0
    global_best_epoch = 0
    started_at = time.perf_counter()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        f"5-fold CNN: channels={CNN_CHANNELS}, kernel={KERNEL_SIZE}, "
        f"batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}"
    )

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.arange(len(dataset)), labels),
        start=1,
    ):
        set_seed(SEED + fold)
        train_subset = Subset(dataset, train_indices)
        validation_subset = Subset(dataset, validation_indices)
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

        model = CNNBaseline(
            input_size=9,
            static_dim=int(dataset.static_features_scaled.shape[1]),
            num_classes=len(dataset.label_encoder.classes_),
        ).to(device)

        train_counts = np.bincount(labels[train_indices], minlength=2)
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

        best_fold_macro_f1 = -1.0
        best_fold_epoch = 0
        best_fold_accuracy = 0.0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        print(
            f"\nFold {fold}/{N_SPLITS}: "
            f"train={len(train_subset)}, validation={len(validation_subset)}"
        )
        for epoch in range(1, MAX_EPOCHS + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            validation_metrics = run_epoch(
                model, validation_loader, criterion, device
            )
            scheduler.step(validation_metrics["macro_f1"])

            histories.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_auc": validation_metrics["auc"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"Fold {fold} epoch {epoch:02d}: "
                f"train_f1={train_metrics['macro_f1']:.4f} "
                f"val_acc={validation_metrics['accuracy']:.4f} "
                f"val_f1={validation_metrics['macro_f1']:.4f} "
                f"val_auc={validation_metrics['auc']:.4f}"
            )

            if validation_metrics["macro_f1"] > best_fold_macro_f1 + 1e-6:
                best_fold_macro_f1 = validation_metrics["macro_f1"]
                best_fold_epoch = epoch
                best_fold_accuracy = validation_metrics["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= EARLY_STOPPING_PATIENCE:
                    print(f"Fold {fold}: early stopping at epoch {epoch}.")
                    break

        if best_state is None:
            raise RuntimeError(f"Fold {fold} did not produce a checkpoint.")

        model.load_state_dict(best_state)
        fold_metrics = run_epoch(model, validation_loader, criterion, device)
        matrix = confusion_matrix(
            fold_metrics["targets"],
            fold_metrics["predictions"],
            labels=[0, 1],
        )
        aggregate_matrix += matrix
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0

        fold_result = {
            "fold": fold,
            "epochs_run": epoch,
            "best_epoch": best_fold_epoch,
            "accuracy": fold_metrics["accuracy"],
            "macro_f1": fold_metrics["macro_f1"],
            "f1_class1": fold_metrics["f1_class1"],
            "f1_class0": fold_metrics["f1_class0"],
            "auc": fold_metrics["auc"],
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": json.dumps(matrix.tolist()),
        }
        fold_results.append(fold_result)

        for dataset_index, target, prediction, probability in zip(
            validation_indices,
            fold_metrics["targets"],
            fold_metrics["predictions"],
            fold_metrics["probabilities"],
        ):
            oof_rows.append(
                {
                    "fold": fold,
                    "dataset_index": int(dataset_index),
                    "target": int(target),
                    "prediction": int(prediction),
                    "probability_class1": float(probability),
                }
            )

        if best_fold_macro_f1 > global_best_macro_f1:
            global_best_macro_f1 = best_fold_macro_f1
            global_best_fold = fold
            global_best_epoch = best_fold_epoch
            torch.save(
                {
                    "model_state_dict": best_state,
                    "fold": fold,
                    "epoch": best_fold_epoch,
                    "val_macro_f1": best_fold_macro_f1,
                    "val_accuracy": best_fold_accuracy,
                    "label_classes": dataset.label_encoder.classes_.tolist(),
                    "config": {
                        "n_splits": N_SPLITS,
                        "channels": list(CNN_CHANNELS),
                        "kernel_size": KERNEL_SIZE,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "dropout": DROPOUT,
                        "max_epochs": MAX_EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "seed": SEED,
                    },
                },
                best_model_path,
            )

        del model, optimizer, train_loader, validation_loader
        torch.cuda.empty_cache()

    fold_frame = pd.DataFrame(fold_results)
    fold_frame.to_csv(output_dir / "fold_results.csv", index=False)
    pd.DataFrame(histories).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(oof_rows).sort_values("dataset_index").to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )

    summary: dict[str, object] = {
        "experiment": "CNN 5-fold",
        "entry_script": "LSTM/CNN_5fold.py",
        "device": torch.cuda.get_device_name(0),
        "n_splits": N_SPLITS,
        "samples": len(dataset),
        "total_epochs_run": int(fold_frame["epochs_run"].sum()),
        "best_model_fold": global_best_fold,
        "best_model_epoch": global_best_epoch,
        "best_model_val_macro_f1": global_best_macro_f1,
        "aggregate_confusion_matrix": aggregate_matrix.tolist(),
        "parameter_count": sum(
            parameter.numel()
            for parameter in CNNBaseline(
                input_size=9,
                static_dim=int(dataset.static_features_scaled.shape[1]),
                num_classes=len(dataset.label_encoder.classes_),
            ).parameters()
        ),
        "training_seconds": time.perf_counter() - started_at,
    }
    for metric in (
        "accuracy",
        "macro_f1",
        "f1_class1",
        "f1_class0",
        "auc",
        "sensitivity",
        "specificity",
    ):
        mean, std = mean_std(fold_frame[metric].tolist())
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
