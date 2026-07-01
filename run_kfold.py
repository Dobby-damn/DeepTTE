from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

import logger
import models.Baseline_LITEMV as BaselineLITEMV
import models.Baseline_LSTM as BaselineLSTM
import models.Baseline_Transformer as BaselineTransformer
import models.DeepTTE as DeepTTE
from DSTLF import BucketBatchSampler, ParquetDataset, collate_fn, evaluate, train
from evaluation.calibration import calibration_table, decision_curve_table
from evaluation.metrics import compute_binary_metrics, summarize_metric_dicts
from evaluation.subgroup import evaluate_default_subgroups


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "dstlf":
        return DeepTTE.Net(
            num_classes=2,
            num_filter=32,
            hidden_size=48,
            num_fc_layers=1,
            dropout_p=0.5,
        )
    if model_name == "bilstm":
        return BaselineLSTM.Net(num_classes=2, hidden_size=48, dropout_p=0.5)
    if model_name == "transformer":
        return BaselineTransformer.Net(num_classes=2, d_model=64, num_layers=2, dropout_p=0.5)
    if model_name == "litemv":
        return BaselineLITEMV.Net(num_classes=2, dropout_p=0.5)
    raise ValueError(f"Unknown model_name={model_name}. Choose from dstlf, bilstm, transformer, litemv.")


def make_fold_dataset(raw_df: pd.DataFrame, ids, normalizer=None, fit_normalizer=False):
    fold_df = raw_df[raw_df["evaluation_id"].isin(ids)].copy()
    return ParquetDataset(
        dataframe=fold_df,
        normalize=True,
        normalizer=normalizer,
        fit_normalizer=fit_normalizer,
    )


def make_loader(dataset, batch_size: int, *, train_mode: bool):
    if train_mode:
        sampler = BucketBatchSampler(dataset, batch_size)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def split_train_val_ids(train_ids, id_label_map, *, val_ratio=0.2, seed=10):
    train_labels = np.array([id_label_map[subj_id] for subj_id in train_ids])
    fold_train_ids, fold_val_ids = train_test_split(
        train_ids,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
    )
    return np.array(fold_train_ids), np.array(fold_val_ids)


def run_k_fold(
    file_path,
    *,
    model_name="dstlf",
    k=5,
    batch_size=32,
    epochs=50,
    lr=1e-3,
    output_dir="outputs/acm_health_eval",
    seed=10,
    device=None,
    bootstrap_iters=2000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dataset = ParquetDataset(file_path=file_path, normalize=False)
    raw_df = raw_dataset.raw_df.copy()
    id_label_map = raw_df.groupby("evaluation_id")["diagnose"].first().to_dict()
    all_ids = np.array(sorted(id_label_map.keys()))
    id_labels = np.array([id_label_map[subj_id] for subj_id in all_ids])

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics = []
    prediction_frames = []

    for fold, (train_idx_ids, test_idx_ids) in enumerate(splitter.split(all_ids, id_labels), start=1):
        print(f"\n========== Fold {fold}/{k} | model={model_name} ==========")
        fold_train_ids = all_ids[train_idx_ids]
        fold_test_ids = all_ids[test_idx_ids]
        fold_train_ids, fold_val_ids = split_train_val_ids(
            fold_train_ids,
            id_label_map,
            val_ratio=0.2,
            seed=seed,
        )

        train_dataset = make_fold_dataset(raw_df, fold_train_ids, fit_normalizer=True)
        normalizer = train_dataset.get_normalizer()
        val_dataset = make_fold_dataset(raw_df, fold_val_ids, normalizer=normalizer, fit_normalizer=False)
        test_dataset = make_fold_dataset(raw_df, fold_test_ids, normalizer=normalizer, fit_normalizer=False)

        train_loader = make_loader(train_dataset, batch_size, train_mode=True)
        val_loader = make_loader(val_dataset, batch_size, train_mode=False)
        test_loader = make_loader(test_dataset, batch_size, train_mode=False)

        model = build_model(model_name)
        elogger = logger.Logger(f"run_log_{model_name}_fold_{fold}")
        checkpoint_path = output_dir / f"{model_name}_fold_{fold}_best.pth"

        train(
            model,
            elogger,
            train_loader,
            val_loader,
            test_loader,
            epochs,
            batch_size,
            lr=lr,
            device=device,
            checkpoint_path=checkpoint_path,
            evaluate_test=False,
        )

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        metrics, predictions = evaluate(model, test_loader, device=device, return_details=True)
        metrics["fold"] = fold
        metrics["validation_scope"] = "internal_subject_level_cross_validation"
        fold_metrics.append(metrics)

        predictions["fold"] = fold
        predictions["model"] = model_name
        prediction_frames.append(predictions)
        predictions.to_csv(output_dir / f"{model_name}_fold_{fold}_predictions.csv", index=False)

        print(
            f"Fold {fold}: "
            f"Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, Sens={metrics['sensitivity']:.4f}, "
            f"Spec={metrics['specificity']:.4f}, PPV={metrics['ppv']:.4f}, NPV={metrics['npv']:.4f}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_metric_dicts(fold_metrics)
    pooled_metrics = compute_binary_metrics(
        all_predictions["y_true"],
        all_predictions["y_prob"],
        bootstrap_iters=bootstrap_iters,
        random_state=seed,
    )
    pooled_metrics["model"] = model_name
    pooled_metrics["validation_scope"] = "internal_subject_level_cross_validation"
    pooled_metrics_df = pd.DataFrame([pooled_metrics])
    subgroup_df = evaluate_default_subgroups(all_predictions, min_count=10)
    calibration_df = calibration_table(all_predictions["y_true"], all_predictions["y_prob"])
    decision_curve_df = decision_curve_table(all_predictions["y_true"], all_predictions["y_prob"])

    metrics_df.to_csv(output_dir / f"{model_name}_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / f"{model_name}_summary_metrics.csv", index=False)
    pooled_metrics_df.to_csv(output_dir / f"{model_name}_pooled_metrics_with_ci.csv", index=False)
    all_predictions.to_csv(output_dir / f"{model_name}_all_predictions.csv", index=False)
    subgroup_df.to_csv(output_dir / f"{model_name}_subgroup_metrics.csv", index=False)
    calibration_df.to_csv(output_dir / f"{model_name}_calibration_table.csv", index=False)
    decision_curve_df.to_csv(output_dir / f"{model_name}_decision_curve.csv", index=False)

    print("\nInternal cross-validation summary:")
    print(summary_df)
    print(
        "\nNote: No independent external validation cohort is available in this project. "
        "These results should be described as subject-level internal cross-validation."
    )
    return {
        "fold_metrics": metrics_df,
        "summary": summary_df,
        "pooled_metrics": pooled_metrics_df,
        "predictions": all_predictions,
        "subgroups": subgroup_df,
        "calibration": calibration_df,
        "decision_curve": decision_curve_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run subject-level k-fold evaluation for DSTLF and baselines.")
    parser.add_argument("--file-path", default="data2.parquet")
    parser.add_argument("--model-name", default="dstlf", choices=["dstlf", "bilstm", "transformer", "litemv"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs/acm_health_eval")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_k_fold(
        args.file_path,
        model_name=args.model_name,
        k=args.folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap_iters=args.bootstrap_iters,
    )
