import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split 
import torch
from torch.utils.data import DataLoader, Subset
# 引入你现有的模块
from DSTLF import ParquetDataset, collate_fn, BucketBatchSampler, train, evaluate 
import models.DeepTTE as DeepTTE
import models.Baseline_LSTM as BaselineLSTM
import models.Baseline_Transformer as BaselineTransformer
import models.Baseline_LITEMV as BaselineLITEMV

import logger

def run_k_fold(file_path, k=5, batch_size=32, epochs=50):
    # 1. 加载全量数据
    # 注意：这里不需要归一化，归一化最好在 fold 内部做，但为了简化，全局归一化也可接受
    full_dataset = ParquetDataset(file_path=file_path, normalize=True)
    df = full_dataset.df

    id_label_map = df.groupby('evaluation_id')['diagnose'].first()
    all_ids = np.array(sorted(id_label_map.index.tolist()))
    id_labels = np.array([id_label_map[subj_id] for subj_id in all_ids])
    # 2. 获取所有唯一的 Subject ID
    #all_ids = np.array(sorted(full_dataset.df['evaluation_id'].unique().tolist()))
    
    # 3. K-Fold 分割器
    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    
    acc_list, f1_list, auc_list, sens_list, spec_list = [], [], [], [], [] # 记录每一折的结果
    
    print(f"🚀 开始 {k}-Fold 交叉验证...")
    
    for fold, (train_idx_ids, test_idx_ids) in enumerate(kf.split(all_ids, id_labels)):
        print(f"\n========== Fold {fold+1}/{k} ==========")
        
        # 获取当前折的 ID
        fold_train_ids = all_ids[train_idx_ids]
        fold_test_ids = all_ids[test_idx_ids]

        fold_train_labels_all = id_labels[train_idx_ids]
        
        # 进一步从 Training IDs 中分出 Validation IDs (用于早停)
        # 比如取 20% 做 Val

        fold_train_ids, fold_val_ids = train_test_split(fold_train_ids, test_size=0.2, random_state=10, stratify=fold_train_labels_all)
        
        # 映射回 DataFrame 的索引 (Indices)
        train_indices = full_dataset.df.index[full_dataset.df['evaluation_id'].isin(fold_train_ids)].tolist()
        val_indices = full_dataset.df.index[full_dataset.df['evaluation_id'].isin(fold_val_ids)].tolist()
        test_indices = full_dataset.df.index[full_dataset.df['evaluation_id'].isin(fold_test_ids)].tolist()
        
        # 构建 Subset
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
        
        # 构建 Loader (Train用分桶，Val/Test不用)
        train_sampler = BucketBatchSampler(train_set, batch_size)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 初始化模型 (每一折都要重新初始化！)
        model = DeepTTE.Net(
            num_classes=2, 
            num_filter=32, 
            hidden_size=48, 
            num_fc_layers=1, 
            dropout_p=0.5
        )
        # 实验 B: 跑 Vanilla BiLSTM 基线
        # model = BaselineLSTM.Net(num_classes=2, hidden_size=48, dropout_p=0.5)
        # 实验 C: 跑 Transformer 基线
        # model = BaselineTransformer.Net(num_classes=2, d_model=64, num_layers=2, dropout_p=0.5)
        # 实验 D: 跑 LITEMV 基线
        model = BaselineLITEMV.Net(num_classes=2, dropout_p=0.5)
        # 初始化 Logger
        elogger = logger.Logger(f"run_log_fold_{fold+1}")
        
        # 训练 (复用你现有的 train 函数)
        # 注意：train 函数内部必须包含 load_best_model 的逻辑
        train(model, elogger, train_loader, val_loader, test_loader, epochs, batch_size, lr=5e-3)
        
        # 加载本折的最优模型进行测试
        model.load_state_dict(torch.load("best_model.pth")) # 假设 train 函数保存为这个名字
        acc, f1, auc, sens, spec = evaluate(model, test_loader, device=torch.device('cuda'))
        #acc, f1 = evaluate(model, test_loader, device=torch.device('cuda'))
        
        print(f"✅ Fold {fold+1} Result: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        sens_list.append(sens)
        spec_list.append(spec)

    # 输出平均结果
    print("📊 DSTLF (Ours) 5-Fold CV 最终结果 (均值 ± 标准差):")
    print(f"Accuracy    : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Sensitivity : {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}")
    print(f"Specificity : {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
    print(f"F1-Score    : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"AUC         : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print("=======================================================")

if __name__ == "__main__":
    run_k_fold("data2.parquet") # 替换你的路径