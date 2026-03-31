'''
文件名: run_svm_baseline.py
描述: 使用 SVM/随机森林 对 Parquet 数据进行基线测试，并计算 Accuracy、Sensitivity、Specificity、F1-Score 和 AUC 等指标。
作者: yhw
日期: 2026-02
备注:
- 这个脚本专注于使用 SVM（或随机森林）对静态特征进行分类，作为深度学习模型的基线对比。'''
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class ParquetDataset(Dataset):
    def __init__(self, file_path, normalize=True):
        """
        Args:
            file_path: Parquet 文件路径
            mode: 'train' (开启数据增强) 或 'test'/'val' (关闭数据增强)
            normalize: 是否执行 Z-score 归一化
        读取 Parquet 文件并构建数据集
        """
   
        # 1. 读取 Parquet
        # engine='pyarrow' 或 'fastparquet'，通常默认即可
        self.df = pd.read_parquet(file_path).drop(columns=[ 'moca_score'])
        
        # 打印一下读到了多少特征，方便核对
        print(f"成功加载 Parquet 数据: {len(self.df)} 条样本")
        print(f"包含列名: {list(self.df.columns)}")
        
        # 2. 预处理
        # 确保 ex, ey 是 list 类型 (有些 parquet 存的是 string 或 numpy array)
        # 如果存的是 numpy array，tolist() 是必要的
        # 如果存的是 string (如 "[1.1, 2.2]"), 需要 json.loads 解码 (视你存的方式而定)
        self.exclude_cols = ['ex', 'ey', 'label', 'driverID', 'evaluation_id', 'diagnose', 
                             'sex', 'hand'] 
        self.cont_cols = [c for c in self.df.columns if c not in self.exclude_cols and pd.api.types.is_numeric_dtype(self.df[c])]
        print(f"检测到 {len(self.cont_cols)} 个连续特征需要归一化: {self.cont_cols[:5]}...")
        self.normalize = normalize
        if self.normalize:
            self._apply_normalization()
            
    def _apply_normalization(self):
        print("正在执行数据归一化 (Z-Score)...")
        
        # 1. 静态属性归一化 (直接在 DataFrame 上修改)
        # 使用 sklearn 的 StandardScaler
        scaler = StandardScaler()
        # 处理可能的 NaN (填充 0 或 均值，这里简单填充 0)
        self.df[self.cont_cols] = self.df[self.cont_cols].fillna(0)
        
        # fit_transform 会计算均值方差并转换数据
        self.df[self.cont_cols] = scaler.fit_transform(self.df[self.cont_cols])
        print("✅ 静态属性归一化完成")

        # 2. 轨迹坐标归一化 (计算全局均值和标准差)
        # 注意：ex 和 ey 是列表，不能直接用 scaler。我们需要展平所有轨迹点来计算。
        # 这里为了效率，我们先随机采样一部分数据估算，或者全量计算（取决于数据量）
        
        # 获取所有轨迹点的列表
        all_ex = np.concatenate(self.df['ex'].values)
        all_ey = np.concatenate(self.df['ey'].values)
        
        # 计算全局统计量
        self.ex_mean = all_ex.mean()
        self.ex_std = all_ex.std() + 1e-6 # 防止除0
        
        self.ey_mean = all_ey.mean()
        self.ey_std = all_ey.std() + 1e-6
        
        print(f"✅ 轨迹归一化参数计算完成:")
        print(f"   ex: mean={self.ex_mean:.4f}, std={self.ex_std:.4f}")
        print(f"   ey: mean={self.ey_mean:.4f}, std={self.ey_std:.4f}")

        # 释放内存
        del all_ex, all_ey

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取 DataFrame 的一行 (Series 对象)
        row = self.df.iloc[idx]
        
        # === 1. 构建 Attr 字典 === 
        attr_dict = {}
        for col_name in self.cont_cols:
            # 这里的 row[col_name] 已经是归一化后的数据了
            attr_dict[col_name] = float(row[col_name])
        # 处理离散特征（不归一化的）
        if 'hand' in row:
            val = row['hand']
            # 判断是否为 NaN (Pandas中 NaN != NaN) 或者使用 pd.isna(val)
            if pd.isna(val):
                attr_dict['hand'] = 0  # <--- 遇到空值，填充默认值 0
            else:
                attr_dict['hand'] = int(val)
        
        # 对 sex 也做同样的保护，防止报错
        if 'sex' in row:
            val = row['sex']
            if pd.isna(val):
                attr_dict['sex'] = 0   # <--- 遇到空值，填充默认值 0
            else:
                attr_dict['sex'] = int(val)
        # === 2. 构建 Traj 字典 ===
        # 获取原始列表
        raw_ex = row['ex'].tolist() if hasattr(row['ex'], 'tolist') else list(row['ex'])
        raw_ey = row['ey'].tolist() if hasattr(row['ey'], 'tolist') else list(row['ey'])

        # === 计算运动学特征 (关键步骤) ===
        # 速度 (一阶差分)
        # np.diff 后长度会少 1，我们需要补一个 0 保持长度一致
        vel_x = np.diff(raw_ex, prepend=raw_ex[0])
        vel_y = np.diff(raw_ey, prepend=raw_ey[0])
        speed = np.sqrt(vel_x**2 + vel_y**2)
        # 加速度 (速度的差分)
        acc_x = np.diff(vel_x, prepend=vel_x[0])
        acc_y = np.diff(vel_y, prepend=vel_y[0])
        acc_mag = np.sqrt(acc_x**2 + acc_y**2) # 加速度大小
        speed = np.log1p(speed) 
        acc_mag = np.log1p(acc_mag)

        # 转换为 numpy 以便进行数学运算
        np_ex = np.array(raw_ex, dtype=np.float32)
        np_ey = np.array(raw_ey, dtype=np.float32)
        # *** 实时应用轨迹归一化 ***
        if self.normalize:
            np_ex = (np_ex - self.ex_mean) / self.ex_std
            np_ey = (np_ey - self.ey_mean) / self.ey_std
            speed = (speed - speed.mean()) / (speed.std() + 1e-6)
            acc_mag = (acc_mag - acc_mag.mean()) / (acc_mag.std() + 1e-6)
            
        # 处理 NaN (轨迹中可能存在无效点)
        np_ex = np.nan_to_num(np_ex, nan=0.0)
        np_ey = np.nan_to_num(np_ey, nan=0.0)
        traj_dict = {
            'ex': np_ex.tolist(),
            'ey': np_ey.tolist(),
            'speed': speed.tolist(),       # 新增
            'acc': acc_mag.tolist()        # 新增
        }
        # === 3. ID 和 Label ===
        # 兼容 driverID 或 evaluation_id
        driver_id = int(row.get('driverID', row.get('evaluation_id', 0)))
        label = int(row['diagnose'])
        
        return {
            'attr': attr_dict,
            'traj': traj_dict,
            'label': label,
            'driverID': driver_id 
        }


def extract_tabular_data(dataset):
    """
    将 ParquetDataset 中的字典格式转换为 sklearn 可用的二维矩阵 (X, y)
    """
    print("正在从 Dataset 中提取表格数据用于 SVM 训练...")
    X =[]
    y =[]
    
    # 遍历整个数据集
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        attr_dict = sample['attr']
        
        # 为了保证每次提取的特征顺序一致，我们对 keys 进行排序
        feature_keys = sorted(list(attr_dict.keys()))
        
        # 将字典转换为特征列表
        feature_vector = [attr_dict[k] for k in feature_keys]
        
        X.append(feature_vector)
        y.append(sample['label'])
        
    return np.array(X), np.array(y), feature_keys

def run_svm_baseline(file_path):
    # 1. 初始化你的数据集 (这里会自动应用你写好的 Z-score 归一化)
    # 注意：只提取统计特征，自动忽略变长的 ex, ey, speed, acc 序列
    dataset = ParquetDataset(file_path=file_path, normalize=True)
    
    # 2. 提取 X (特征) 和 y (标签)
    X, y, feature_names = extract_tabular_data(dataset)
    print(f"\n提取完成！特征矩阵 X 形状: {X.shape}, 标签 y 形状: {y.shape}")
    print(f"使用的特征数量: {len(feature_names)}")
    
    # 3. 设置 5-Fold 分层交叉验证 (与 DL 模型保持公平一致)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 记录评价指标的列表
    acc_list = []
    f1_list =[]
    sens_list = []  # Sensitivity (Recall for MCI)
    spec_list =[]  # Specificity (Recall for HC)
    auc_list =[]
    
    print(f"\n🚀 开始 {k_folds}-Fold SVM 基线测试...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # 划分训练集和测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 4. 初始化 SVM 模型
        # 参数说明：
        # kernel='rbf': 径向基核函数，处理非线性关系最常用
        # class_weight='balanced': 自动处理类别不平衡（对应你深度学习里的 weights）
        # probability=True: 允许输出概率，为了计算 AUC 必备
        # svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
        svm_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        
        # 5. 训练模型
        svm_model.fit(X_train, y_train)
        
        # 6. 预测与评估
        y_pred = svm_model.predict(X_test)
        y_prob = svm_model.predict_proba(X_test)[:, 1] # 获取预测为正类(1)的概率
        
        # 计算基础指标
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob)
        
        # 计算敏感度(Sensitivity)和特异度(Specificity)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # 患病被正确找出的概率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # 健康被正确判断的概率
        
        # 打印当前折结果
        print(f"Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, AUC={auc:.4f}")
        
        # 存入列表
        acc_list.append(acc)
        f1_list.append(f1)
        sens_list.append(sensitivity)
        spec_list.append(specificity)
        auc_list.append(auc)

    # 7. 汇总与输出最终报告（直接可填入论文表格）
    print("\n=======================================================")
    print("📊 SVM Baseline 最终结果 (5-Fold CV 平均值 ± 标准差):")
    print(f"Accuracy    : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Sensitivity : {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}")
    print(f"Specificity : {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
    print(f"F1-Score    : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"AUC         : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print("=======================================================")

if __name__ == "__main__":
    # 替换为你的 parquet 文件路径
    FILE_PATH = "data2.parquet" 
    run_svm_baseline(FILE_PATH)