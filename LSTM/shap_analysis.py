import torch
import shap
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from LSTM.train import TrajectoryDataset, LSTMClassifier 
print(torch.__version__)

"""
SHAP 分析脚本
说明：
1. 需要在训练完成后运行
2. 默认加载 model.pth
3. 对静态特征做 SHAP（序列 SHAP 成本过高）
4. 输出全局 SHAP 值
['Unnamed: 0', 'record_id', 'age', 'sex', 'hand', 'edu_year', 'edugrade', 'occupation',
 'habit1', 'habit2', 'habit3', 'habit4', 'habit5', 'habit6',
   'ssoid', 'oppo_sub', 'eduYears', 'evaluation_id', 'evaluation_type',
     'data_type', 'completed', 'correctConnections', 'incorrectConnections', 'clickCount', 
     'noTouchCount', 'gameDuration', 'type', 'connTime', 'show_T', 'show_D', 'ConnTime_ET1', 
     'ConnTime_T1', 'ConnTime_D', 'ConnTime_T', 'education', 'mmse_s', 'adl_s', 'phq_s', 'sas_s', 
     'ab42', 'ab40', 'ab42_40', 'pt217', 'pt217_ab42', 'cmbctid', 'Unnamed: 3', 'mean_speed', 
     'std_speed', 'total_distance', 'total_time', 'pause_count', 'max_speed', 'min_speed', 
     'speed_variation', 'point_count', 'complexity_ratio', 'direction_changes', 'mean_acceleration', 
     'jerk_std', 'max_pause_duration', 'pause_time_ratio', 'mean_curvature', 'max_curvature', 
     'curvature_std', 'high_curvature_points']
"""

# -------------------------------
# 配置
# -------------------------------
MODEL_PATH = "model.pth"
DATA_PARQUET = "data.parquet"
TRAJ_CSV = "轨迹测试数据(1).csv"
TRAJ_XLSX = "轨迹测试数据(2).xlsx"
TRAJ_NEW = "traj_path_new.csv"
STATIC_FEATURES_PATH = "static_features.npy"
PARQUET_PATH = "data.parquet"   # 你的静态特征来源
BATCH_SIZE = 32

def generate_static_features():
    print("Generating static_features.npy from data.parquet ...")

    df = pd.read_parquet(PARQUET_PATH)

    # 根据你训练时用到的静态特征列来选
    # ⚠ 如果你有具体列名，我可以帮你自动生成
    static_cols = [ 'record_id', 'age', 'sex', 'hand', 'edu_year', 'edugrade', 'occupation', 'habit1', 'habit2', 'habit3', 'habit4', 'habit5', 'habit6', 'hkbcscore', 'moca_score', 'diagnose', 'ssoid', 'oppo_sub', 'eduYears', 'evaluation_id', 'evaluation_type', 'data_type', 'completed', 'correctConnections', 'incorrectConnections', 'clickCount', 'noTouchCount', 'gameDuration', 'type', 'connTime', 'show_T', 'show_D', 'ConnTime_ET1', 'ConnTime_T1', 'ConnTime_D', 'ConnTime_T', 'education', 'mmse_s', 'moca_s', 'adl_s', 'phq_s', 'sas_s', 'ab42', 'ab40', 'ab42_40', 'pt217', 'pt217_ab42', 'cmbctid', 'Unnamed: 3', 'mean_speed', 'std_speed', 'total_distance', 'total_time', 'pause_count', 'max_speed', 'min_speed', 'speed_variation', 'point_count', 'complexity_ratio', 'direction_changes', 'mean_acceleration', 'jerk_std', 'max_pause_duration', 'pause_time_ratio', 'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points']  # ← 修改为你的列名

    static_feats = df[static_cols].values.astype(np.float32)

    np.save(STATIC_FEATURES_PATH, static_feats)
    print("Saved static_features.npy")

# 如果不存在 static_features.npy，就自动生成
if not os.path.exists(STATIC_FEATURES_PATH):
    generate_static_features()


npy_data = np.load(STATIC_FEATURES_PATH)  # 例如：np.load("/data/trajectory_features.npy")

# 查看数据基本信息（验证是否加载成功）
print("数据类型:", type(npy_data))  # 输出 <class 'numpy.ndarray'>（.npy 本质是 NumPy 数组）
print("数据形状:", npy_data.shape)  # 输出数组维度（如 (1000, 65) 表示 1000 行、65 列，对应你的 65 个特征）
print("数据类型:", npy_data.dtype)  # 输出数据类型（如 float64、int32）
print("\n前 5 行数据:")
print(npy_data[:5])  # 查看

def load_model(device):
    print("Loading model...")
    model = LSTMClassifier(
        input_size=9,
        static_dim=68,
        hidden_size=64,
        num_layers=3,
        num_classes=2,
        dropout=0.2,
        bidirectional=True
    ).to(device)
    ckpt = torch.load(r'D:\Code\DeepTTE\checkpoints\best_model_epoch15.pt', map_location=device)
    # 如果 ckpt 是 { 'model_state_dict': ... }
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    print("模型加载成功!")
    return model

    model.load_state_dict(torch.load(r'D:\Code\DeepTTE\checkpoints\best_model_epoch15.pt', map_location=device))
    model.to(device)
    model.eval()
    return model


def load_static_features():
    print("Loading static features...")
    static_feats = np.load(STATIC_FEATURES_PATH)
    return static_feats


# -------------------------------
# SHAP 核心函数
# -------------------------------

class StaticModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_static_only(x)


def clean_feats(x):
    x = np.array(x, dtype=np.float32)
    x[np.isinf(x)] = np.nan
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x


def do_shap(model, static_feats, device):

    print("Cleaning static features...")
    static_feats = clean_feats(static_feats)

    # 选一小部分作为背景 + 测试
    background = static_feats[:50]
    test_samples = static_feats[:200]

    print("Preparing model wrapper...")
    wrapper = StaticModelWrapper(model).to(device)

    # Python function for SHAP
    def f(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            y = wrapper(x_tensor).detach().cpu().numpy()
        return y

    print("Creating SHAP SamplingExplainer...")
    explainer = shap.SamplingExplainer(f, background)

    print("Computing SHAP values (may take time)...")
    shap_values = explainer.shap_values(test_samples)

    print("SHAP finished.")
    return shap_values, explainer

def visualize_shap(static_feats, shap_values, feature_names):
    import shap
    import numpy as np

    # shap_values shape = (200, 68, 2)
    # 选择解释 output dim=0
    shap_values = np.array(shap_values)
    shap_values_0 = shap_values[:, :, 1]

    # 只匹配对应 200 条输入
    features = static_feats[:shap_values_0.shape[0]]

    print("Using features:", features.shape)
    print("Using shap_values:", shap_values_0.shape)

    shap.summary_plot(
        shap_values_0,
        features,
        feature_names=feature_names
    )


# -------------------------------
# 主流程
# -------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载模型
    model = load_model(device)

    # 加载静态特征
    static_feats = load_static_features()
    print("Cleaning static features...")
    static_feats = clean_feats(static_feats)
    static_feats = static_feats[:, :68]


    # 获得 SHAP 值
    shap_values, explainer = do_shap(model, static_feats, device)
    print("static_feats:", static_feats.shape)
    print("shap_values:", np.array(shap_values).shape)
    # 特征名（你要替换成真实的）
    feature_names = [f"feat_{i}" for i in range(static_feats.shape[1])]
    visualize_shap(static_feats, shap_values, feature_names)



    print("SHAP 完成！shap_values.npy 已保存。")


if __name__ == "__main__":
    main()
