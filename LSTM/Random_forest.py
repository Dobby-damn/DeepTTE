import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,GridSearchCV
import shap
import seaborn as sns
import os
from datetime import timedelta
import datetime as date
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import entropy
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV

def process_time_column(df):
    '''统一处理两种时间格式并重新分配时间戳'''
    # 第一种格式‘年/月/日 时:分:秒’
    try:
        df['time'] = pd.to_datetime(df['time'], format = '%Y/%m/%d %H:%M:%S')
    except:
        print("第一种时间格式解析失败，尝试第二种格式")
        # 第二种格式‘年-月-日 时:分:秒’
        df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S:%f')
    # 对每个evaluation_id处理重复时间
    def redistribute_time(group):
        time_counts = group['time'].value_counts()
        duplicate_times = time_counts[time_counts > 1].index
        
        for time_val in duplicate_times:
            # 找到相同时间的所有行
            dup_mask = group['time'] == time_val
            n_duplicates = sum(dup_mask)
            
            if n_duplicates > 1:
                time_increment = timedelta(seconds=1) / n_duplicates
                new_times = [time_val + i*time_increment for i in range(n_duplicates)]
                group.loc[dup_mask, 'time'] = new_times
        
        return group
    
    return df.groupby('evaluation_id', group_keys=False).apply(redistribute_time)

def process_duplicate_times(group):
    # 找出重复的时间点
    time_counts = group['time'].value_counts()
    duplicate_times = time_counts[time_counts > 1].index
    
    for time_val in duplicate_times:
        # 获取相同时间的所有行
        dup_mask = group['time'] == time_val
        n_duplicates = sum(dup_mask)
        
        if n_duplicates > 1:
            # 计算时间增量（1分钟均匀分配）
            time_increment = pd.Timedelta(minutes=1) / n_duplicates
            
            # 为重复时间点分配新时间
            new_times = [time_val + i*time_increment 
                        for i in range(n_duplicates)]
            
            # 保持原始行顺序更新时间
            group.loc[dup_mask, 'time'] = new_times
    
    return group

# 1. 数据加载与合并
print("正在加载数据...")
iddiag_path = r"D:/Code/DeepTTE/LSTM/iddiag.csv"
traj_csv_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(1).csv"
traj_xlsx_path = r"D:/Code/DeepTTE/LSTM/连线测试轨迹(2).xlsx"

# 加载被试基本信息
df_demo = pd.read_csv(iddiag_path)

# 加载并合并两个轨迹文件
if os.path.exists(traj_csv_path):
    df_traj1 = pd.read_csv(traj_csv_path, engine='python')
    df_traj1 = df_traj1.drop('create_time', axis=1) 
    df_traj1['time'] = pd.to_datetime(df_traj1['time'], format='%Y/%m/%d %H:%M')
    # 处理重复时间戳
    df_traj1 = df_traj1.groupby('evaluation_id', group_keys=False).apply(process_duplicate_times)
    #df_traj1 = df_traj1.drop('time', axis=1) 
    # 4. 转换为时间戳（秒）
    df_traj1['timestamp'] = df_traj1['time'].astype('int64') / 1e9
else:
    raise FileNotFoundError(f"未找到文件: {traj_csv_path}")

if os.path.exists(traj_xlsx_path):
    df_traj2 = pd.read_excel(traj_xlsx_path)
    df_traj2 = df_traj2.drop('create_time', axis=1) 
    df_traj2 = process_time_column(df_traj2)
    #df_traj2 = df_traj2.drop('time', axis=1) 
    df_traj2['timestamp'] = df_traj2['time'].astype('int64') / 1e9
else:
    raise FileNotFoundError(f"未找到文件: {traj_xlsx_path}")

# 合并轨迹数据
df_traj = pd.concat([df_traj1, df_traj2], ignore_index=True)
print(f"合并后的轨迹数据量: {len(df_traj)}条记录")

# 2. 数据预处理 - 确保时间列是数值类型
print("\n正在预处理数据...")
df_traj['time'] = pd.to_numeric(df_traj['time'], errors='coerce')
df_traj = df_traj.dropna(subset=['time'])  # 删除无效时间记录

# 3. 特征工程 - 计算轨迹特征
print("\n正在计算轨迹特征...")
def calculate_curvature(group):
    group = group.sort_values('timestamp')
    
    # 获取坐标和时间序列
    x = group['ex'].values
    y = group['ey'].values
    t = group['timestamp'].values
    
    # 计算一阶导数（速度）
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    
    # 计算二阶导数（加速度）
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)
    
    # 计算曲率
    eps = 1e-10
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt**2 + dy_dt**2)**1.5 + eps)
    
    # 返回曲率相关特征
    return pd.Series({
        'mean_curvature': float(np.nanmean(curvature)),
        'max_curvature': float(np.nanmax(curvature)),
        'curvature_std': np.nanstd(curvature),
        'high_curvature_points': (curvature > 0.5).sum()  # 阈值可根据数据调整
    })

def calculate_trajectory_features(group):
    # 确保数据按时间排序
    group = group.sort_values('time')
    
    # 计算坐标变化
    dx = group['ex'].diff()
    dy = group['ey'].diff()
    
    # 计算速度和加速度
    distance = np.sqrt(dx**2 + dy**2)
    time_diff = group['time'].diff()
    speed = distance / (time_diff.replace(0, np.nan) + 1e-6)  # 避免除以0
    # 曲率计算
    curvature_features = calculate_curvature(group)
    l = len(group)
    
    # print(f"当前evaluation_id: {group['evaluation_id'].iloc[0]}, 轨迹点数量: {l}")
    # 新增动态特征
    if len(group) > 2:
        # 加速度
        acceleration = np.diff(speed) / (time_diff[1:] + 1e-6)
        # 急动度（加速度变化率）
        jerk = np.diff(acceleration) / (time_diff[2:] + 1e-6)
        # 方向变化
        directions = np.arctan2(dy, dx)
        direction_changes = np.sum(np.abs(np.diff(directions)) > np.pi/4)
    else:
        acceleration = jerk = np.array([0])
        direction_changes = 0

    # 新增复杂度特征
    total_distance = np.sum(distance)
    straight_distance = np.sqrt((group['ex'].iloc[-1] - group['ex'].iloc[0])**2 + 
                               (group['ey'].iloc[-1] - group['ey'].iloc[0])**2)

    # 时间相关特征
    pause_mask = (speed < 0.1) & (distance < 1)
    pause_durations = time_diff[pause_mask]


    # 返回特征
    base_features = pd.Series({
        'mean_speed': float(np.nanmean(speed)),
        'std_speed': float(np.nanstd(speed)),
        'total_distance': float(np.nansum(distance)),
        'total_time': float(np.nansum(time_diff)),
        'pause_count': int((speed < 0.1).sum()),  # 速度低于阈值视为停顿
        'max_speed': float(np.nanmax(speed)),
        'min_speed': float(np.nanmin(speed)),
        'speed_variation': float(np.nanstd(speed) / (np.nanmean(speed) + 1e-6)),
        'point_count': int(l),  # 轨迹点数量
        # 新增特征
        'complexity_ratio': total_distance / (straight_distance + 1e-6),
        'direction_changes': direction_changes,
        'mean_acceleration': np.nanmean(acceleration) if len(acceleration) > 0 else 0,
        'jerk_std': np.nanstd(jerk) if len(jerk) > 0 else 0,
        'max_pause_duration': np.max(pause_durations) if len(pause_durations) > 0 else 0,
        'pause_time_ratio': np.sum(pause_durations) / np.sum(time_diff) if np.sum(time_diff) > 0 else 0,
        # 'entropy_speed': entropy(np.histogram(speed, bins=10)[0]) if len(speed) > 1 else 0,
    })
    return pd.concat([base_features, curvature_features])

# 按evaluation_id分组计算特征
traj_features = df_traj.groupby('evaluation_id').apply(calculate_trajectory_features)
print(traj_features['point_count'].head())
# 直接删除索引中的 'evaluation_id'
traj_features = traj_features.reset_index()  # 清空索引

# 4. 合并所有数据
print("\n正在合并数据集...")
df = pd.merge(df_demo, traj_features, left_on='evaluation_id', right_on='evaluation_id')

# 5. 定义目标变量
df['cognitive_impairment'] = (df['moca_score'] <= 25).astype(int)

# 转换步骤
print(df['point_count'].dtypes)
df['point_count'] = (
    pd.to_numeric(df['point_count'], errors='coerce')  # 转数字，无效→NaN
    .fillna(0)                                         # 填充 NaN
    .astype('int64')                                   # 转整数
)
c = df['point_count']
# print(c.dtypes)

# print(c.head())
df['point_count'] = pd.to_numeric(df['point_count'], errors='coerce').astype('int64')

print(df.head())

# 6. 特征选择
features = [
    'age', 'edu_year', 'habit1',  # 人口统计'edu_year','sex'
    'mean_speed', 'std_speed', 'total_distance',  # 轨迹特征
    'total_time', 'pause_count', 'speed_variation', 'point_count',
    'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points',  # 曲率特征
    'complexity_ratio', 'direction_changes', 'mean_acceleration',
    'jerk_std', 'max_pause_duration', 'pause_time_ratio', #'entropy_speed',
]

X = df[features]
y = df['cognitive_impairment']

# 检查数据平衡性
print("\n类别分布:")
print(y.value_counts(normalize=True))

# 7. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=46
)
# 类别平衡
# ros = RandomOverSampler(
#     sampling_strategy='minority',  # 只对少数类过采样
#     random_state=42
# )
# X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
# sm = SMOTE(sampling_strategy='minority', random_state=42)
# X_train,y_train = sm.fit_resample(X_train, y_train)
# print(f"过采样后训练集类别分布: {pd.Series(y_train).value_counts(normalize=True).round(4) * 100}")
# 参数网格#######################################################################################################
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0, 0.1],
    'scale_pos_weight': [len(y_train[y_train==0])/len(y_train[y_train==1])]  # 自动计算权重
}
# 网格搜索
xgb = XGBClassifier(
    n_estimators=300,  # 树数量
    objective='binary:logistic',  # 二分类目标
    eval_metric='auc',   # 评估指标
    # use_label_encoder=False,   # 新版本不在需要
    random_state=42
)
# 网格搜索（5折交叉验证）
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',    # 优化目标
    cv=5,
    n_jobs=-1   # 并行计算
)
grid_search.fit(X_train, y_train)    # 使用过采样数据训练
best_xgb = grid_search.best_estimator_    # 最优模型
print(f"最优超参数: {grid_search.best_params_}")
# 概率校准
calibrated_xgb = CalibratedClassifierCV(
    best_xgb, 
    method='isotonic',     # 等渗回归校准
    cv=5,    # 交叉验证校准
)
calibrated_xgb.fit(X_train, y_train)    # 用训练集校准
# 阈值优化
from sklearn.metrics import precision_recall_curve

y_proba_train = calibrated_xgb.predict_proba(X_train)[:, 1]    # 训练集正类概率
precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)    # 计算F1
optimal_idx = np.argmax(f1_scores[:-1])  # 最后一个是边界值,排除边界值
# 根据最优索引获取对应的最优阈值（使F1最大的阈值）
optimal_threshold = thresholds[optimal_idx]
# 获取校准后模型在测试集上对正类的预测概率
print("\n在测试集上评估模型...")
y_proba_test = calibrated_xgb.predict_proba(X_test)[:, 1]
# 应用最优阈值将概率转换为类别预测：概率>=最优阈值则预测为1（正类），否则为0（负类）
y_pred_test = (y_proba_test >= optimal_threshold).astype(int)

print("\n" + "="*50)
print("模型性能评估:")
print("="*50)
print(classification_report(y_test, y_pred_test))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_test):.4f}")

# 10. 交叉验证
print("\n进行交叉验证...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_xgb, X, y, cv=cv, scoring='roc_auc')
print("\n交叉验证结果:")
print(f"5折交叉验证 AUC-ROC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# 11. 特征重要性分析
print("\n计算特征重要性...")
# feature_importance = pd.DataFrame({
#     'Feature': features,
#     'Importance': rf.feature_importances_
# }).sort_values('Importance', ascending=False)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('importance of feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 12. SHAP值分析
print("\n计算SHAP值...")
plt.figure(figsize=(10, 6))

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_test)

shap_values_single_output = shap_values  # 形状变为 (102, 15)

# SHAP摘要图
shap.summary_plot(shap_values_single_output, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar.png')
plt.show()

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_dot.png')
plt.show()

# 13. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['normal', 'abnormal'], 
            yticklabels=['normal', 'abnormal'])
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.title('Confusion matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 14. 保存重要结果
results = pd.DataFrame({
    'evaluation_id': df.loc[X_test.index, 'evaluation_id'],
    'true_label': y_test,
    'predicted_label': y_pred_test,
    'probability': y_proba_test,
})

results.to_csv('prediction_results.csv', index=False)
print("\n分析完成！结果已保存到当前目录")