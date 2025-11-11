import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ----------------------
# 1. 数据加载与预处理（示例）
# ----------------------
def preprocess_data(samples):
    # 提取轨迹序列和标量特征
    trajectories = []
    scalars = []
    labels = []
    
    for sample in samples:
        # 处理轨迹：组合ex和ey，归一化，固定长度4000
        ex = np.array(sample['ex'])
        ey = np.array(sample['ey'])
        traj = np.stack([ex, ey], axis=1)  # 形状 (N, 2)
        
        # 归一化（使用训练集统计量，这里简化为样本内归一化，实际应全局计算）
        traj_mean = np.mean(traj, axis=0)
        traj_std = np.std(traj, axis=0) + 1e-8  # 避免除零
        traj_norm = (traj - traj_mean) / traj_std
        
        # 固定长度4000
        if len(traj_norm) < 4000:
            traj_padded = np.pad(traj_norm, ((0, 4000 - len(traj_norm)), (0, 0)), mode='constant')
        else:
            traj_padded = traj_norm[:4000]
        trajectories.append(traj_padded)
        
        # 处理标量特征
        scalar = [
            sample['dist'],
            sample['time'],
            sample['driverID'],
            sample['pause_count'],
            sample['mean_speed'],
            sample['curvature_std']
        ]
        scalars.append(scalar)
        labels.append(sample['label'])
    
    # 标量特征标准化
    scalar_scaler = StandardScaler()
    scalars = scalar_scaler.fit_transform(scalars)
    
    return np.array(trajectories), np.array(scalars), np.array(labels)


# 假设samples是你的数据列表（包含多个样本字典）
# trajectories: (num_samples, 4000, 2)
# scalars: (num_samples, 6)
# labels: (num_samples,)
trajectories, scalars, labels = preprocess_data(samples)

# 划分数据集
X_traj_train, X_traj_val, X_scalar_train, X_scalar_val, y_train, y_val = train_test_split(
    trajectories, scalars, labels, test_size=0.2, random_state=42
)
X_traj_val, X_traj_test, X_scalar_val, X_scalar_test, y_val, y_test = train_test_split(
    X_traj_val, X_scalar_val, y_val, test_size=0.5, random_state=42
)


# ----------------------
# 2. 模型构建
# ----------------------
# 轨迹分支（1D-CNN）
traj_input = Input(shape=(4000, 2), name='trajectory_input')
x = Conv1D(32, kernel_size=5, activation='relu')(traj_input)  # 捕捉局部轨迹特征
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(64, kernel_size=5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
traj_features = Flatten()(x)


# 标量特征分支
scalar_input = Input(shape=(6,), name='scalar_input')
y = Dense(32, activation='relu')(scalar_input)
y = Dropout(0.3)(y)  # 防止过拟合
y = Dense(32, activation='relu')(y)
scalar_features = Dropout(0.3)(y)


# 融合分支
merged = Concatenate()([traj_features, scalar_features])
z = Dense(128, activation='relu')(merged)
z = Dropout(0.3)(z)
output = Dense(1, activation='sigmoid')(z)  # 二分类输出


# 构建模型
model = Model(inputs=[traj_input, scalar_input], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ----------------------
# 3. 模型训练
# ----------------------
history = model.fit(
    x=[X_traj_train, X_scalar_train],
    y=y_train,
    batch_size=32,
    epochs=20,
    validation_data=([X_traj_val, X_scalar_val], y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),  # 早停防止过拟合
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)  # 学习率衰减
    ]
)


# ----------------------
# 4. 模型评估
# ----------------------
test_loss, test_acc = model.evaluate([X_traj_test, X_scalar_test], y_test)
print(f"Test Accuracy: {test_acc:.4f}")