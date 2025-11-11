import pandas as pd
import numpy as np
import os
from datetime import timedelta
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

class TrajectoryFeatureExtractor:
    def __init__(self, iddiag_path, traj_csv_path, traj_xlsx_path):
        # 文件路径
        self.iddiag_path = iddiag_path
        self.traj_csv_path = traj_csv_path
        self.traj_xlsx_path = traj_xlsx_path

        # 加载并处理数据
        self.df_demo = self._load_demo_data()
        self.df_traj = self._load_and_merge_trajectory_data()

    def _load_demo_data(self):
        """加载被试基本信息数据"""
        print("正在加载被试基本信息数据...")
        return pd.read_csv(self.iddiag_path)

    def _process_time_column(self, df):
        """处理时间列，统一两种时间格式"""
        try:
            df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S')
        except:
            print("第一种时间格式解析失败，尝试第二种格式")
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S:%f')

        def redistribute_time(group):
            time_counts = group['time'].value_counts()
            duplicate_times = time_counts[time_counts > 1].index

            for time_val in duplicate_times:
                dup_mask = group['time'] == time_val
                n_duplicates = sum(dup_mask)

                if n_duplicates > 1:
                    time_increment = timedelta(seconds=1) / n_duplicates
                    new_times = [time_val + i * time_increment for i in range(n_duplicates)]
                    group.loc[dup_mask, 'time'] = new_times

            return group
        
        return df.groupby('evaluation_id', group_keys=False).apply(redistribute_time)

    def _process_duplicate_times(self, group):
        """处理重复时间戳"""
        time_counts = group['time'].value_counts()
        duplicate_times = time_counts[time_counts > 1].index

        for time_val in duplicate_times:
            dup_mask = group['time'] == time_val
            n_duplicates = sum(dup_mask)

            if n_duplicates > 1:
                time_increment = pd.Timedelta(minutes=1) / n_duplicates
                new_times = [time_val + i * time_increment for i in range(n_duplicates)]
                group.loc[dup_mask, 'time'] = new_times

        return group

    def _load_and_merge_trajectory_data(self):
        """加载并合并轨迹数据"""
        print("正在加载轨迹数据...")
        
        # 加载CSV轨迹数据
        if os.path.exists(self.traj_csv_path):
            df_traj1 = pd.read_csv(self.traj_csv_path, engine='python')
            df_traj1 = df_traj1.drop('create_time', axis=1) 
            df_traj1['time'] = pd.to_datetime(df_traj1['time'], format='%Y/%m/%d %H:%M')
            df_traj1 = df_traj1.groupby('evaluation_id', group_keys=False).apply(self._process_duplicate_times)
            df_traj1['timestamp'] = df_traj1['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_csv_path}")

        # 加载XLSX轨迹数据
        if os.path.exists(self.traj_xlsx_path):
            df_traj2 = pd.read_excel(self.traj_xlsx_path)
            df_traj2 = df_traj2.drop('create_time', axis=1) 
            df_traj2 = self._process_time_column(df_traj2)
            df_traj2['timestamp'] = df_traj2['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_xlsx_path}")

        # 合并轨迹数据
        df_traj = pd.concat([df_traj1, df_traj2], ignore_index=True)
        print(f"合并后的轨迹数据量: {len(df_traj)}条记录")

        # 预处理时间列
        df_traj['time'] = pd.to_numeric(df_traj['time'], errors='coerce')
        df_traj = df_traj.dropna(subset=['time'])  # 删除无效时间记录
        return df_traj

    def _calculate_curvature(self, group):
        """计算曲率相关特征"""
        group = group.sort_values('timestamp')
        
        x = group['ex'].values
        y = group['ey'].values
        t = group['timestamp'].values
        
        dx_dt = np.gradient(x, t)
        dy_dt = np.gradient(y, t)
        
        d2x_dt2 = np.gradient(dx_dt, t)
        d2y_dt2 = np.gradient(dy_dt, t)
        
        eps = 1e-10
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt**2 + dy_dt**2)**1.5 + eps)
        
        return pd.Series({
            'mean_curvature': float(np.nanmean(curvature)),
            'max_curvature': float(np.nanmax(curvature)),
            'curvature_std': np.nanstd(curvature),
            'high_curvature_points': (curvature > 0.5).sum()
        })

    def _calculate_trajectory_features(self, group):
        """计算轨迹特征"""
        group = group.sort_values('time')
        
        dx = group['ex'].diff()
        dy = group['ey'].diff()
        
        distance = np.sqrt(dx**2 + dy**2)
        time_diff = group['time'].diff()
        speed = distance / (time_diff.replace(0, np.nan) + 1e-6)
        
        curvature_features = self._calculate_curvature(group)
        l = len(group)
        
        if len(group) > 2:
            acceleration = np.diff(speed) / (time_diff[1:] + 1e-6)
            jerk = np.diff(acceleration) / (time_diff[2:] + 1e-6)
            directions = np.arctan2(dy, dx)
            direction_changes = np.sum(np.abs(np.diff(directions)) > np.pi/4)
        else:
            acceleration = jerk = np.array([0])
            direction_changes = 0
        
        total_distance = np.sum(distance)
        straight_distance = np.sqrt((group['ex'].iloc[-1] - group['ex'].iloc[0])**2 + 
                                   (group['ey'].iloc[-1] - group['ey'].iloc[0])**2)
        
        pause_mask = (speed < 0.1) & (distance < 1)
        pause_durations = time_diff[pause_mask]

        base_features = pd.Series({
            'mean_speed': float(np.nanmean(speed)),
            'std_speed': float(np.nanstd(speed)),
            'total_distance': float(np.nansum(distance)),
            'total_time': float(np.nansum(time_diff)),
            'pause_count': int((speed < 0.1).sum()),
            'max_speed': float(np.nanmax(speed)),
            'min_speed': float(np.nanmin(speed)),
            'speed_variation': float(np.nanstd(speed) / (np.nanmean(speed) + 1e-6)),
            'point_count': int(l),
            'complexity_ratio': total_distance / (straight_distance + 1e-6),
            'direction_changes': direction_changes,
            'mean_acceleration': np.nanmean(acceleration) if len(acceleration) > 0 else 0,
            'jerk_std': np.nanstd(jerk) if len(jerk) > 0 else 0,
            'max_pause_duration': np.max(pause_durations) if len(pause_durations) > 0 else 0,
            'pause_time_ratio': np.sum(pause_durations) / np.sum(time_diff) if np.sum(time_diff) > 0 else 0,
            # 'evaluation_id': group['evaluation_id'].iloc[0],
        })
        return pd.concat([base_features, curvature_features])

    def _extract_features(self):
        """从轨迹数据计算特征"""
        traj_features = self.df_traj.groupby('evaluation_id').apply(self._calculate_trajectory_features)
        traj_features = traj_features.reset_index()
        return traj_features

    def get_feature_matrix_and_target(self):
        """提取特征矩阵和目标变量"""
        # 先重置索引，防止evaluation_id作为索引

        if 'evaluation_id' in self.df_demo.index.names:
            self.df_demo = self.df_demo.reset_index(drop=True)  # 重设索引


        print("\n正在合并数据集...")
        traj_features = self._extract_features()

            # 重设索引，防止冲突
        if 'evaluation_id' in traj_features.index.names:
            traj_features = traj_features.reset_index(drop=True)
        df = pd.merge(self.df_demo, traj_features, left_on='evaluation_id', right_on='evaluation_id')
        print(df.head())

        # 定义目标变量
        df['cognitive_impairment'] = (df['moca_score'] <= 25).astype(int)

        # 处理point_count列
        df['point_count'] = pd.to_numeric(df['point_count'], errors='coerce').fillna(0).astype('int64')

        # 选择特征列
        features = [
            'age', 'edu_year', 'habit1',  # 人口统计'edu_year','sex'
            'mean_speed', 'std_speed', 'total_distance',  # 轨迹特征
            'total_time', 'pause_count', 'speed_variation', 'point_count',
            'mean_curvature', 'max_curvature', 'curvature_std', 'high_curvature_points',  # 曲率特征
            'complexity_ratio', 'direction_changes', 'mean_acceleration',
            'jerk_std', 'max_pause_duration', 'pause_time_ratio', 'evaluation_id',
        ]

        X = df[features]
        y = df['cognitive_impairment']
        
        return X, y
