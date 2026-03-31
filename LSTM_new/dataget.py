'''接受轨迹数据文件路径，提取轨迹特征，并返回特征矩阵和目标变量，返回的数据格式为DataFrame。'''


import pandas as pd
import numpy as np
import os
from datetime import timedelta
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

class TrajectoryFeatureExtractor:
    def __init__(self, id_path, traj_csv_path1, traj_xlsx_path, traj_csv_path2, traj_csv_path3):
        # 文件路径
        self.id_path = id_path
        self.traj_csv_path1 = traj_csv_path1
        self.traj_xlsx_path = traj_xlsx_path
        self.traj_csv_path2 = traj_csv_path2
        self.traj_csv_path3 = traj_csv_path3

        self.df_demo['diagnose'] = (self.df_demo['pt217_ab42'] > 0.0075).astype(int)
        print(self.df_demo.head(5))
        print(f"被试基本信息数据量: {len(self.df_demo)}条记录") #327条记录
        self.df_traj = self._load_and_merge_trajectory_data()
        print(f"轨迹数据量: {len(self.df_traj)}条记录")   #1462741条记录
        

    def _process_time_column(self, df):
        """处理时间列，统一两种时间格式"""
        try:
            print(df['time'])
            df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S')
        except:
            try:
                print("第一种时间格式解析失败，尝试第二种格式")
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S:%f')
            except:
                print("第二种时间格式解析失败，尝试第三种格式")
                try:
                    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
                except:
                    print("第三种时间格式解析失败")
                    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')

        
        # 改函数用于处理重复时间戳 处理到一秒内 （毫秒级）
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

        # 加载1CSV轨迹数据
        if os.path.exists(self.traj_csv_path1):
            df_traj1 = pd.read_csv(self.traj_csv_path1, engine='python')
            df_traj1 = df_traj1.drop('create_time', axis=1) 
            df_traj1 = self._process_time_column(df_traj1)
            df_traj1['time'] = pd.to_datetime(df_traj1['time'], format='%Y/%m/%d %H:%M')
            df_traj1 = df_traj1.groupby('evaluation_id', group_keys=False).apply(self._process_duplicate_times)
            df_traj1['timestamp'] = df_traj1['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_csv_path1}")
        

        # 加载2CSV轨迹数据
        if os.path.exists(self.traj_csv_path2):
            df_traj2 = pd.read_csv(self.traj_csv_path2, engine='python')
            df_traj2 = df_traj2.drop('create_time', axis=1) 
            df_traj2 = self._process_time_column(df_traj2)
            df_traj2['time'] = pd.to_datetime(df_traj2['time'], format='%Y/%m/%d %H:%M')
            df_traj2 = df_traj2.groupby('evaluation_id', group_keys=False).apply(self._process_duplicate_times)
            df_traj2['timestamp'] = df_traj2['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_csv_path2}")
        
        # 加载3CSV轨迹数据
        if os.path.exists(self.traj_csv_path3):
            df_traj3 = pd.read_csv(self.traj_csv_path3, engine='python')
            df_traj3 = df_traj3.drop('create_time', axis=1) 
            df_traj3 = self._process_time_column(df_traj3)
            df_traj3['time'] = pd.to_datetime(df_traj3['time'], format='%Y/%m/%d %H:%M')
            df_traj3 = df_traj3.groupby('evaluation_id', group_keys=False).apply(self._process_duplicate_times)
            df_traj3['timestamp'] = df_traj3['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_csv_path3}")
        
        # 加载XLSX轨迹数据
        if os.path.exists(self.traj_xlsx_path):
            df_traj4 = pd.read_excel(self.traj_xlsx_path)
            df_traj4 = df_traj4.drop('create_time', axis=1) 
            df_traj4 = self._process_time_column(df_traj4)
            df_traj4['timestamp'] = df_traj4['time'].astype('int64') / 1e9
        else:
            raise FileNotFoundError(f"未找到文件: {self.traj_xlsx_path}")
    
        
        # 合并轨迹数据
        df_traj = pd.concat([df_traj1, df_traj2, df_traj3, df_traj4], ignore_index=True)

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
        temp = self._calculate_trajectory_features
        traj_features = self.df_traj.groupby('evaluation_id').apply(temp)
        traj_features = traj_features.reset_index()
        return traj_features


    def get_feature_matrix_and_target(self):
        """提取特征矩阵和目标变量,合并id，video"""
        # 提取轨迹特征,以及计算运动学特征
        traj_features = self._extract_features()   #（790.20）

        # 将每个 evaluation_id 的 ex, ey 按时间顺序聚合为列表，便于保存原始轨迹点
        traj_coords = self.df_traj.groupby('evaluation_id', sort=False).agg({
            'ex': lambda s: s.tolist(),
            'ey': lambda s: s.tolist()
        }).reset_index()

        # 重设索引，防止冲突
        if 'evaluation_id' in traj_features.index.names:
            traj_features = traj_features.reset_index(drop=True)
        df_temp = pd.merge(self.df_demo, traj_features, left_on='cmbctid', right_on='evaluation_id')  #shape:697.80
        # 将原始轨迹点序列合并到临时表，列名为 ex_seq, ey_seq（每行是列表）
        df_temp = pd.merge(df_temp, traj_coords, on='evaluation_id', how='left')
        # print(df.head())

        # 定义目标变量
        #df['cognitive_impairment'] = (df['moca_score'] <= 25).astype(int)

        # 处理point_count列
        df_temp['point_count'] = pd.to_numeric(df_temp['point_count'], errors='coerce').fillna(0).astype('int64')
        
        df_temp.to_parquet('data_new.parquet')
        print("已保存到 data_new.parquet 文件中。")

        
if __name__ == "__main__":
    id_path = r"D:/Code/DeepTTE/data/轨迹/轨迹/demoid.xlsx"    # 身份id"video",age,sex,,edu_year,habit,label
    
    traj_csv_path1 = r"D:/Code/DeepTTE/data/轨迹/轨迹/连线测试轨迹0730(1).csv"    #轨迹1
    traj_xlsx_path = r"D:/Code/DeepTTE/data/轨迹/轨迹/连线测试轨迹0730(2).xlsx"   #轨迹2
    traj_csv_path2 = r"D:/Code/DeepTTE/data/轨迹/轨迹/连线测试轨迹0919.csv"       #轨迹3
    traj_csv_path3 = r"D:/Code/DeepTTE/data/轨迹/轨迹/连线测试轨迹体检.csv"        #轨迹4
   
    # 获取属性特征，edu_year, sex, age ,mean_speed etlc.    我的目标：将所有的特征整合为一个打的dataframe变量。（储存到csv中？）
    extractor = TrajectoryFeatureExtractor(id_path, traj_csv_path1, traj_xlsx_path, traj_csv_path2, traj_csv_path3)
    extractor.get_feature_matrix_and_target() #这里返回的是DataFrame格式