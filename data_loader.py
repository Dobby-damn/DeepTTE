import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
        # 读取数据文件（每行是一个JSON格式的轨迹记录）
        # self.content = open('./data/' + input_file, 'r').readlines()
        with open('./data/' + input_file, 'r') as f:
            self.content = f.readlines()
        # 解析JSON数据
        self.content = list(map(lambda x: json.loads(x), self.content))
        # 计算每条轨迹的长度（坐标点数量）
        self.lengths = list(map(lambda x: len(x['lngs']), self.content))

    def __getitem__(self, idx):
        """获取单条轨迹数据"""
        return self.content[idx]

    def __len__(self):
        """返回数据集总大小"""
        return len(self.content)

def collate_fn(data):
    """自定义批处理函数，用于填充变长序列和归一化数据
    Args:
        data: 一个batch的原始数据列表（每个元素是MySet返回的一条轨迹）
    Returns:
        attr: 静态属性字典（标准化后的数值特征和类别ID）
        traj: 轨迹数据字典（填充后的序列和原始长度）
    """
    # 需要统计归一化的静态属性
    stat_attrs = ['dist', 'time']   # 总距离和总时间
    # 类别型ID属性
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    # 轨迹序列属性
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}
    # 获取当前batch中各轨迹的实际长度
    lens = np.asarray([len(item['lngs']) for item in data])
    # 处理静态数值属性（归一化）
    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)   # 假设utils.normalize实现均值方差归一化
    
    # 处理类别型ID属性
    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    # 处理轨迹序列数据（填充变长序列）
    for key in traj_attrs:
        # pad to the max length 创建填充矩阵（batch_size x max_len）
        # seqs = np.asarray([item[key] for item in data])
        # mask = np.arange(lens.max()) < lens[:, None]   # 生成掩码矩阵（标记有效数据位置）
        # padded = np.zeros(mask.shape, dtype = np.float32)   # 填充实际数据
        # padded[mask] = np.concatenate(seqs)   
        # python2/3兼容性问题，直接构建填充后的张量
        seqs = [item[key] for item in data]
        max_len = max(len(seq) for seq in seqs)
        padded = torch.zeros(len(data), max_len, dtype=torch.float32)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.FloatTensor(seq)
        # 对数值型序列进行归一化
        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        # padded = torch.from_numpy(padded).float()
        traj[key] = padded
    # 保存原始长度信息（用于后续处理）
    lens = [len(item['lngs']) for item in data]
    assert all(l > 0 for l in lens), "存在零长度序列"
    # lens = lens.tolist()
    traj['lens'] = torch.tensor(lens, dtype=torch.int64)  # 确保是int64张量

    return attr, traj

class BatchSampler:
    """自定义批次采样器，优化变长序列的批处理效率"""
    def __init__(self, dataset, batch_size):
        """生成批次索引，按长度排序减少填充量"""
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = range(self.count)

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        indices = np.array(self.indices)  # 创建副本
        np.random.shuffle(indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = indices[i * chunk_size: (i + 1) * chunk_size]
            # partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            partial_indices = sorted(partial_indices, 
                        key=lambda x: self.lengths[x], 
                        reverse=True)
            indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    """创建数据加载器
    Args:
        input_file: 输入文件名（位于./data/目录下）
        batch_size: 批次大小
    Returns:
        配置好的DataLoader实例
    """
    dataset = MySet(input_file=input_file)
    batch_sampler = BatchSampler(dataset, batch_size)

    # 关键配置说明：
    # - batch_size=1 因为批处理已在collate_fn中实现
    # - num_workers=4 使用4个子进程加载数据
    # - pin_memory=True 加速GPU数据传输
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # 实际批次大小由batch_sampler控制
        collate_fn=collate_fn,  # 使用自定义批处理函数
        num_workers=4,
        batch_sampler=batch_sampler,
        pin_memory=True
    )

    return data_loader
if __name__ == "__main__":
    # 创建一个小的测试批次
    test_data = [
        # 样本1 (长度3)
        {
            'dist': 10.5,                    # 总距离(km)
            'time': 3600,                     # 总时间(秒)
            'driverID': 1,                    # 司机ID
            'dateID': 101,                    # 日期ID
            'weekID': 3,                      # 周ID
            'timeID': 1,                      # 时间段ID
            'lngs': [116.404, 116.405, 116.406],  # 经度序列
            'lats': [39.915, 39.916, 39.917],      # 纬度序列
            'states': [0, 1, 0],              # 状态序列(0:行驶,1:停留)
            'time_gap': [60, 120],            # 时间间隔(秒)
            'dist_gap': [0.5, 0.5]            # 距离间隔(km)
        },
        # 样本2 (长度2)
        {
            'dist': 8.2,
            'time': 2400,
            'driverID': 2,
            'dateID': 102,
            'weekID': 3,
            'timeID': 2,
            'lngs': [116.407, 116.408],
            'lats': [39.918, 39.919],
            'states': [1, 0],
            'time_gap': [180],
            'dist_gap': [0.3]
        }
    ]
    attr, traj = collate_fn(test_data)
    print(traj['lens'])  # 应该输出类似tensor([3, 2])