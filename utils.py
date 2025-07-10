import torch
from torch.autograd import Variable

import json

from math import radians, cos, sin, asin, sqrt

config = json.load(open('./config.json', 'r'))

def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def normalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return (x - mean) / std

def unnormalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return x * std + mean

def pad_sequence(sequences, lengths):
    padded = torch.zeros(len(sequences), lengths[0]).float()
    for i, seq in enumerate(sequences):
        seq = torch.Tensor(seq)
        padded[i, :lengths[i]] = seq[:]
    return padded

def to_var(var):
    """将输入数据转换为PyTorch Variable（自动处理CUDA转移和嵌套数据结构）
    
    Args:
        var: 输入数据，可以是以下类型之一：
            - torch.Tensor
            - int/float 标量
            - dict 字典
            - list 列表
    
    Returns:
        转换后的Variable或原始数据（保持结构不变）
        
    功能说明：
        1. 对于Tensor：转换为Variable并自动移至GPU（如果可用）
        2. 对于字典/列表：递归处理所有元素
        3. 对于标量：原样返回
    """
    
    # 情况1：处理PyTorch Tensor
    if torch.is_tensor(var):
        var = Variable(var)  # 包装为Variable（旧版PyTorch需要，新版可直接用Tensor）
        
        # 如果CUDA可用，将数据移至GPU
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    
    # 情况2：处理Python标量（直接返回）
    if isinstance(var, int) or isinstance(var, float):
        return var
    
    # 情况3：处理字典（递归处理每个值）
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])  # 递归调用处理每个值
        return var
    
    # 情况4：处理列表（递归处理每个元素）
    if isinstance(var, list):
        # 使用map函数处理列表元素（Python2返回list，Python3返回map对象）
        var = list(map(lambda x: to_var(x), var))  # 显式转换为list保证Python3兼容性
        return var
    
    # 注：如果输入是其他类型（如None），将直接返回
    return var

def get_local_seq(full_seq, kernel_size, mean, std):
    seq_len = full_seq.size()[1]

    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor(seq_len)
    else:
        indices = torch.LongTensor(seq_len)

    torch.arange(0, seq_len, out = indices)

    indices = Variable(indices, requires_grad = False)

    first_seq = torch.index_select(full_seq, dim = 1, index = indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim = 1, index = indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq

