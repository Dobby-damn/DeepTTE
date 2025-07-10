import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GeoConv
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''
    def __init__(self, attr_size, kernel_size = 3, num_filter = 32, pooling_method = 'attention', rnn = 'lstm'):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        self.geo_conv = GeoConv.Net(kernel_size = kernel_size, num_filter = num_filter)
	    #num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size = num_filter + 1 + attr_size, \
                                      hidden_size = 128, \
                                      num_layers = 2, \
                                      batch_first = True
            )
        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size = num_filter + 1 + attr_size, \
                              hidden_size = 128, \
                              num_layers = 1, \
                              batch_first = True
            )


        if pooling_method == 'attention':
            self.attr2atten = nn.Linear(attr_size, 128)

    def out_size(self):
        # return the output size of spatio-temporal component
        return 128
    
    # 最小池化
    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim = 1, keepdim = False)

        if torch.cuda.is_available():
            lens = torch.cuda.FloatTensor(lens)
        else:
            lens = torch.FloatTensor(lens)

        lens = Variable(torch.unsqueeze(lens, dim = 1), requires_grad = False)

        hiddens = hiddens / lens

        return hiddens


    def attent_pooling(self, hiddens, lens, attr_t):
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

	#hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(-alpha)

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens


    def forward(self, traj, attr_t, config):
        # 确保长度是CPU上的整数张量
        # 卷积处理后的轨迹特征 [batch_size, seq_len, num_filter]
        conv_locs = self.geo_conv(traj, config)
        # print("Geo输出：", conv_locs.shape)

        attr_t = torch.unsqueeze(attr_t, dim = 1)  # 增加一维 [batch_size, attr_size]变成[batch_size, 1, attr_size]
        # print("属性张良形状", attr_t.shape)
        # 静态属性特征扩展 [batch_size, 1, attr_size] -> [batch_size, seq_len, attr_size]
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1], ))
        # print("扩展属性张量形状", expand_attr_t.shape)
        # concat the loc_conv and the attributes 张量拼接
        # 拼接操作 [batch_size, seq_len, num_filter + attr_size]
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim = 2)
        # print("拼接后的卷积张量形状", conv_locs.shape)
        # return 
        lens = traj['lens'].to(torch.int64).cpu()  # 确保类型和位置正确
        # 打印调试信息 验证长度一致性
        # print(f"输入形状: {conv_locs.shape}, 长度总和: {sum(lens)}")
        # print(conv_locs.size(1),max(lens))
        # assert conv_locs.size(1) >= max(lens), "序列长度超过输入维度"
        lens = [x - self.kernel_size + 1 for x in traj['lens']]  # 卷积会缩短序列
        lens = torch.tensor(lens, dtype=torch.int64).cpu()
        # print(f"调整后的长度: {lens}")  # 调试输出
        
        # lens = map(lambda x: x - self.kernel_size + 1, traj['lens'])

        # packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first = True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            conv_locs, 
            lens, 
            batch_first=True,
            enforce_sorted = False
        )
        # Rnn处理
        try:
            packed_hiddens, _ = self.rnn(packed_inputs)
            print("RNN输出大小:", packed_hiddens.data.shape)  # 调试
        except Exception as e:
            print("RNN处理失败:", e)
            raise
        # 解包序列前检查
        if packed_hiddens.data.numel() == 0:
            raise ValueError("RNN输出为空！检查输入数据和模型参数")
        # 解包序列 - 添加调试信息
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(
            packed_hiddens, 
            batch_first = True,
            total_length = max(lens)  # 确保解包后长度一致
        )
        # print("Unpacked shapes:")
        # print("解包后形状:", hiddens.shape)
        # print("Output lengths:", lens)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

        elif self.pooling_method == 'attention':
            return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)
