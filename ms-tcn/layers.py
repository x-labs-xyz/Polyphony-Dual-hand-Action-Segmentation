import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.special import softmax

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import copy

def exchange_time(x, exchange_rate=0.15):
    #x是输入的一个batch，维度（batch_size,feature_dim,feature_num(最长一段视频中帧数)）
    #exchange_label的维度是（batch_size,feature_num）
    exchange_label = torch.zeros(x.shape[0], x.shape[2], dtype=torch.long).to(x.device)
    #视频的长度，整除2再乘2是为了获得偶数
    seq_length = (x.shape[2]//2)*2
    #获得交换的帧的数量
    exchange_num = int(seq_length/2*exchange_rate)
    #随机生成交换对，torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
    #n是采样的上限，out是输出张量，layout内存布局stried就是顺序存储
    #获得随机打乱的样本对，这里随机生成一个长度为视频长度的序列，然后取序列中前seq_length长度，然后将尺寸改为（2，seq_length/2）
    randn_pair = torch.randperm(x.shape[2])[:seq_length].reshape(2,-1)

    ##############修改部分############
    exchange_pair = randn_pair[:, :1]
    for i in range(exchange_num):
        if i != 0:
            exchange_pair = torch.cat((exchange_pair, randn_pair[:, i:i + 1]), 1)
        if randn_pair[0, i] + 1 in exchange_pair or randn_pair[1, i] + 1 in exchange_pair:
            continue
        elif randn_pair[0, i] + 1 in randn_pair and randn_pair[1, i] + 1 in randn_pair:
            exchange_pair = torch.cat((exchange_pair, torch.tensor([[randn_pair[0, i] + 1], [randn_pair[1, i] + 1]])), 1)
        else:
            continue
    ################################

    #将样本对中取出来exchange_num个，每列的两个数就对应一个要交换的样本对
    #exchange_pair = randn_pair[:,:exchange_num]

    #arange生成的序列不包含end，range包含。
    exchange_index = torch.arange(start=0, end=x.shape[2])
    #获得交换后的序列
    exchange_index[exchange_pair[0]] = exchange_pair[1]
    exchange_index[exchange_pair[1]] = exchange_pair[0]

    #获得交换后的输入batch，x
    exchange_x = x[:,:,exchange_index]
    #标注交换标签，label的shape是（batch_size,feature_num）也就是（视频个数，视频长度），因此就是标注出每段视频中发生交换的帧就标为1
    exchange_label[:,exchange_pair[0]]=1
    exchange_label[:,exchange_pair[1]]=1

    return exchange_x, exchange_label

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        #每一层是一个膨胀残差层
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.exchange_out = nn.Conv1d(num_f_maps, 2, 1)

    def forward(self, x, mask, ex_x, ex_label):
        #先对输入特征进行一维卷积改变维度到num_f_maps的维度
        out = self.conv_1x1(x)
        #输入膨胀残差层
        for layer in self.layers:
            out = layer(out, mask)
        #获得输出预测结果
        pred = self.conv_out(out) * mask[:, 0:1, :]

        #对交换后的x进行预测获得ex_pred预测结果
        ex_out = self.conv_1x1(ex_x)
        for layer in self.layers:
            ex_out = layer(ex_out, mask)
        #ex_pred判断的是存不存在交换，因此输出维度是（2）
        ex_pred = self.exchange_out(ex_out) * mask[:, 0:1, :]
        #ex_clspred是交换后样本输出的预测结果
        ex_clspred = self.conv_out(ex_out) * mask[:, 0:1, :]

        return pred, ex_pred, ex_label, ex_clspred


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class DRCGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True, kernel_size=3, dilation=1, padding=1):
        super(DRCGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        #torch.FloatTensor是转换类型，将list或numpy转换成tensor.Parameter函数是将生成的参数纳入到模型的参数中，参与训练
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #计算余弦相似度
    def cosine_pairwise(self, x):
        #x维度是（batch_size，3，视频长度,特征维度），改为（3,特征维度，batch_size，视频长度）
        #输入维度（1,3,15,64）
        x = x.permute((1, 3, 0, 2))
        #x维度改为（3,64，1,15）
        #print('x.shape',x.shape)
        #print('x',x)
        #print('x.unsqueeze(1).shape',x.unsqueeze(1).shape)
        #(3,1,64,1,15)
        #print('x.unsqueeze(1)',x.unsqueeze(1))

        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-3)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 3, 0, 1))
        batch_size, seq_length, kernel_size, _ = cos_sim_pairwise.shape
        adj = cos_sim_pairwise.reshape(batch_size, seq_length, -1)
        adj = F.softmax(adj, dim=1).reshape(batch_size, seq_length, kernel_size, kernel_size)
        #这里其实是针对每一帧，构建一个3邻域的子图，每个子图计算一个邻接矩阵，所以adj的维度是（batch_size,seq_length,3,3）
        return adj

    #输入是TCN输出预测的sofmax结果
    def forward(self, x, adj=None):
        #batch大小，特征维度，视频长度
        #print('x.shape', x.shape)
        #[1,64,15]
        #print('x_before',x)
        batch_size, feat_dim, L = x.shape
        #给x在视频长度后再开一个维度
        x = x.unsqueeze(3)
        #print('x.shape ', x.shape)
        #[1,64,15,1]
        #print('x_after',x)
        #torch.nn.functional.unfold（input, kernel_size, dilation=1, padding=0, stride=1）从批量的输入张量中(batch_size,channel,H,W)提取滑动局部块(kernel_size的长宽）。
        #input: tensor数据，四维， Batchsize, channel, height, width
        #kernel_size: 核大小，决定输出tensor的数目
        #dilation: 输出形式是否有间隔
        #padding：一般是没有用的必要
        #stride： 核的滑动步长
        #最后input的维度是（batch_size,3*特征维度*1，视频长度）其实就是把没三帧的特征合并一起
        #print('x', x[0, :, 0:3, 0])
        #print('x[0, :, 1:4, 0].shape',x[0, :, 1:4, 0].shape)
        input = F.unfold(x, kernel_size=(self.kernel_size,1), dilation=(self.dilation,1), padding=(self.padding,0))
        #print('input.shape=',input.shape)
        #[1,192,15],且192维度的特征是x中三个特征一行一行拼起来的，也就是192维度的前三个数分别来自x相邻三个特征的第一个数
        #print('input', input[0, :, 2])
        #print('input[0, :, 2].shape',input[0, :, 2].shape)
        #把input的维度改为（batch_size,特征维度，3，视频长度），再改为（batch_size，3，视频长度,特征维度）
        input = input.reshape(batch_size, feat_dim, self.kernel_size, L).permute(0,2,3,1)
        #print('input_before_permute.shape',input.reshape(batch_size, feat_dim, self.kernel_size, L).shape)
        # [1, 64, 3, 15]
        #print('input_before_permute',input.reshape(batch_size, feat_dim, self.kernel_size, L)[0, :, :, 2])
        #print('input.shape',input.shape)
        #[1, 3, 15, 64]
        # print('input', input[0, :, 2, :])

        if adj is None:
            adj = self.cosine_pairwise(input)
        #print('self.weight', self.weight)
        #self.weight是生成的初始权重，（64,64）
        #input维度是（batch_size，3，视频长度,特征维度）,也就是（1,3,15，64）
        support = torch.matmul(input, self.weight)
        #print('support.shape',support.shape)
        #[1,3,15,64]
        support = support.permute(0,2,1,3)
        #adj维度为[1,15,3,3], support维度改为[1,15,3,64]
        output = torch.matmul(adj, support)
        #output的维度是[1,15,3,64]

        if self.bias is not None:
            output = output + self.bias
        output = output.permute(0,3,2,1).reshape((batch_size, feat_dim*self.kernel_size, L))
        #print('output[0,0:3,:]=',output[0,0:3,:])
        #对于output先调整维度为（1,64,3,15），在改变形状为（1,64*3,15）
        #F.fold函数是将一组滑动局部块张量组合成一个包含这些张量的大张量，参数为（input,output_size,kernel_size,dilation,padding,stride）
        output = F.fold(output, (1,L), (self.kernel_size,1), dilation=(self.dilation,1), padding=(self.padding,0))
        #print('output.shape', output.shape)
        #[1,64,1,15]
        #这里比较奇怪的是fold的时候，每三个特征不是取和或者平均，而是直接取每三个特征中间一个作为输出特征
        #print('output[0,:,0,:]=',output[0,:,0,:])
        output = output.squeeze(2)
        #print('output.shape',output.shape)
        #[1,64,15]
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResidualLayer(nn.Module):
    def __init__(self, dilation, df_size, in_channels, out_channels):
        super(GCNResidualLayer, self).__init__()
        padding = int((dilation*(df_size-1))/2)
        self.df_size = df_size
        self.gcn_dilated1 = DRCGraphConvolution(in_channels, out_channels, kernel_size=df_size, dilation=dilation, padding=padding)
        self.gcn_dilated2 = DRCGraphConvolution(in_channels, out_channels, kernel_size=df_size, dilation=dilation, padding=padding)
        #膨胀卷积（输入维度，输出维度，核大小是3），得到邻接矩阵
        self.conv_dilated_adj = nn.Conv1d(out_channels, df_size*df_size, df_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    #这里输入的x是TCN输出的softmax值
    def forward(self, x, mask):
        #print('x.shape', x.shape)
        #[1, 64, 15]
        batch_size, _, seq_length = x.shape
        #输出的每个特征的维度是3*3=9，整体输出是（batch_size,9,seq_length）
        adj = self.conv_dilated_adj(x)
        #print('adj.shape',adj.shape)
        #[1,9,15]
        #permute是用来改变维度，(0,2,1)就是把第二第三维度交换,也就变成（batch_size,seq_length,9）
        adj = F.softmax(adj, dim=1).permute(0,2,1)
        #permute后，adj的shape是[1,15,9]
        #reshape后维度变为（batch_size,seq_length,3,3），于是就得到了邻接矩阵，这里是（1,15,3,3）
        adj = adj.reshape(batch_size, seq_length, self.df_size, self.df_size)
        #第一个gcn是相似度图S-Graph，第二个是L-Graph
        out = F.relu(self.gcn_dilated1(x)) + F.relu(self.gcn_dilated2(x, adj))
        #print('out.shape',out.shape)
        #[1,64,15]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #return时候进行一个残差运算
        #print('mask[:,0:1,:]',mask[:,0:1,:])

        return (x + out) * mask[:, 0:1, :]


class GCNStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, df_size, dim, num_classes):
        super(GCNStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GCNResidualLayer(2 ** i, df_size, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.exchange_out = nn.Conv1d(num_f_maps, 2, 1)

    def forward(self, x, mask, ex_x, ex_label):
        #对输入进行一次1维卷积，输出维度为num_f_maps
        #print('x.shape',x.shape)
        #[1, 48, 15]
        out = self.conv_1x1(x)
        #print('out.shape',out.shape)
        #[1, 64, 15]
        #多个gcn残差层
        for layer in self.gcn_layers:
            out = layer(out, mask)
        pred = self.conv_out(out) * mask[:, 0:1, :]

        ex_out = self.conv_1x1(ex_x)
        for layer in self.gcn_layers:
            ex_out = layer(ex_out, mask)
        ex_pred = self.exchange_out(ex_out) * mask[:, 0:1, :]
        #print('ex_pred.shape', ex_pred.shape)
        #[1,2,15]
        #print('ex_pred', ex_pred)

        ex_clspred = self.conv_out(ex_out) * mask[:, 0:1, :]

        return pred, ex_pred, ex_label, ex_clspred