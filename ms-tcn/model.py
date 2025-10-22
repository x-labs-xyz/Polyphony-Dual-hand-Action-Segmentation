#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    #num_f_maps是filter个数64，dim是特征维度2048
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        #stage1就是单独一层TCN模型
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        #ModuleList是把多个模型放在一个列表里。deepcopy就是深复制，就是把对象完全复制一遍成一个单独个体。复制num_stages-1个TCN模型。
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        print('num_classes = ',num_classes)

    def forward(self, x, mask):
        #先计算一层TCN输出
        out = self.stage1(x, mask)
        #print('out size =',out.size())

        #给输出维度增加一个维度，用来存储每层的输出结果
        outputs = out.unsqueeze(0)
        #每个s就是stages那个模型列表中的一个模型，也就是一个TCN层
        for s in self.stages:
            #取上一层的out作为当前层的输入
            out_before_softmax = out

            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            #print(type(out))
            #print('out softmax = ', out)
            #将当前的输出存如outputs，dim=0是按维度0拼一起，就是竖着拼，也就拼在下面
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        #输出outputs是每层output的集合
        #print('outputs size',outputs.size())
        return outputs, out_before_softmax


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        #1*1卷积的参数是2048*64,也就是将特征维度从2048降到64.
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        #每层都是一个膨胀残差连接层，参数是dilation=2**i，in_channels = 64, out_channels=64
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        #先对输入进行1*1卷积，改变输入维度。
        out = self.conv_1x1(x)
        for layer in self.layers:
            #将降维后的输入放入膨胀残差层计算
            out = layer(out, mask)
        #经过多层膨胀残差层之后进入输出层，将输出映射到num_classes的维度上。
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        #定义膨胀层卷积，输出维度不变还是64,核的大小是3，padding可以让头尾的特征也参与进来，不会损失特征，
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        #先计算输入的膨胀卷积，然后再进行一次映射。
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #这里x+out就是残差连接
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        #model是多层堆叠的模型，(堆叠的模型数量，每层模型里的网络层数，filter的数量也就是输出维度，特征维度，动作类别数量)
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)  #分类损失函数
        self.mse = nn.MSELoss(reduction='none')           #平滑损失函数 MSE是mean squared error
        self.num_classes = num_classes                    #动作类别数量

    #定义训练函数，（模型保持地址，batch生成器，epoch数量，batch大小，学习率，运算设备）
    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)

        training_result=open(save_dir+'/training_reault.txt','w')

        #torch.optim.Adam(params, lr=0.001) params(iterable) – 待优化参数的iterable或者是定义了参数组的dict, lr是学习率
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next(): #如果当前batch还没到最后，就接着计算，是判断当前self.index是否小于数据总量，初始index=0
                #next_batch返回的是batch的特征tensor，batch的ground truth标签，掩码，掩码的shape是（batch大小，动作类别数量，batch中最长视频长度）
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()  #每个batch初始化梯度为0，以防止梯度累加
                predictions,_ = self.model(batch_input, mask) #计算forward

                loss = 0
                for p in predictions: #计算loss，累加起来每层模型的loss
                    #torch.tanspose(input,dim0,dim1)是将input的dim0和dim1交换, contiguous是使数据连续，view前要保证数据连续
                    #ce是CrossEntropyLoss(out，label)，这里out是一个样本中每帧图像类别的softmax得分，所以维度是(batch_targe_size,num_classes)
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    #torch.clamp(imput,min,max,out=None)将输入夹紧在min到max的区间，也就是小于min的变成min
                    #p.detach就是把p分离出来成一个新的tensor，这个p不再随着之前的p改变。
                    #[:, :, 1:]-[:, :, :-1]就是t时刻的输出-t-1时刻的输出
                    #这里的max其实就是文中公式中的阈值
                    #0.15是文中的λ
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item() #用item取出来的数据精度更高。
                loss.backward()           #反向传播计算梯度
                optimizer.step()          #根据梯度更新参数

                _, predicted = torch.max(predictions[-1].data, 1)  #获取预测结果
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item() #计算正确的数量
                total += torch.sum(mask[:, 0, :]).item()     #总的视频帧数

            batch_gen.reset() #当一个epoch计算完了之后，初始化batch的生成器，让当前数据的index为0，并重新打乱一遍数据
            #state_dict是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            #优化器对象Optimizer也有一个state_dict,它包含了优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            
            training_result.write("[epoch %d]: epoch loss = %f,   acc = %f \n" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        #测试时候就不用计算梯度了
        with torch.no_grad():
            self.model.to(device)
            #加载最后一个epoch训练出来的模型
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            #加载测试集
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                #加载特征
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                #采样特征
                features = features[:, ::sample_rate]
                #将特征数据类型转为tensor，作为输入x
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                #前面unsqueeze步骤是为了让输入维度与训练时候一样，这里相当于一个batch就是一段视频。torch.ones()创建的就是mask
                predictions, out_before_softmax = self.model(input_x, torch.ones(input_x.size(), device=device))
                
               # print('predictions size = ', predictions.size())
                _, predicted = torch.max(predictions[-1].data, 1)

                #print('predicted size = ', predicted.size())
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    #values是编号，keys是动作类型，所以最后recognition存储的动作类型
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                print('f_name = ', f_name)
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
                out_before_softmax_path = results_dir+'/' + f_name +'.pt'
                #torch.save(out_before_softmax, out_before_softmax_path)
