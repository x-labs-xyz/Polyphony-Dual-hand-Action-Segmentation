#!/usr/bin/python2.7

import torch
import numpy as np
import random

##生成训练用的batch
class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes     #动作的类别数量
        self.actions_dict = actions_dict   #动作字典
        self.gt_path = gt_path             #真实标签地址 gt = ground truth
        self.features_path = features_path #特征路径，是I3D特征
        self.sample_rate = sample_rate     #采样率，出了50Salad是2，其它都是1，保证时间分辨率为15fps

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):  #判断当前batch后面还有没有batch
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):         #读取数据集，可以是训练集或测试集
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]   #读取一条条数据，vid_list_file里面其实就是一段段视频的名字，如P03_cam01_P03_cereals.txt
        file_ptr.close()
        random.shuffle(self.list_of_examples)               #对读取的数据进行打乱

    def next_batch(self, batch_size):  #生成一个batch
        batch = self.list_of_examples[self.index:self.index + batch_size] #取数据中从当前index开始取一个batch
        self.index += batch_size  #当前数据的index就要加上一个batch大小

        batch_input = []
        batch_target = []
        for vid in batch:   #batch里面是一个batch大小的数据的名字如P03_cam01_P03_cereals.txt
            features = np.load(self.features_path + vid.split('.')[0] + '.npy') #load(features_path/P03_cam01_P03_cereals.npy)也就是读取对应数据名的特征数据
            file_ptr = open(self.gt_path + vid, 'r')       #读取对应数据名的ground truth数据
            content = file_ptr.read().split('\n')[:-1]     #对gt数据进行拆分，赋值给content，gt中保持的是动作名字'SIL','pour_water'
            classes = np.zeros(min(np.shape(features)[1], len(content))) #正常来说feature数量和content长度是一致的也就是帧数，features的shape是（2048,帧数），所以一列数据是一帧图的特征
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]       #从动作字典中获取每一帧图像的动作所对应的动作编码，如获取SIL的动作编码为0
            batch_input .append(features[:, ::self.sample_rate]) #按照采样率为间隔将特征放入batch_input
            batch_target.append(classes[::self.sample_rate])     #batch_target是按照采样率为间隔取从第一帧到最后一帧的动作类别

        #length_of_sequences = map(len, batch_target)
        # map函数在python2中返回一个list，在python3里面返回一个迭代器对象,map(function,iterable),计算interable序列里每个数的函数值
        #这里计算batch_target中每个样本的视频长度
        length_of_sequences = list(map(len, batch_target))
        #batch_input_tensor的数据维度为（batch大小，特征长度也就是2048，batch中最长视频长度）
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        #batch_target_tensor的维度为（batch大小，batch中最长视频长度）
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        #创建一个掩码，维度为（batch大小，动作类别数量，batch中最长视频长度）
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            # 将特征转变为tensor数据类型，batch_input的shape为（2048,帧数）
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            #mask全部赋值为1，也就是视频长度小于最长视频长度的部分为0
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
