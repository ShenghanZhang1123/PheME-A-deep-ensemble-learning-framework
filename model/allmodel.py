# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:56:52 2022

@author: dell
"""
# classes
import torch.nn as nn
import torch
from einops import rearrange, repeat

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads , dim_head , dropout ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class lstm(nn.Module):
    def __init__(self, num_class, dim):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=128,
            batch_first=True)  # batch_first 是因为DataLoader所读取的数据与lstm所需的输入input格式是不同的，
        # 所在的位置不同，故通过batch_first进行修改
        self.fc = nn.Linear(128, num_class)  # 连接层的输入维数是hidden_size的大小
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        out, (h_0, c_0) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sf(out)
        return out

class EnCNN(nn.Module):
    def __init__(self,dim, max_length=512):
        super(EnCNN, self).__init__()
        out_channel = 100 #可以等价为通道的解释。
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),#卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
    def forward(self, x):
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        return out

class BertLinear(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(BertLinear, self).__init__()
        self.bertlinear = nn.Linear(dim * max_length, 1024)
        self.fc = nn.Linear(1024, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.bertlinear(x)
        logits = self.fc(out)  # 结果输出[128, 2]
        return self.sf(logits)

class TabulatedLinear(nn.Module):
    def __init__(self, num_class, dim):
        super(TabulatedLinear, self).__init__()
        self.bertlinear0 = nn.Linear(dim, 512)
        self.bertlinear1 = nn.Linear(512, 256)
        self.bertlinear2 = nn.Linear(256, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.bertlinear0(x)
        x = self.bertlinear1(x)
        logits = self.bertlinear2(x)
#        logits = self.fc(out)  # 结果输出[128, 2]
        return self.sf(logits)

class BertCNN(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(BertCNN, self).__init__()
        out_channel = 100 #可以等价为通道的解释。
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),#卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.fc = nn.Linear(out_channel * len(filter_sizes), num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        logits = self.fc(out)  # 结果输出[128, 2]
        return self.sf(logits)

class TextLinear_Merge(nn.Module):
    def __init__(self, num_class, dim, struc_size, max_length=512):
        super(TextLinear_Merge, self).__init__()
        self.bertlinear = nn.Linear(dim * max_length, 1024)
        self.linear_struct = nn.Linear(struc_size, 1024)
        # self.fc = nn.Linear(out_channel * len(filter_sizes), num_class) # batchsize*data
        self.fc0 = nn.Linear(2048, 512)
        self.fc = nn.Linear(512, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x, x_struc):

        out = self.bertlinear(x)
#        x_struc = x_struc.unsqueeze(1)
#        out_struc = [conv(x_struc) for conv in self.convs_struc]
#        out_struc = torch.cat(out_struc, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
#        out_struc = out_struc.view(x_struc.size(0), -1)  # 展平

        out_struc = self.linear_struct(x_struc)
        out_merge = torch.cat((out, out_struc), dim=1)  # 拼接structured data和UNstructured
        out_merge = self.fc0(out_merge)
        logits = self.fc(out_merge)  # 结果输出[128, 2]
        return self.sf(logits)

class TextCNN_Merge(nn.Module):
    def __init__(self, num_class, dim, struc_size, max_length=512):
        super(TextCNN_Merge, self).__init__()
        out_channel = 100 #可以等价为通道的解释。
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),#卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.convs_struc = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, out_channel, (fs,)),  # 卷积核大小为2*Embedding_size,默认当然是步长为1
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(struc_size - fs + 1),
        ) for fs in filter_sizes])
        self.linear_struct = nn.Linear(struc_size, out_channel * len(filter_sizes))
        # self.fc = nn.Linear(out_channel * len(filter_sizes), num_class) # batchsize*data
        self.fc = nn.Linear(out_channel * len(filter_sizes) * 2, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x, x_struc):

        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平

#        x_struc = x_struc.unsqueeze(1)
#        out_struc = [conv(x_struc) for conv in self.convs_struc]
#        out_struc = torch.cat(out_struc, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
#        out_struc = out_struc.view(x_struc.size(0), -1)  # 展平

        out_struc = self.linear_struct(x_struc)
        out_merge = torch.cat((out, out_struc), dim=1)  # 拼接structured data和UNstructured

        logits = self.fc(out_merge)  # 结果输出[128, 2]
        return self.sf(logits)

class TextCNN(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(TextCNN, self).__init__()
        out_channel = 100 #可以等价为通道的解释。
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),#卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.fc = nn.Linear(out_channel * len(filter_sizes), num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        logits = self.fc(out)  # 结果输出[128, 2]
        return self.sf(logits)
    
    
    
    
    