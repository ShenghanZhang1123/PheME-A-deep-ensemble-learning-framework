# -*- coding: utf-8 -*-
"""
@author: shenghan
"""
# classes
import torch.nn as nn
import torch
from einops import rearrange, repeat


class lstm(nn.Module):
    def __init__(self, num_class, dim):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=128,
            batch_first=True) 
        self.fc = nn.Linear(128, num_class)  
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        out, (h_0, c_0) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sf(out)
        return out

class BertLinear(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(BertLinear, self).__init__()
        self.bertlinear = nn.Linear(dim * max_length, 1024)
        self.fc = nn.Linear(1024, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.bertlinear(x)
        logits = self.fc(out)
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
#        logits = self.fc(out) 
        return self.sf(logits)

class BertCNN(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(BertCNN, self).__init__()
        out_channel = 100 
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.fc = nn.Linear(out_channel * len(filter_sizes), num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        logits = self.fc(out)
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
#        out_struc = torch.cat(out_struc, dim=1)
#        out_struc = out_struc.view(x_struc.size(0), -1)

        out_struc = self.linear_struct(x_struc)
        out_merge = torch.cat((out, out_struc), dim=1)
        out_merge = self.fc0(out_merge)
        logits = self.fc(out_merge)
        return self.sf(logits)

class TextCNN_Merge(nn.Module):
    def __init__(self, num_class, dim, struc_size, max_length=512):
        super(TextCNN_Merge, self).__init__()
        out_channel = 100 
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.convs_struc = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, out_channel, (fs,)),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(struc_size - fs + 1),
        ) for fs in filter_sizes])
        self.linear_struct = nn.Linear(struc_size, out_channel * len(filter_sizes))
        # self.fc = nn.Linear(out_channel * len(filter_sizes), num_class)
        self.fc = nn.Linear(out_channel * len(filter_sizes) * 2, num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x, x_struc):

        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1) 
        out = out.view(x.size(0), -1) 

#        x_struc = x_struc.unsqueeze(1)
#        out_struc = [conv(x_struc) for conv in self.convs_struc]
#        out_struc = torch.cat(out_struc, dim=1)
#        out_struc = out_struc.view(x_struc.size(0), -1)

        out_struc = self.linear_struct(x_struc)
        out_merge = torch.cat((out, out_struc), dim=1)

        logits = self.fc(out_merge)
        return self.sf(logits)

class TextCNN(nn.Module):
    def __init__(self, num_class, dim, max_length=512):
        super(TextCNN, self).__init__()
        out_channel = 100
        filter_sizes = [1,2,3]
        self.convs = nn.ModuleList([ nn.Sequential(
                    nn.Conv2d(1, out_channel, (fs, dim)),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d((max_length-fs+1,1)),
        ) for fs in filter_sizes])
        self.fc = nn.Linear(out_channel * len(filter_sizes), num_class)
        self.sf = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        logits = self.fc(out) 
        return self.sf(logits)
    
    
    
    
    
