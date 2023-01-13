# -*- coding: utf-8 -*-
"""
@author: shenghan
"""
import torch
import torch.nn as nn
from . import allmodel
import torch.utils.data as Data
import math
import copy
import os
from sklearn import metrics

def train_Model(
        train_data,  
        train_label, 
        valid_data,
        valid_label,
        num_classes, 
        num_epochs,  
        batch_size, 
        learning_rate,  
        dim=1, 
        dropout=0., 
        maxlen=512,
        model_type='lstm',
        merge = False,
        struc_size = 1024,
        train_struc=None,
        valid_struc=None,
        root = os.getcwd(),
        verbose = True,
):
    torch_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    loader = Data.DataLoader( 
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,  
#        num_workers=2, 
        drop_last=False
    )

    if model_type == 'lstm':
        model = allmodel.lstm(num_classes, dim).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)
    elif model_type == 'CNN':
        model = allmodel.TextCNN(num_classes, dim, maxlen).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)
    elif model_type == 'BertCNN':
        model = allmodel.BertCNN(num_classes, dim, maxlen).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)
    elif model_type == 'BertLinear':
        model = allmodel.BertLinear(num_classes, dim, maxlen).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)

    elif model_type == 'TabulatedLinear':
        model = allmodel.TabulatedLinear(num_classes, dim).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=10000,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)

    elif model_type == 'BertLinear_Merge':
        torch_dataset1 = Data.TensorDataset(torch.from_numpy(train_struc), torch.from_numpy(train_label))
        loader1 = Data.DataLoader(
            dataset=torch_dataset1,
            batch_size=batch_size,
            shuffle=False,  
#            num_workers=2,
            drop_last=False
        )
        model = allmodel.TextLinear_Merge(num_classes, dim, struc_size, maxlen).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)
    elif model_type == 'BertCNN_Merge':
        torch_dataset1 = Data.TensorDataset(torch.from_numpy(train_struc), torch.from_numpy(train_label))
        loader1 = Data.DataLoader( 
            dataset=torch_dataset1,
            batch_size=batch_size,
            shuffle=False, 
#            num_workers=2, 
            drop_last=False
        )
        model = allmodel.TextCNN_Merge(num_classes, dim, struc_size, maxlen).cuda()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)
    else:
        model = allmodel.lstm(num_classes, dim).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=70,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-7, eps=1e-08)

    print(model)
    model.train()


    min_loss_val = 10
    # Train
    best_model = None
    loss_list = []
    valid_loss_list = []
    if merge:
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            epoch_loss = 0
            loss_ = 0
            for idx, data in enumerate(zip(loader, loader1), 0):
                x = data[0][0].cuda()
                x_struc = data[1][0].cuda()
                target = data[0][1].cuda()
                predict = model(x, x_struc)
                #            losses.append(loss)
                correct += int(torch.sum(torch.argmax(predict, dim=1) == torch.argmax(target, dim=1)))
                total += len(target)
                optimizer.zero_grad()
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_loss += loss.item()
                loss_ = epoch_loss / (idx + 1)
                if idx % (len(loader) // 2) == 0 and verbose:
                    print("Epoch={}/{},{}/{}of train, loss={}, acc={}, lr={}".format(
                        epoch + 1, num_epochs, idx, len(loader), loss_, correct / total,
                        optimizer.state_dict()['param_groups'][0]['lr']))
                del x, x_struc, data, target, predict
                torch.cuda.empty_cache()
            loss_list.append(loss_)
            with torch.no_grad():
                pred = model(torch.tensor(valid_data).cuda(), torch.tensor(valid_struc).cuda())
                y = torch.tensor(valid_label).cuda()
                valid_loss = criterion(pred, y)
                valid_loss_list.append(valid_loss.item())
#            if valid_loss <= min_loss_val and epoch > 30:
#                min_loss_val = valid_loss
#                torch.save(model, os.path.join(root,'best_model.pth'))
        print("Done")
    else:
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            epoch_loss = 0
            loss_ = 0
            for idx, (x, target) in enumerate(loader, 0):
                x = x.cuda()
                target = target.cuda()
                predict = model(x)
                #            losses.append(loss)
                correct += int(torch.sum(torch.argmax(predict, dim=1) == torch.argmax(target, dim=1)))
                total += len(target)
                optimizer.zero_grad()
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                epoch_loss += loss.item()
                loss_ = epoch_loss / (idx + 1)
                if idx % (len(loader) // 2) == 0 and verbose:
                    print("Epoch={}/{},{}/{}of train, loss={}, acc={}, lr={}".format(
                        epoch + 1, num_epochs, idx, len(loader), loss_, correct/total,optimizer.state_dict()['param_groups'][0]['lr']))
                del x, target, predict
                torch.cuda.empty_cache()
            loss_list.append(loss_)
            with torch.no_grad():
                pred = model(torch.tensor(valid_data).cuda())
                y = torch.tensor(valid_label).cuda()
                valid_loss = criterion(pred, y)
                valid_loss_list.append(valid_loss.item())
#            if valid_loss <= min_loss_val and epoch > 30:
#                min_loss_val = valid_loss
#                torch.save(model, os.path.join(root, 'best_model.pth'))
        print("Done")
        # Save Model

    return model, loss_list, valid_loss_list

def test_Model(test_data, test_label,model):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy().astype('int64')
        labels = torch.argmax(test_label, dim=1).cpu().numpy().astype('int64')
        Acc = metrics.accuracy_score(labels, predicted)
        Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
        Auc = metrics.roc_auc_score(labels, predicted)
        F1 = metrics.f1_score(labels, predicted)
        Recall = metrics.recall_score(labels, predicted)
#        print("label:",labels)
#        print("prediction:", predicted)
        print('Acc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc,Auc,Precision, F1,Recall))
    return [Acc,Auc,Precision, F1,Recall]

def test_MergeModel(test_data, test_struc, test_label,model):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data, test_struc)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy().astype('int64')
        labels = torch.argmax(test_label, dim=1).cpu().numpy().astype('int64')
        Acc = metrics.accuracy_score(labels, predicted)
        Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
        Auc = metrics.roc_auc_score(labels, predicted)
        F1 = metrics.f1_score(labels, predicted)
        Recall = metrics.recall_score(labels, predicted)
#        print("label:",labels)
#        print("prediction:", predicted)
        print('Acc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc,Auc,Precision, F1,Recall))
    return [Acc, Auc, Precision, F1, Recall]
