import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import KFold

torch.cuda.set_device(1)
# ['Atrial Fibrillation','Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']

disease_list = ['Dementia']
embedding = 'BlueBert'
Model_list = []
metric_list = []
size_list = []
save_model = False
save_result = True
feature_selection = False

path_model = './Model'
if not os.path.exists(path_model):
    os.mkdir(path_model)

for disease in disease_list:
    mode_list = [disease + '_BertCNN(Note)', disease + '_Merged_Model', disease + '_LR(Tabulated)',
                 disease + '_NN(Tabulated)']
    Model_list = Model_list + mode_list
    with open('../Data/data_merge_' + embedding + '_' + disease + '_random.pkl', 'rb') as file:
        subject_list = pickle.load(file)
        x_struc = pickle.load(file).astype('float32')
        x_tensor = pickle.load(file).astype('float32')
        y = pickle.load(file).astype('float32').reshape(-1, 1)
    size_list = size_list + [x_struc.shape[0]] * 4
    from sklearn import preprocessing

    one_hot = preprocessing.OneHotEncoder(sparse=False)
    y_tensor = one_hot.fit_transform(y)

    if embedding == 'BlueBert':
        x_tensor = x_tensor.reshape(x_tensor.shape[0], x_tensor.shape[2], -1)
    dim = x_tensor.shape[2]
    maxlen = x_tensor.shape[1]

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif


    def select_features_chi2(X_train, y_train, dim):
        fs = SelectKBest(score_func=chi2, k=dim)
        #        X_train = MinMaxScaler().fit_transform(X_train)
        fs.fit(X_train, np.argmax(y_train, axis=1))
        X_train_fs = fs.transform(X_train)
        return X_train_fs


    def select_features_mul(X_train, y_train, dim):
        fs = SelectKBest(score_func=mutual_info_classif, k=dim)
        fs.fit(X_train, np.argmax(y_train, axis=1))
        X_train_fs = fs.transform(X_train)
        return X_train_fs


    if feature_selection:
        x_struc = select_features_mul(x_struc, y_tensor, 1200)
    #    print(x_struc.shape)

    X = x_struc
    kf = KFold(n_splits=3)
    kf.get_n_splits(X)

    BertCNN = []
    Merge = []
    LR_tabular = []
    NN_tabular = []

    for train_index, test_index in kf.split(X):
        x_train, x_train_struc, x_test, x_test_struc = x_tensor[train_index], x_struc[train_index], x_tensor[test_index], x_struc[test_index],
        y_train, y_test = y_tensor[train_index], y_tensor[test_index]

        struc_size = x_test_struc.shape[1]

        epoch = 50
        if x_train.shape[0] > 512:
            bs = 64
        elif x_train.shape[0] > 128:
            bs = 64
        else:
            bs = 32
        Mode_list = [False, True]
        from EHR.code.train import train_Model, test_Model, test_MergeModel

        for Mode in Mode_list:
            if Mode:
                classifier, loss, valid_loss = train_Model(
                    train_data=x_train,  # 训练数据
                    train_struc=x_train_struc,
                    train_label=y_train,  # 训练标签
                    valid_data=x_test,
                    valid_struc=x_test_struc,
                    valid_label=y_test,
                    num_classes=2,  # 分类数量
                    num_epochs=epoch,  # 训练轮数
                    batch_size=bs,  # 批大小
                    learning_rate=0.001,  # 学习率
                    dim=dim,  # 每个特征的维度
                    dropout=0.4,  # transformer里的dropout
                    maxlen=maxlen,
                    model_type='BertCNN_Merge',
                    merge=True,
                    struc_size=struc_size,
                    root='./saved_model',
                )

                test_MergeModel(torch.tensor(x_train).cuda(), torch.tensor(x_train_struc).cuda(),
                                torch.tensor(y_train).cuda(),
                                classifier)
                Merge.append(test_MergeModel(torch.tensor(x_test).cuda(), torch.tensor(x_test_struc).cuda(),
                                                   torch.tensor(y_test).cuda(),
                                                   classifier))
                if save_model:
                    torch.save(classifier, os.path.join(path_model, embedding + mode_list[1] + '.pth'))
                del classifier
                torch.cuda.empty_cache()
            else:
                classifier, loss, valid_loss = train_Model(
                    train_data=x_train,  # 训练数据
                    #                 train_struc = x_train_struc,
                    train_label=y_train,  # 训练标签
                    valid_data=x_test,
                    #                 valid_struc = x_valid_struc,
                    valid_label=y_test,
                    num_classes=2,  # 分类数量
                    num_epochs=epoch,  # 训练轮数
                    batch_size=bs,  # 批大小
                    learning_rate=0.001,  # 学习率
                    dim=dim,  # 每个特征的维度
                    dropout=0.4,  # transformer里的dropout
                    maxlen=maxlen,
                    model_type='BertCNN',
                    merge=False,
                    #                 struc_size = struc_size,
                    root='./saved_model',
                )

                test_Model(torch.tensor(x_train).cuda(), torch.tensor(y_train).cuda(), classifier)
                BertCNN.append(test_Model(torch.tensor(x_test).cuda(), torch.tensor(y_test).cuda(), classifier))
                if save_model:
                    torch.save(classifier, os.path.join(path_model, embedding + mode_list[0] + '.pth'))
                del classifier
                torch.cuda.empty_cache()

        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics

        lr = LogisticRegression(random_state=0, max_iter=1000, solver='liblinear', class_weight="balanced")


        def train(model, x, y):
            model.fit(x, y)


        def predict(model, x, labels):
            predicted = model.predict(x)
            Acc = metrics.accuracy_score(labels, predicted)
            Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
            Auc = metrics.roc_auc_score(labels, predicted)
            F1 = metrics.f1_score(labels, predicted)
            Recall = metrics.recall_score(labels, predicted)
            return [Acc, Auc, Precision, F1, Recall]


        train(lr, x_train_struc, np.argmax(y_train, axis=1))
        print(predict(lr, x_train_struc, np.argmax(y_train, axis=1)))
        print(predict(lr, x_test_struc, np.argmax(y_test, axis=1)))
        LR_tabular.append(predict(lr, x_test_struc, np.argmax(y_test, axis=1)))

        if save_model:
            import pickle

            with open(os.path.join(path_model, embedding + mode_list[2] + '.pkl'), 'wb') as file:
                pickle.dump(lr, file)

        classifier, loss, valid_loss = train_Model(
            train_data=x_train_struc,  # 训练数据
            train_label=y_train,  # 训练标签
            valid_data=x_test_struc,
            valid_label=y_test,
            num_classes=2,  # 分类数量
            num_epochs=epoch,  # 训练轮数
            batch_size=bs,  # 批大小
            learning_rate=0.001,  # 学习率
            dim=x_train_struc.shape[1],  # 每个特征的维度
            model_type='TabulatedLinear',
            merge=False,
            root='./saved_model',
        )
        test_Model(torch.tensor(x_train_struc).cuda(), torch.tensor(y_train).cuda(), classifier)
        NN_tabular.append(test_Model(torch.tensor(x_test_struc).cuda(), torch.tensor(y_test).cuda(), classifier))
        if save_model:
            torch.save(classifier, os.path.join(path_model, embedding + mode_list[3] + '.pth'))
        del classifier
        torch.cuda.empty_cache()
    metric_list.append([format(np.mean(np.asarray(BertCNN)[:,i]), '.3f')+'±'+ format(np.std(np.asarray(BertCNN)[:,i]), '.3f') for i in range(5)])
    metric_list.append([format(np.mean(np.asarray(Merge)[:, i]), '.3f') + '±' + format(np.std(np.asarray(Merge)[:, i]), '.3f') for i in range(5)])
    metric_list.append([format(np.mean(np.asarray(LR_tabular)[:, i]), '.3f') + '±' + format(np.std(np.asarray(LR_tabular)[:, i]), '.3f') for i in range(5)])
    metric_list.append([format(np.mean(np.asarray(NN_tabular)[:, i]), '.3f') + '±' + format(np.std(np.asarray(NN_tabular)[:, i]), '.3f') for i in range(5)])

if save_result:
    result_dic = {'Size': size_list, 'Acc': list(np.array(metric_list)[:, 0]), 'Auc': list(np.array(metric_list)[:, 1]),
                  'Precision': list(np.array(metric_list)[:, 2]), 'F1': list(np.array(metric_list)[:, 3]),
                  'Recall': list(np.array(metric_list)[:, 4])}
    result_df = pd.DataFrame(result_dic, index=Model_list)
    path = './Result'
    if not os.path.exists(path):
        os.mkdir(path)
    result_df.to_csv(os.path.join(path, 'K_Fold_'+embedding+'_Result.csv'))
