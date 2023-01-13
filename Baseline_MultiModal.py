import pickle
import pandas as pd
import torch
import numpy as np
import os
from snorkel.labeling import labeling_function, LFApplier
# logistic regression
from sklearn.linear_model import LogisticRegression
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from sklearn import metrics
from sklearn.model_selection import KFold
import xgboost as xgb
from EHR.code.train import train_Model, test_Model
import random
import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(3407)


num_mode = 4
metric_name = []
lr_list = [[]]*num_mode
NN_list = [[]]*num_mode
GBC_list = [[]]*num_mode
GBC_1_list = [[]]*num_mode
MJ_list = [[]]*num_mode
LM_list = [[]]*num_mode
LM_lr_list = [[]]*num_mode
LM_NN_list = [[]]*num_mode

mode_list = [0,1,2,3]

def train(model, x, y):
    model.fit(x, y)


def predict(model, x, y):
    labels = y
    predicted = model.predict(x)
    Acc = metrics.accuracy_score(labels, predicted)
    Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
    Auc = metrics.roc_auc_score(labels, predicted)
    F1 = metrics.f1_score(labels, predicted)
    Recall = metrics.recall_score(labels, predicted)
    return [Acc, Auc, Precision, F1, Recall]

torch.cuda.set_device(1)
# ['Atrial Fibrillation','Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']

disease_list = ['Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']
embedding = 'Word2vec'
Model_list = []
metric_list = []
size_list = []
save_model = False
save_result = True
K_Fold = True
feature_selection = True
k_dim_note = 4096
k_dim_tabular = 1200

path = './Result'
if not os.path.exists(path):
    os.mkdir(path)
writer = pd.ExcelWriter(os.path.join(path, 'K_Fold('+str(K_Fold)+str(tuple([k_dim_note, k_dim_tabular]))+')_'+'baseline_MultiModal_Ensemble_result.xls'))

for disease in disease_list:
    metric_name = metric_name + [disease + '_' + 'Acc', disease + '_' + 'Auc', disease + '_' + 'Precision',
                                 disease + '_' + 'F1', disease + '_' + 'Recall']
    with open('../Data/data_merge_' + embedding + '_' + disease + '_random.pkl', 'rb') as file:
        subject_list = pickle.load(file)
        x_struc = pickle.load(file).astype('float32')
        x_tensor = pickle.load(file).astype('float32')
        y = pickle.load(file).astype('float32')

    print(disease+' : ', x_tensor.shape)

    if embedding == 'BlueBert':
        x_tensor = x_tensor.reshape(x_tensor.shape[0], x_tensor.shape[2], -1)
    x_tensor = x_tensor.reshape(x_tensor.shape[0], -1)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif
    from sklearn import preprocessing
    one_hot = preprocessing.OneHotEncoder(sparse=False)


    def select_features_chi2(X_train, y_train, dim):
        fs = SelectKBest(score_func=chi2, k=dim)
        #        X_train = MinMaxScaler().fit_transform(X_train)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        return X_train_fs


    def select_features_mul(X_train, y_train, dim):
        fs = SelectKBest(score_func=mutual_info_classif, k=dim)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        return X_train_fs

    if feature_selection:
        x_tensor = (x_tensor-np.min(x_tensor))/(np.max(x_tensor)-np.min(x_tensor))
        x_tensor = select_features_chi2(x_tensor, y, k_dim_note)
        x_struc = select_features_chi2(x_struc, y, k_dim_tabular)
    x_mul = np.concatenate([x_tensor, x_struc], axis=1)

    if K_Fold:
        for j in mode_list:
            x = x_tensor
            kf = KFold(n_splits=5)
            kf.get_n_splits(x)

            LR_li = []
            NN_li = []
    #        RF_li = []
            GBC_li = []
            GBC_1_li = []
            MJ_li = []
            LM_li = []
            LM_lr_li = []
            LM_NN_li = []

            for train_index, test_index in kf.split(x):
                x_train, x_test = x_tensor[train_index], x_tensor[test_index]
                x_train_struc, x_test_struc = x_struc[train_index], x_struc[test_index]
                x_train_mul, x_test_mul = x_mul[train_index], x_mul[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if j == 0:
                    x1, x2, x3 = x_train_struc, x_train_struc, x_train_struc
                    x1_test, x2_test, x3_test = x_test_struc, x_test_struc, x_test_struc
                elif j == 1:
                    x1, x2, x3 = x_train, x_train, x_train
                    x1_test, x2_test, x3_test = x_test, x_test, x_test
                elif j == 2:
                    x1, x2, x3 = x_train_mul, x_train_mul, x_train_mul
                    x1_test, x2_test, x3_test = x_test_mul, x_test_mul, x_test_mul
                elif j == 3:
                    x1, x2, x3 = x_train_struc, x_train, x_train_mul
                    x1_test, x2_test, x3_test = x_test_struc, x_test, x_test_mul
                else:
                    x1, x2, x3 = x_train_struc, x_train_struc, x_train_struc
                    x1_test, x2_test, x3_test = x_test_struc, x_test_struc, x_test_struc

                snorkel_train = [[a, b, c] for a, b, c in zip(x_train, x_train_struc, x_train_mul)]
                snorkel_test = [[a, b, c] for a, b, c in zip(x_test, x_test_struc, x_test_mul)]

                lr = LogisticRegression(random_state=0, max_iter=500, solver='liblinear', class_weight="balanced")
#                NN = MLPClassifier([512, 256], learning_rate_init=0.0001, activation='relu', solver='adam', alpha=0.0001, max_iter=500)
    #            rf = RandomForestClassifier(n_estimators=20)
    #            GBC = GradientBoostingClassifier(n_estimators=100)
                GBC = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=1, random_state=3407)

                train(lr, x1, y_train)
#                train(NN, x2, y_train)
                if j in [0,1,2]:
                    NN, loss, valid_loss = train_Model(
                        train_data=x2,  # 训练数据
                        train_label=one_hot.fit_transform(y_train.reshape(-1,1)),  # 训练标签
                        valid_data=x2_test,
                        valid_label=one_hot.fit_transform(y_test.reshape(-1,1)),
                        num_classes=2,  # 分类数量
                        num_epochs=300,  # 训练轮数
                        batch_size=64,  # 批大小
                        learning_rate=0.0001,  # 学习率
                        dim=x2.shape[1],  # 每个特征的维度
                        model_type='TabulatedLinear',
                        merge=False,
                        verbose=False,
                    )
                if j == 3:
                    GBC_1 = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=1, random_state=3407)
                    train(GBC_1, x2, y_train)
                    GBC_1_li.append(predict(GBC_1, x2_test, y_test))
                train(GBC, x3, y_train)


                LR_li.append(predict(lr, x1_test, y_test))
                if j in [0, 1, 2]:
                    NN_li.append(test_Model(torch.tensor(x2_test).cuda(), torch.tensor(one_hot.fit_transform(y_test.reshape(-1,1))).cuda(), NN))
    #            RF_li.append(predict(rf, x_test, y_test))
                GBC_li.append(predict(GBC, x3_test, y_test))

                @labeling_function()
                def label_by_lr(x):
                    if j == 0:
                        mode = 1
                    elif j == 1:
                        mode = 0
                    elif j == 2:
                        mode = 2
                    elif j == 3:
                        mode = 1
                    else:
                        mode = 1
                    return lr.predict([x[mode]])[0]

                @labeling_function()
                def label_by_NN(x):
                    if j == 0:
                        mode = 1
                    elif j == 1:
                        mode = 0
                    elif j == 2:
                        mode = 2
                    elif j == 3:
                        mode = 0
                    else:
                        mode = 1
                    return torch.argmax(NN(torch.tensor([x[mode]]).cuda()), dim=1).cpu().numpy()[0]

                @labeling_function()
                def label_by_GBC(x):
                    if j == 0:
                        mode = 1
                    elif j == 1:
                        mode = 0
                    elif j == 2:
                        mode = 2
                    elif j == 3:
                        mode = 2
                    else:
                        mode = 1
                    return GBC.predict([x[mode]])[0]


                @labeling_function()
                def label_by_GBC_1(x):
                    if j == 0:
                        mode = 1
                    elif j == 1:
                        mode = 0
                    elif j == 2:
                        mode = 2
                    elif j == 3:
                        mode = 0
                    else:
                        mode = 1
                    return GBC.predict([x[mode]])[0]

                if j in [0,1,2]:
                    lfs = [label_by_NN, label_by_lr, label_by_GBC]
                elif j in [3]:
                    lfs = [label_by_GBC, label_by_lr, label_by_GBC_1]
                else:
                    lfs = [label_by_NN, label_by_lr, label_by_GBC]
                # Apply the LFs to the unlabeled training data
                applier = LFApplier(lfs=lfs)
                # test_data['subject_id'].reshape(-1, 1)
                # test_data = test_data.drop(['label'], axis=1)
                L_train = applier.apply(snorkel_train)
                L_test = applier.apply(snorkel_test)
                LFAnalysis(L=L_test, lfs=lfs).lf_summary()

                # take the majority vote
                majority_model = MajorityLabelVoter()
                label_model = LabelModel(cardinality=2, device='cuda:1')
                label_model.fit(L_train=L_train)
#                lr_LM = LogisticRegression(random_state=0, max_iter=500, solver='liblinear', class_weight="balanced")
#                train(lr_LM,L_train,y_train)
#                NN_LM = MLPClassifier([512, 256], learning_rate_init=0.0001, activation='relu', solver='adam', alpha=0.0001, max_iter=500)
#                train(NN_LM, L_train, y_train)

                def return_metric(model, L_test, y):
                    x = L_test
                    labels = y
                    predicted = model.predict(L=x)
                    Acc = metrics.accuracy_score(labels, predicted)
                    Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
                    Auc = metrics.roc_auc_score(labels, predicted)
                    F1 = metrics.f1_score(labels, predicted)
                    Recall = metrics.recall_score(labels, predicted)
                    return [Acc, Auc, Precision, F1, Recall]

                MJ_li.append(return_metric(majority_model, L_test, y_test))
                LM_li.append(return_metric(label_model, L_test, y_test))
#                LM_lr_li.append(return_metric(lr_LM, L_test, y_test))
 #               LM_NN_li.append(return_metric(NN_LM, L_test, y_test))

            lr_list[j] = lr_list[j] + [format(np.mean(np.asarray(LR_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(LR_li)[:, i]), '.3f') for i in range(5)]
            if j in [0,1,2]:
                NN_list[j] = NN_list[j] + [format(np.mean(np.asarray(NN_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(NN_li)[:, i]), '.3f') for i in range(5)]
            GBC_list[j] = GBC_list[j] + [format(np.mean(np.asarray(GBC_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(GBC_li)[:, i]), '.3f') for i in range(5)]
            if j == 3:
                GBC_1_list[j] = GBC_1_list[j] + [format(np.mean(np.asarray(GBC_1_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(GBC_1_li)[:, i]), '.3f') for i in range(5)]
            MJ_list[j] = MJ_list[j] + [format(np.mean(np.asarray(MJ_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(MJ_li)[:, i]), '.3f') for i in range(5)]
            LM_list[j] = LM_list[j] + [format(np.mean(np.asarray(LM_li)[:, i]), '.3f') + '±' + format(np.std(np.asarray(LM_li)[:, i]), '.3f') for i in range(5)]
#            LM_lr_list[j] = LM_lr_list[j] + [str(np.around(np.mean(np.asarray(LM_lr_li)[:, i]), 3)) + '±' + str(
#                np.around(np.std(np.asarray(LM_lr_li)[:, i]), 3)) for i in range(5)]
#            LM_NN_list[j] = LM_NN_list[j] + [str(np.around(np.mean(np.asarray(LM_NN_li)[:, i]), 3)) + '±' + str(
#                np.around(np.std(np.asarray(LM_NN_li)[:, i]), 3)) for i in range(5)]


if save_result:
    for j in mode_list:
        if j == 0:
            result_dic_tabular = pd.DataFrame({'lr': lr_list[0],'NN': NN_list[0],
                          'GBC': GBC_list[0],
                          'MJ': MJ_list[0],'LM': LM_list[0]}, index=metric_name)
            result_dic_tabular.to_excel(excel_writer=writer, sheet_name='Tabular')
        elif j == 1:
            result_dic_note = pd.DataFrame({'lr': lr_list[1], 'NN': NN_list[1],
                                  'GBC': GBC_list[1],
                                  'MJ': MJ_list[1], 'LM': LM_list[1]}, index=metric_name)
            result_dic_note.to_excel(excel_writer=writer, sheet_name='Note')
        elif j == 2:
            result_dic_mul = pd.DataFrame({'lr': lr_list[2], 'NN': NN_list[2],
                                  'GBC': GBC_list[2],
                                  'MJ': MJ_list[2], 'LM': LM_list[2]}, index=metric_name)
            result_dic_mul.to_excel(excel_writer=writer, sheet_name='MultiModal')
        elif j == 3:
            result_dic_mul_em = pd.DataFrame({'lr(Tabular)': lr_list[3], 'GBC(Note)':GBC_1_list[3],
                                           'GBC(MultiModal)': GBC_list[3],
                                           'MJ': MJ_list[3], 'LM': LM_list[3]}, index=metric_name)
            result_dic_mul_em.to_excel(excel_writer=writer, sheet_name='MultiModal_Ensemble')
    writer.save()
    writer.close()