import pandas as pd
import os
import numpy as np
from snorkel.labeling import labeling_function, LFApplier
from sklearn import metrics
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
import torch
from sklearn.model_selection import KFold
import pickle
import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(0)
# ['Atrial Fibrillation','Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']
disease_list = ['HF']
embedding = 'BlueBert'
Model_list = []
metric_list = []
size_list = []
save_result = True
K_Fold = True
voting_method_list = ['MajorityVoting','LabelModel']
mode = 1

path_model = './Model'

for disease in disease_list:
    with open('../Data/data_merge_'+embedding+'_'+disease+'_random.pkl', 'rb') as file:
        subject_list = pickle.load(file)
        x_struc = pickle.load(file).astype('float32')
        x_tensor = pickle.load(file).astype('float32')
        y = pickle.load(file).astype('float32').reshape(-1, 1)
    from sklearn import preprocessing

    one_hot = preprocessing.OneHotEncoder(sparse=False)
    y_tensor = one_hot.fit_transform(y)
    if embedding == 'BlueBert':
        x_tensor = x_tensor.reshape(x_tensor.shape[0],x_tensor.shape[2],-1)
    dim = x_tensor.shape[2]
    maxlen = x_tensor.shape[1]

    mode_list = [disease + '_BertCNN(Note)', disease + '_Merged_Model', disease + '_LR(Tabulated)', disease + '_NN(Tabulated)']

    Bert = torch.load(os.path.join(path_model, embedding + mode_list[0] + '.pth')).cuda()
    Merge = torch.load(os.path.join(path_model, embedding + mode_list[1] + '.pth')).cuda()

    with open(os.path.join(path_model, embedding + mode_list[2] + '.pkl'), 'rb') as file:
        lr = pickle.load(file)

    NN = torch.load(os.path.join(path_model, embedding + mode_list[3] + '.pth')).cuda()

    @labeling_function()
    def label_by_lr(x):
        return lr.predict([x[1]])[0]

    @labeling_function()
    def label_by_NN(x):
        return torch.argmax(NN(torch.tensor([x[1]]).cuda()), dim=1).cpu().numpy()[0]

    @labeling_function()
    def label_by_Bert(x):
        return torch.argmax(Bert(torch.tensor([x[0]]).cuda()), dim=1).cpu().numpy()[0]

    @labeling_function()
    def label_by_Merge(x):
        return torch.argmax(Merge(torch.tensor([x[0]]).cuda(), torch.tensor([x[1]]).cuda()), dim=1).cpu().numpy()[0]

    # @labeling_function()
    # def label_by_SVM(x):
    #    return clf.predict([x])[0]

    if K_Fold:
        X = x_struc
        kf = KFold()
        kf.get_n_splits(X)

        voter_list = [disease + '(NN+Bert+Merge)', disease + '(lr+Bert+Merge)', disease + '(NN+lr+Merge)',
                      disease + '(NN+Bert+lr)']
        size_list = size_list + [x_struc.shape[0]] * 4
        lfs = [label_by_lr, label_by_NN, label_by_Bert, label_by_Merge]
        Model_list = Model_list + voter_list
        Snorkel_list = [[] for i in range(len(voter_list))]

        for train_index, test_index in kf.split(X):
            x_train, x_train_struc, x_test, x_test_struc = x_tensor[train_index], x_struc[train_index], x_tensor[
                test_index], x_struc[test_index],
            y_train, y_test = y_tensor[train_index], y_tensor[test_index]

            struc_size = x_test_struc.shape[1]

            snorkel_train = [[a, b] for a, b in zip(x_train, x_train_struc)]
            snorkel_test = [[a, b] for a, b in zip(x_test, x_test_struc)]

            # Apply the LFs to the unlabeled training data
            voting = voting_method_list[mode]
            applier = LFApplier(lfs=lfs)
            # test_data[sub_key].reshape(-1, 1)
            # test_data = test_data.drop(['label'], axis=1)
            L_train = applier.apply(snorkel_train)
            L_test = applier.apply(snorkel_test)
            LFAnalysis(L=L_test, lfs=lfs).lf_summary()
            for index in range(len(voter_list)):
                col_list = list(range(len(voter_list)))
                col_list.remove(index)
                # take the majority vote
                if voting == 'MajorityVoting':
                    majority_model = MajorityLabelVoter()
                    preds_test = majority_model.predict(L=L_test[:, col_list])
                elif voting == 'LabelModel':
                    label_model = LabelModel()
                    label_model.fit(L_train=L_train[:, col_list])
                    preds_test = label_model.predict(L_test[:, col_list])

                labels = np.argmax(y_test, axis=1)
                predicted = preds_test
                Acc = metrics.accuracy_score(labels, predicted)
                Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
                Auc = metrics.roc_auc_score(labels, predicted)
                F1 = metrics.f1_score(labels, predicted)
                Recall = metrics.recall_score(labels, predicted)
                print('\nAcc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc, Auc, Precision, F1, Recall))
                Snorkel_list[index].append([Acc, Auc, Precision, F1, Recall])
        for a in range(len(Snorkel_list)):
            metric_list.append([format(np.mean(np.asarray(Snorkel_list[a])[:,i]), '.3f')+'±'+format(np.std(np.asarray(Snorkel_list[a])[:,i]), '.3f') for i in range(5)])

    else:
        from sklearn.model_selection import train_test_split

        x_train, x_1, x_train_struc, x_2, y_train, y_1 = train_test_split(x_tensor, x_struc, y_tensor, test_size=0.4,
                                                                          random_state=33)
        x_valid, x_test, x_valid_struc, x_test_struc, y_valid, y_test = train_test_split(x_1, x_2, y_1, test_size=0.5,
                                                                                         random_state=33)

        snorkel_train = [ [a,b] for a, b in zip(x_train, x_train_struc)]
        snorkel_test = [[a, b] for a, b in zip(x_test, x_test_struc)]

        lfs = [label_by_lr, label_by_NN, label_by_Bert, label_by_Merge]
        voter_list = [disease + '(NN+Bert+Merge)', disease + '(lr+Bert+Merge)', disease + '(NN+lr+Merge)', disease + '(NN+Bert+lr)']
        Model_list = Model_list + voter_list
        size_list = size_list + [x_struc.shape[0]] * 4
        # Apply the LFs to the unlabeled training data
        voting = voting_method_list[mode]
        applier = LFApplier(lfs=lfs)
        # test_data[sub_key].reshape(-1, 1)
        # test_data = test_data.drop(['label'], axis=1)
        L_train = applier.apply(snorkel_train)
        L_test = applier.apply(snorkel_test)
        LFAnalysis(L=L_test, lfs=lfs).lf_summary()
        for index in range(len(voter_list)):
            col_list = list(range(len(voter_list)))
            col_list.remove(index)
            # take the majority vote
            if voting == 'MajorityVoting':
                majority_model = MajorityLabelVoter()
                preds_test = majority_model.predict(L=L_test[:,col_list])
            elif voting == 'LabelModel':
                label_model = LabelModel()
                label_model.fit(L_train=L_train[:,col_list])
                preds_test = label_model.predict(L_test[:,col_list])

            labels = np.argmax(y_test,axis=1)
            predicted = preds_test
            Acc = metrics.accuracy_score(labels, predicted)
            Precision = metrics.precision_score(y_true=labels, y_pred=predicted, zero_division=0)
            Auc = metrics.roc_auc_score(labels, predicted)
            F1 = metrics.f1_score(labels, predicted)
            Recall = metrics.recall_score(labels, predicted)
            print('\nAcc: {} , Auc: {} , Pre: {} , F1: {} , Recall: {} '.format(Acc, Auc, Precision, F1, Recall))
            metric_list.append([Acc,Auc,Precision, F1,Recall])

if save_result:
    result_dic = {'Size': size_list, 'Acc': list(np.array(metric_list)[:,0]),'Auc': list(np.array(metric_list)[:,1]),
                  'Precision': list(np.array(metric_list)[:,2]),'F1': list(np.array(metric_list)[:,3]),
                  'Recall': list(np.array(metric_list)[:,4])}
    result_df = pd.DataFrame(result_dic, index=Model_list)
    path = './Result'
    if not os.path.exists(path):
        os.mkdir(path)
    result_df.to_csv(os.path.join(path, 'K_Fold('+str(K_Fold)+')_'+embedding+'_result_snorkel_'+voting_method_list[mode]+'.csv'))
