import pickle
import pandas as pd
import torch
import numpy as np
import os
from snorkel.labeling import labeling_function, LFApplier
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from sklearn import metrics
from sklearn.model_selection import KFold

metric_name = []
lr_list = []
NN_list = []
rf_list = []
GBC_list = []
MJ_list = []
LM_list = []

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

path_model = './Model'
if not os.path.exists(path_model):
    os.mkdir(path_model)

for disease in disease_list:
    metric_name = metric_name + [disease + '_' + 'Acc', disease + '_' + 'Auc', disease + '_' + 'Precision',
                                 disease + '_' + 'F1', disease + '_' + 'Recall']
    mode_list = [disease + '_BertCNN(Note)', disease + '_Merged_Model', disease + '_LR(Tabulated)',
                 disease + '_NN(Tabulated)']
    Model_list = Model_list + mode_list
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
        x_tensor = select_features_chi2(x_tensor, y, 2048)

    if K_Fold:
        x = x_tensor
        kf = KFold()
        kf.get_n_splits(x)

        LR_li = []
        NN_li = []
        RF_li = []
        GBC_li = []
        MJ_li = []
        LM_li = []

        for train_index, test_index in kf.split(x):
            x_train, x_test = x_tensor[train_index], x_tensor[test_index]
            x_train_struc, x_test_struc = x_struc[train_index], x_struc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr = LogisticRegression(random_state=0, max_iter=500, solver='liblinear', class_weight="balanced")
            NN = MLPClassifier([512, 256], learning_rate_init=0.0001, activation='relu', solver='adam', alpha=0.0001, max_iter=500)
            rf = RandomForestClassifier(n_estimators=20)
            GBC = GradientBoostingClassifier(n_estimators=100)

            train(lr, x_train, y_train)
            train(NN, x_train, y_train)
            train(rf, x_train, y_train)
            train(GBC, x_train, y_train)

            LR_li.append(predict(lr, x_test, y_test))
            NN_li.append(predict(NN, x_test, y_test))
            RF_li.append(predict(rf, x_test, y_test))
            GBC_li.append(predict(GBC, x_test, y_test))

            @labeling_function()
            def label_by_logisticRegression(x):
                return lr.predict([x])[0]


            @labeling_function()
            def label_by_NN(x):
                return NN.predict([x])[0]


            @labeling_function()
            def label_by_randomforest(x):
                return rf.predict([x])[0]


            @labeling_function()
            def label_by_GBC(x):
                return GBC.predict([x])[0]

            lfs = [label_by_NN, label_by_logisticRegression, label_by_GBC]
            # Apply the LFs to the unlabeled training data
            applier = LFApplier(lfs=lfs)
            # test_data['subject_id'].reshape(-1, 1)
            # test_data = test_data.drop(['label'], axis=1)
            L_train = applier.apply(x_train)
            L_test = applier.apply(x_test)
            LFAnalysis(L=L_test, lfs=lfs).lf_summary()

            # take the majority vote
            majority_model = MajorityLabelVoter()
            label_model = LabelModel(cardinality=2, device='cuda:1')
            label_model.fit(L_train=L_train)

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

        lr_list = lr_list + [str(np.around(np.mean(np.asarray(LR_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(LR_li)[:, i]), 3)) for i in range(5)]
        NN_list = NN_list + [str(np.around(np.mean(np.asarray(NN_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(NN_li)[:, i]), 3)) for i in range(5)]
        rf_list = rf_list + [str(np.around(np.mean(np.asarray(RF_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(RF_li)[:, i]), 3)) for i in range(5)]
        GBC_list = GBC_list + [str(np.around(np.mean(np.asarray(GBC_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(GBC_li)[:, i]), 3)) for i in range(5)]
        MJ_list = MJ_list + [str(np.around(np.mean(np.asarray(MJ_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(MJ_li)[:, i]), 3)) for i in range(5)]
        LM_list = LM_list + [str(np.around(np.mean(np.asarray(LM_li)[:, i]), 3)) + '±' + str(
            np.around(np.std(np.asarray(LM_li)[:, i]), 3)) for i in range(5)]

    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

        lr = LogisticRegression(random_state=0, max_iter=1000)
        NN = MLPClassifier([512, 256], learning_rate_init=0.0001, activation='relu', solver='adam', alpha=0.0001, max_iter=3000)
        rf = RandomForestClassifier(n_estimators=20)
        GBC = GradientBoostingClassifier(n_estimators=200)

        # clf = SVC()

        train(lr, x_train, y_train)
        train(NN, x_train, y_train)
        train(rf, x_train, y_train)
        train(GBC, x_train, y_train)

        lr_list = lr_list + predict(lr, x_test, y_test)
        NN_list = NN_list + predict(NN, x_test, y_test)
        rf_list = rf_list + predict(rf, x_test, y_test)
        GBC_list = GBC_list + predict(GBC, x_test, y_test)

        @labeling_function()
        def label_by_logisticRegression(x):
            return lr.predict([x])[0]


        @labeling_function()
        def label_by_NN(x):
            return NN.predict([x])[0]


        @labeling_function()
        def label_by_randomforest(x):
            return rf.predict([x])[0]


        @labeling_function()
        def label_by_GBC(x):
            return GBC.predict([x])[0]


        # @labeling_function()
        # def label_by_SVM(x):
        #    return clf.predict([x])[0]

        lfs = [label_by_NN, label_by_logisticRegression, label_by_GBC]
        # Apply the LFs to the unlabeled training data
        applier = LFApplier(lfs=lfs)
        # test_data['subject_id'].reshape(-1, 1)
        # test_data = test_data.drop(['label'], axis=1)
        L_train = applier.apply(x_train)
        L_test = applier.apply(x_test)
        LFAnalysis(L=L_test, lfs=lfs).lf_summary()

        # take the majority vote
        majority_model = MajorityLabelVoter()
        label_model = LabelModel(cardinality=2, device='cuda:1')
        label_model.fit(L_train=L_train)

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

        MJ_list = MJ_list + return_metric(majority_model, L_test, y_test)
        LM_list = LM_list + return_metric(label_model, L_test, y_test)


if save_result:
    result_dic = {'lr': lr_list,'NN': NN_list,
                  'rf': rf_list,'GBC': GBC_list,
                  'MJ': MJ_list,'LM': LM_list}
    result_df = pd.DataFrame(result_dic, index=metric_name)
    path = './Result'
    if not os.path.exists(path):
        os.mkdir(path)
    result_df.to_csv(os.path.join(path, 'K_Fold('+str(K_Fold)+')_'+'baseline_Note_result.csv'))