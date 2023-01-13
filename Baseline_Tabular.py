import pandas as pd
import os
import numpy as np
import random
import warnings
import re
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def check_exist(icd, list_prefix):
    flag = False
    for prefix in list_prefix:
        if str(icd).startswith(prefix):
            flag = True
        else:
            pass
    return flag

disease_list = ['Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']

metric_name = []
lr_list = []
NN_list = []
rf_list = []
GBC_list = []
MJ_list = []
LM_list = []
save_result = True
random_select = True
feature_selection = False
K_Fold = True

mimic_iii = True
disease_prefix_list = [
                ['diag_2900','diag_2901','diag_2902','diag_2903','diag_2904','diag_2910','diag_2911','diag_2912','diag_29282','diag_2948','diag_2941','diag_3310','diag_3311','diag_33182'],
                ['diag_052', 'diag_053'],
                ['diag_493','diag_2870','diag_37214','diag_477','diag_495','diag_691','diag_7080','diag_9953','diag_V14','diag_V150','diag_V196'],
                ['diag_7373', 'diag_75420', 'diag_75481'],
                ['diag_282'],
                ['diag_314'],
                ['diag_428']]
nondisease_prefix_list = [
                        [],
                        [],
                        ['diag_2734','diag_277','diag_415','diag_416','diag_4783','diag_4785','diag_492','diag_494','diag_4952','diag_496','diag_714','diag_7485','diag_769','diag_V813'],
                        [],
                        [],
                        [],
                        []]

if mimic_iii:
    mimic_root = '../mimic-iii-1.4'
    icd_key = 'ICD9_CODE'
    cpt_key = 'CPT_CD'
    sub_key = 'SUBJECT_ID'
    item_key = 'ITEMID'
    gsn_key = 'GSN'
    note_key = 'TEXT'
    med_key = 'DRUG'
    note_df = pd.read_csv(os.path.join(mimic_root, 'NOTEEVENTS.csv.gz'),
                          compression='gzip',
                          header=0,
                          error_bad_lines=False)
    diag_df = pd.read_csv(os.path.join(mimic_root, 'DIAGNOSES_ICD' + '.csv.gz'),
                          compression='gzip',
                          header=0,
                          error_bad_lines=False)
    cpt_df = pd.read_csv(os.path.join(mimic_root, 'CPTEVENTS' + '.csv.gz'),
                         compression='gzip',
                         header=0,
                         error_bad_lines=False)
    prep_df = pd.read_csv(os.path.join(mimic_root, 'PRESCRIPTIONS' + '.csv.gz'),
                          compression='gzip',
                          header=0,
                          error_bad_lines=False)
    col_list = [sub_key, item_key, 'FLAG']
    labReader = pd.read_csv(os.path.join(mimic_root, 'LABEVENTS.csv.gz'), chunksize=1000000,
                            usecols=col_list)  # the number of rows per chunk
    dfList = []
    for df in labReader:
        df = df.loc[df['FLAG'] == 'abnormal']
        dfList.append(df)
    lab_df = pd.concat(dfList, sort=False)
else:
    mimic_root = '../mimic-iv-1.0'
    icd_key = 'icd_code'
    cpt_key = 'hcpcs_cd'
    sub_key = 'subject_id'
    item_key = 'itemid'
    gsn_key = 'gsn'
    med_key = 'drug'
    diag_df = pd.read_csv(os.path.join(mimic_root, 'hosp', 'diagnoses_icd' + '.csv.gz'),
                          compression='gzip',
                          header=0,
                          error_bad_lines=False)
    cpt_df = pd.read_csv(os.path.join(mimic_root, 'hosp', 'hcpcsevents' + '.csv.gz'),
                         compression='gzip',
                         header=0,
                         error_bad_lines=False)
    prep_df = pd.read_csv(os.path.join(mimic_root, 'hosp', 'prescriptions' + '.csv.gz'),
                          compression='gzip',
                          header=0,
                          error_bad_lines=False)
    col_list = [sub_key, item_key, 'flag']
    labReader = pd.read_csv(os.path.join(mimic_root, 'hosp', 'labevents.csv.gz'), chunksize=1000000,
                            usecols=col_list)  # the number of rows per chunk
    dfList = []
    for df in labReader:
        df = df.loc[df['flag'] == 'abnormal']
        dfList.append(df)
    lab_df = pd.concat(dfList, sort=False)

diag_df[icd_key] = 'diag_' + diag_df[icd_key].str.strip()
prep_df[gsn_key] = 'gsn_' + prep_df[gsn_key].str.strip()
prep_df = prep_df[[sub_key, gsn_key, med_key]]

lab_df[item_key] = 'lab_' + lab_df[item_key].astype(str).str.strip()
lab_df = lab_df[[sub_key, item_key]]

cpt_df[cpt_key] = 'cpt_' + cpt_df[cpt_key].str.strip()
cpt_df = cpt_df[[sub_key, cpt_key]]

for disease, disease_prefix, nondisease_prefix in zip(disease_list, disease_prefix_list, nondisease_prefix_list):
    metric_name = metric_name + [disease+'_'+'Acc', disease+'_'+'Auc', disease+'_'+'Precision', disease+'_'+'F1', disease+'_'+'Recall']
    Disease_icd_in = np.unique(np.asarray([icd for icd in diag_df[icd_key] if check_exist(icd, disease_prefix)]))
    random.seed(33)
    Disease_icd_ex = np.unique(np.asarray([icd for icd in diag_df[icd_key] if check_exist(icd, nondisease_prefix)]))

    Disease_Patient_IDs_ = diag_df[diag_df[icd_key].isin(Disease_icd_in)][sub_key]
    Disease_Patient_IDs_.drop_duplicates(inplace=True)
    Disease_Patient_IDs_ = list(Disease_Patient_IDs_)

    Disease_Patient_EX = diag_df[diag_df[icd_key].isin(Disease_icd_ex)][sub_key]
    Disease_Patient_EX.drop_duplicates(inplace=True)
    Disease_Patient_EX = list(Disease_Patient_EX)
    Disease_Patient_IDs = set(Disease_Patient_IDs_).difference(Disease_Patient_EX)
    # Disease_Patient_IDs =  random.sample(Disease_Patient_IDs,4000)

    if random_select:
        NonDisease_Patient_IDs = diag_df[~diag_df[sub_key].isin(Disease_Patient_IDs)][sub_key]
        NonDisease_Patient_IDs.drop_duplicates(inplace=True)
        NonDisease_Patient_IDs = list(NonDisease_Patient_IDs)
        num_nondisease = len(Disease_Patient_IDs)
        NonDisease_Patient_IDs = random.sample(NonDisease_Patient_IDs, int(num_nondisease))
    else:
        NonDisease_Patient_IDs = Disease_Patient_EX
    # NonDisease_Patient_IDs = set(Disease_Patient_IDs_).intersection(Disease_Patient_EX)

    print("Disease_Patient_IDs", len(Disease_Patient_IDs))
    print("NonDisease_Patient_IDs", len(NonDisease_Patient_IDs))
    print("Intersection", len(set(Disease_Patient_IDs).intersection(NonDisease_Patient_IDs)))
    print("Difference", len(set(Disease_Patient_IDs).difference(NonDisease_Patient_IDs)))

    # ------------------------CPT--------------------------------
    Disease_CPT_Records = cpt_df[cpt_df[sub_key].isin(Disease_Patient_IDs)]
    Disease_CPT_Records = Disease_CPT_Records[[sub_key, cpt_key]]

    NonDisease_CPT_Records = cpt_df[cpt_df[sub_key].isin(NonDisease_Patient_IDs)]
    NonDisease_CPT_Records = NonDisease_CPT_Records[[sub_key, cpt_key]]

    df_data_cpt = pd.concat([Disease_CPT_Records, NonDisease_CPT_Records])
    df_data_cpt = df_data_cpt.dropna()
    df_data_cpt.drop_duplicates(inplace=True)

    df_data_cpt = df_data_cpt.groupby([sub_key])[cpt_key].apply(list)
    df_data_cpt = df_data_cpt.reset_index()
    # print(df_data_cpt)

    # --------------------------LAB---------------------------------
    Disease_lab_Records = lab_df[lab_df[sub_key].isin(Disease_Patient_IDs)]
    NonDisease_lab_Records = lab_df[lab_df[sub_key].isin(NonDisease_Patient_IDs)]
    df_data_lab = pd.concat([Disease_lab_Records, NonDisease_lab_Records])
    df_data_lab = df_data_lab.dropna()
    df_data_lab.drop_duplicates(inplace=True)

    df_data_lab = df_data_lab.groupby([sub_key])[item_key].apply(list)
    df_data_lab = df_data_lab.reset_index()
    # print(df_data_lab)

    # ---------------------------GSN---------------------------------
    Disease_gsn_Records = prep_df[prep_df[sub_key].isin(Disease_Patient_IDs)]
    NonDisease_gsn_Records = prep_df[prep_df[sub_key].isin(NonDisease_Patient_IDs)]

    df_data_gsn = pd.concat([Disease_gsn_Records, NonDisease_gsn_Records])
    df_data_gsn = df_data_gsn.dropna()
    df_data_gsn.drop_duplicates(inplace=True)

    df_data_gsn = df_data_gsn.groupby([sub_key])[gsn_key].apply(list)
    df_data_gsn = df_data_gsn.reset_index()

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb1 = MultiLabelBinarizer(sparse_output=True)
    df = df_data_cpt.copy(deep=True)
    df_input_cpt = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb1.fit_transform(df.pop(cpt_key)),
            index=df_data_cpt.index,
            columns=mlb1.classes_))

    # pca=PCA(n_components=256)
    # pca.fit(df_input_cpt[cpt_key])
    # df_input_cpt = pca.transform(df_input_cpt[cpt_key])
    # print(df_input_cpt.shape)

    # one hot embedding for gsn
    mlb2 = MultiLabelBinarizer(sparse_output=True)
    df = df_data_gsn.copy(deep=True)
    df_input_gsn = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb2.fit_transform(df.pop(gsn_key)),
            index=df_data_gsn.index,
            columns=mlb2.classes_))
    # lgbm_df_input_lab = lgbm_df_input_lab.drop(['subject_id'], axis=1)

    # one hot embedding for lab
    mlb3 = MultiLabelBinarizer(sparse_output=True)
    df = df_data_lab.copy(deep=True)
    df_input_lab = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb3.fit_transform(df.pop(item_key)),
            index=df_data_lab.index,
            columns=mlb3.classes_))

    print("Shape of cpt_df: " + str(df_input_cpt.shape))
    print("Shape of gsn_df: " + str(df_input_gsn.shape))
    print("Shape of lab_df: " + str(df_input_lab.shape))

    from snorkel.labeling import labeling_function, PandasLFApplier, LFApplier
    # logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from snorkel.labeling import LFAnalysis
    from snorkel.labeling.model import MajorityLabelVoter, LabelModel
    from sklearn import metrics

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import KFold


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

#x_struc = select_features_chi2(x_struc, y_tensor, 1200)

    lgbm_df_input_merge = df_input_cpt.merge(df_input_lab, on=sub_key, how='inner') \
        .merge(df_input_gsn, on=sub_key, how='inner')

    df_label = pd.DataFrame(
        {"label": list(np.zeros(len(lgbm_df_input_merge), dtype=int))}
    )

    lgbm_df_input_merge = pd.concat([lgbm_df_input_merge, df_label], axis=1)
    lgbm_df_input_merge.loc[lgbm_df_input_merge[sub_key].isin(Disease_Patient_IDs), 'label'] = 1

    x = lgbm_df_input_merge.drop([sub_key, 'label'], axis=1).to_numpy()
    y = lgbm_df_input_merge['label'].to_numpy()

    if feature_selection:
        x = select_features_chi2(x, y, 1200)

    if K_Fold:
        X = x
        kf = KFold()
        kf.get_n_splits(X)

        LR_li = []
        NN_li = []
        RF_li = []
        GBC_li = []
        MJ_li = []
        LM_li = []

        for train_index, test_index in kf.split(X):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr = LogisticRegression(random_state=0, max_iter=1000)
            NN = MLPClassifier([512, 256], learning_rate_init=0.0001, activation='relu', solver='adam', alpha=0.0001, max_iter=3000)
            rf = RandomForestClassifier(n_estimators=20)
            GBC = GradientBoostingClassifier(n_estimators=200)

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
            label_model = LabelModel(cardinality=2, device='cuda:0')
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
        label_model = LabelModel(cardinality=2, device='cuda:0')
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
    result_df.to_csv(os.path.join(path, 'K_Fold('+str(K_Fold)+')_'+'baseline_Tabular_result.csv'))
















