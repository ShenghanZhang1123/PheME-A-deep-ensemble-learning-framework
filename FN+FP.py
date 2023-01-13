import pandas as pd
import os
import numpy as np
import re
import random
from snorkel.labeling import labeling_function, LFApplier
from sklearn import metrics
import torch
import pickle
import warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(1)

#['Atrial Fibrillation','Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']
disease_list = ['Dementia']
embedding = 'BlueBert'
path_model = './Model'

mimic_iii = True
save_result = True
random_select = True
keyword = True

path = './Result'
if not os.path.exists(path):
    os.mkdir(path)
writer = pd.ExcelWriter(os.path.join(path, embedding + '_MergedModel' + '_FP_Top5.xls'))

def check_exist(icd, list_prefix):
    flag = False
    for prefix in list_prefix:
        if str(icd).startswith(prefix):
            flag = True
        else:
            pass
    return flag

# reg_all = re.compile(r'((^|[ \.])AFL($|[ \.]))|((^|[ \.])AF($|[ \.]))|(a\w*[ .]?(fib[\w]*|flutter))|(A\w*[ .]?(fib[\w]*|flutter))')
reg_pos_list = [re.compile(r'((^|[ .]?)mentia($|[ .]?))|((^|[ .]?)brain($|[ .]?))|((^|[ .]?)Alzheimer($|[ .]?))|((^|[ .]?)cognitive($|[ .]?))|((^|[ .]?)age($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)shingles($|[ .]?))|((^|[ .]?)zoster($|[ .]?))|((^|[ .]?)herpes($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)asthma($|[ .]?))|((^|[ .]?)chronic($|[ .]?))|((^|[ .]?)Inflammation($|[ .]?))|((^|[ .]?)lung($|[ .]?))|((^|[ .]?)cough($|[ .]?))|((^|[ .]?)wheeze($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Post($|[ .]?))|((^|[ .]?)Pain($|[ .]?))|((^|[ .]?)scoliosis($|[ .]?))|((^|[ .]?)pectus(.)excavatum($|[ .]?))|((^|[ .]?)sickle($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)sickle($|[ .]?))|((^|[ .]?)cell($|[ .]?))|((^|[ .]?)blood($|[ .]?))|((^|[ .]?)clog($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Attention($|[ .]?))|((^|[ .]?)deficit($|[ .]?))|((^|[ .]?)disorder($|[ .]?))|((^|[ .]?)neuro($|[ .]?))|((^|[ .]?)neuro($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Heart($|[ .]?))|((^|[ .]?)failure($|[ .]?))|((^|[ .]?)blood($|[ .]?))|((^|[ .]?)ventricular($|[ .]?))|((^|[ .]?)ejection($|[ .]?))', re.I)]

# reg_neg = re.compile(r'((^|[ \.])AFL($|[ \.]))|(a\w*[ .]?(flutter))|(A\w*[ .]?(flutter))')
note_cate = ['ECG', 'Nursing', 'Radiology']
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

if mimic_iii:
    #    note_df = note_df[note_df['CATEGORY'].isin(note_cate)]
    note_df = note_df[[sub_key, note_key, 'CATEGORY']]

for disease, disease_prefix, nondisease_prefix, reg_pos in zip(disease_list, disease_prefix_list, nondisease_prefix_list, reg_pos_list):
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

    if mimic_iii:
        threshold = [20, 10000]
        # Note
        Disease_Note_Records = note_df[note_df[sub_key].isin(Disease_Patient_IDs)]

        if keyword:
            #        Disease_Note_Records_list = [re.search(reg_pos,i) != None for i in Disease_Note_Records[note_key]]
            #        Disease_Note_Records = Disease_Note_Records[Disease_Note_Records_list]

            def keyword_extraction(x):
                sen_list = x.split('.')
                hit_list = []
                for i in range(len(sen_list)):
                    if re.search(reg_pos, sen_list[i]) != None:
                        try:
                            hit_list.append(sen_list[i - 1])
                        except:
                            pass
                        try:
                            hit_list.append(sen_list[i])
                        except:
                            pass
                        try:
                            hit_list.append(sen_list[i + 1])
                        except:
                            pass

                hit_list = np.unique(np.asarray(hit_list))
                if len(hit_list) != 0:
                    return ''.join(hit_list)
                else:
                    if len(sen_list) > 15:
                        hit_list = random.sample(sen_list, 15)
                    else:
                        hit_list = random.sample(sen_list, len(sen_list))
                    return ''.join(hit_list)


            Disease_Note_Records[note_key] = Disease_Note_Records[note_key].apply(keyword_extraction)
            Disease_Patient_IDs = np.unique(np.asarray(Disease_Note_Records[sub_key]))
        Disease_Note_Records = Disease_Note_Records.dropna()
        pos_list = [len(t.split()) >= threshold[0] for t in Disease_Note_Records[note_key]]
        Disease_Note_Records = Disease_Note_Records[pos_list]
        len_list = [len(t.split()) for t in Disease_Note_Records[note_key]]

        threshold[1] = max(len_list)

        num_nondi = Disease_Patient_IDs.shape[0]
        NonDisease_Patient_IDs = random.sample(NonDisease_Patient_IDs, int(num_nondi))
        NonDisease_Note_Records = note_df[note_df[sub_key].isin(NonDisease_Patient_IDs)]
        neg_list = [len(t.split()) >= threshold[0] and len(t.split()) <= threshold[1] for t in
                    NonDisease_Note_Records[note_key]]
        NonDisease_Note_Records = NonDisease_Note_Records[neg_list]
        df_data_note = pd.concat([Disease_Note_Records, NonDisease_Note_Records])
        df_data_note = df_data_note.dropna()
        df_data_note.drop_duplicates(inplace=True)
        df_data_note = df_data_note.groupby([sub_key])[note_key].apply(list)
        df_data_note = df_data_note.reset_index()

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

    # dignosis icd codes
    Disease_Patient_Records = diag_df[diag_df[sub_key].isin(Disease_Patient_IDs)]
    Disease_Patient_Records = Disease_Patient_Records[[sub_key, icd_key]]

    NonDisease_Patient_Records = diag_df[diag_df[sub_key].isin(NonDisease_Patient_IDs)]
    NonDisease_Patient_Records = NonDisease_Patient_Records[[sub_key, icd_key]]

    df_data_diag = pd.concat([Disease_Patient_Records, NonDisease_Patient_Records])
    df_data_diag = df_data_diag.dropna()
    df_data_diag.drop_duplicates(inplace=True)

    df_data_diag = df_data_diag.groupby([sub_key])[icd_key].apply(list)
    df_data_diag = df_data_diag.reset_index()
    #    print(df_data_diag)
    # ---------------------------DRUG---------------------------------
    Disease_med_Records = prep_df[prep_df[sub_key].isin(Disease_Patient_IDs)]
    NonDisease_med_Records = prep_df[prep_df[sub_key].isin(NonDisease_Patient_IDs)]

    df_data_med = pd.concat([Disease_med_Records, NonDisease_med_Records])
    df_data_med = df_data_med.dropna()
    df_data_med.drop_duplicates(inplace=True)

    df_data_med = df_data_med.groupby([sub_key])[med_key].apply(list)
    df_data_med = df_data_med.reset_index()

    df_input_merge = df_data_diag.merge(df_data_lab, on=sub_key, how='outer').merge(df_data_gsn, on=sub_key,how='outer').\
        merge(df_data_lab, on=sub_key,how='outer').merge(df_data_med, on=sub_key,how='outer')\
        .merge(df_data_note, on=sub_key,how='outer').fillna(0)

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

    print(disease+" : "+str(x_tensor.shape[1]*512)+' / '+str(x_struc.shape[1]))

    mode_list = [disease + '_BertCNN(Note)', disease + '_Merged_Model', disease + '_LR(Tabulated)',
                 disease + '_NN(Tabulated)']

    Merge = torch.load(os.path.join(path_model, embedding + mode_list[1] + '.pth')).cuda()
    prob = Merge(torch.tensor(x_tensor).cuda(), torch.tensor(x_struc).cuda()).cpu()
    pred = torch.argmax(prob, dim=1).numpy()
    pos_prob = prob[:,1].detach().numpy()

    labels = np.argmax(y_tensor, axis=1)

    FP_list = []
    for index in range(len(pred)):
        if int(pred[index]) == 1 and int(labels[index]) == 0:
            FP_list.append([subject_list[index], pos_prob[index]])
    FP_list = sorted(FP_list, key = lambda x: x[1], reverse=True)
    Top5 = []
    for i in range(5):
        try:
            Top5.append(FP_list[i][0])
        except:
            Top5.append(-1)

    df_out = df_input_merge[df_input_merge[sub_key].isin(Top5)].astype(str)

    if save_result:
        path = './Result'
        if not os.path.exists(path):
            os.mkdir(path)
        df_out.to_excel(excel_writer=writer,sheet_name=disease)

writer.save()
writer.close()