import pandas as pd
import os
import numpy as np
import random
import warnings
import re
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

save_data = False
keyword = True
random_select = True
model_type = 'CNN'
embedding_type = 'Word2vec'


def check_exist(icd, list_prefix):
    flag = False
    for prefix in list_prefix:
        if str(icd).startswith(prefix):
            flag = True
        else:
            pass
    return flag


disease_list = ['Dementia','HerpesZoster','Asthma','PostEventPain','SickleCell', 'ADHD', 'HF']
# reg_all = re.compile(r'((^|[ \.])AFL($|[ \.]))|((^|[ \.])AF($|[ \.]))|(a\w*[ .]?(fib[\w]*|flutter))|(A\w*[ .]?(fib[\w]*|flutter))')
reg_pos_list = [re.compile(r'((^|[ .]?)mentia($|[ .]?))|((^|[ .]?)brain($|[ .]?))|((^|[ .]?)Alzheimer($|[ .]?))|((^|[ .]?)cognitive($|[ .]?))|((^|[ .]?)age($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)shingles($|[ .]?))|((^|[ .]?)zoster($|[ .]?))|((^|[ .]?)herpes($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)asthma($|[ .]?))|((^|[ .]?)chronic($|[ .]?))|((^|[ .]?)Inflammation($|[ .]?))|((^|[ .]?)lung($|[ .]?))|((^|[ .]?)cough($|[ .]?))|((^|[ .]?)wheeze($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Post($|[ .]?))|((^|[ .]?)Pain($|[ .]?))|((^|[ .]?)scoliosis($|[ .]?))|((^|[ .]?)pectus(.)excavatum($|[ .]?))|((^|[ .]?)sickle($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)sickle($|[ .]?))|((^|[ .]?)cell($|[ .]?))|((^|[ .]?)blood($|[ .]?))|((^|[ .]?)clog($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Attention($|[ .]?))|((^|[ .]?)deficit($|[ .]?))|((^|[ .]?)disorder($|[ .]?))|((^|[ .]?)neuro($|[ .]?))|((^|[ .]?)neuro($|[ .]?))', re.I),
                re.compile(r'((^|[ .]?)Heart($|[ .]?))|((^|[ .]?)failure($|[ .]?))|((^|[ .]?)blood($|[ .]?))|((^|[ .]?)ventricular($|[ .]?))|((^|[ .]?)ejection($|[ .]?))', re.I)]

# reg_neg = re.compile(r'((^|[ \.])AFL($|[ \.]))|(a\w*[ .]?(flutter))|(A\w*[ .]?(flutter))')
mimic_iii = True
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
        #    Disease_Note_Records = note_df[note_df[sub_key].isin(Disease_Patient_IDs)]
        pos_list = [len(t.split()) >= threshold[0] for t in Disease_Note_Records[note_key]]
        Disease_Note_Records = Disease_Note_Records[pos_list]
        len_list = [len(t.split()) for t in Disease_Note_Records[note_key]]
        n, bins, patches = plt.hist(len_list, 100)
        plt.show()
        print("Average Length: ", int(np.mean(np.array(len_list))))
        print("Maximum Length: ", np.max(np.array(len_list)))
        print("Minimum Length: ", np.min(np.array(len_list)))

        threshold[1] = max(len_list)

        num_nondi = Disease_Patient_IDs.shape[0]
        NonDisease_Patient_IDs = random.sample(NonDisease_Patient_IDs, int(num_nondi))
        NonDisease_Note_Records = note_df[note_df[sub_key].isin(NonDisease_Patient_IDs)]
        #    str_lis = [t for t in NonDisease_Note_Records[note_key] if len(t) < 50]
        neg_list = [len(t.split()) >= threshold[0] and len(t.split()) <= threshold[1] for t in
                    NonDisease_Note_Records[note_key]]
        NonDisease_Note_Records = NonDisease_Note_Records[neg_list]
        len_list = [len(t.split()) for t in NonDisease_Note_Records[note_key]]
        n, bins, patches = plt.hist(len_list, 100)
        plt.show()
        print("Average Length: ", int(np.mean(np.array(len_list))))
        print("Maximum Length: ", np.max(np.array(len_list)))
        print("Minimum Length: ", np.min(np.array(len_list)))
        #    NonDisease_Note_Records_list = [re.search(reg_neg, i) != None for i in NonDisease_Note_Records[note_key]]
        #    NonDisease_Note_Records = NonDisease_Note_Records[NonDisease_Note_Records_list]
        df_data_note = pd.concat([Disease_Note_Records, NonDisease_Note_Records])
        df_data_note = df_data_note.dropna()
        df_data_note.drop_duplicates(inplace=True)
        #    df_data_note_list = [len(i.split()) > 50 for i in list(df_data_note[note_key])]
        #    df_data_note = df_data_note[df_data_note_list]
        if embedding_type == 'Bert' or 'BlueBert':
            remove_chars = re.compile('[·.!"#$%&\'()＃！（）*+,/:;<=>?@，：￥★、…．＞\[\]【】《》？“”‘’^_`{|}~]+')
            df_data_note[note_key] = df_data_note[note_key].apply(
                lambda x: re.sub(remove_chars, " ", x).replace('\n', ' ').lower() + " [SEP]")

        if model_type == 'CNN':
            df_data_note = df_data_note.groupby([sub_key])[note_key].apply(list)
            df_data_note = df_data_note.reset_index()
            if embedding_type == 'Bert' or 'BlueBert':
                df_data_note[note_key] = df_data_note[note_key].apply(lambda x: '[CLS] ' + ' '.join(x))
        elif model_type == 'lstm':
            df_data_note = df_data_note.reset_index()
            df_data_note = df_data_note.drop('index', axis=1)
            pass

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
    '''
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
    '''
    from sklearn.preprocessing import MultiLabelBinarizer

    '''
    mlb0 = MultiLabelBinarizer(sparse_output=True)
    df = df_data_diag.copy(deep=True)
    df_input_diag = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb0.fit_transform(df.pop(icd_key)),
            index=df_data_diag.index,
            columns=mlb0.classes_))
    
    mlb4 = MultiLabelBinarizer(sparse_output=True)
    df = df_data_med.copy(deep=True)
    df_input_med = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb4.fit_transform(df.pop(med_key)),
            index=df_data_med.index,
            columns=mlb4.classes_))
    
    print("Shape of diag_df: " + str(df_input_diag.shape))
    print("Shape of lab_df: " + str(df_input_med.shape))
    '''
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

    df_input_merge = df_input_cpt.merge(df_input_lab, on=sub_key, how='outer').merge(df_input_gsn, on=sub_key, how='outer').fillna(0)
    df_input_merge['combined'] = df_input_merge.drop([sub_key], axis=1).values.tolist()
    df_input_merge = df_input_merge[[sub_key, 'combined']]

    import torch.nn.functional
    import numpy as np
    import pandas as pd
    from gensim.models.word2vec import Word2Vec
    from transformers import AutoModel, AlbertTokenizer, BertModel, BertTokenizer

    # stopwords = [line.strip() for line in open('./stopwords-en/stopwords-en.txt', 'r', encoding='utf-8').readlines()]

    '''
    def del_stop_words(text):
        word_ls = [i for i in text if i not in stopwords]
        return word_ls
    
    def count(words):
        dic = {}
        for word in words:
            dic[word] = dic.get(word, 0) + 1
        return dic
    '''

    if embedding_type == 'Word2vec':
        model = Word2Vec(list(df_data_note[note_key]), window=10, vector_size=150)
        model = model.wv
        remove_chars = re.compile('[·.!"#$%&\'()＃！（）*+,/:;<=>?@，：￥★、…．＞【】\[\]《》？“”‘’^`{|}~]+')
        threshold_ = [0, 30000]
        print('Start tokenizing.')
        df_data_note[note_key] = df_data_note[note_key].apply(
            lambda x: re.sub(remove_chars, " ", x).replace('\n', ' ').lower().split()[:threshold_[1]])
        print('End of tokenizing.')

    elif embedding_type == 'Albert':
        #    vocab = [line.strip() for line in open('./bert-base-uncased/vocab.txt', 'r', encoding='utf-8').readlines()]
        vocab_path = './albert_pytorch-master/prev_trained_model/albert_base_v2_/30k-clean.model'
        model_path = './albert_pytorch-master/prev_trained_model/albert_base_v2_'
        tokenizer = AlbertTokenizer.from_pretrained(vocab_path)
        bert_model = AutoModel.from_pretrained(model_path)
        threshold_ = [0, 30000]
        #    thres_list = [len(t.split()) >= threshold_[0] and len(t.split()) <= threshold_[1] for t in df_data_note[note_key]]
        #    df_data_note = df_data_note[thres_list]
        #    df_data_note = df_data_note.reset_index()
        #    df_data_note = df_data_note.drop('index',axis=1)
        #    config_path = './bert-base-uncased/config.json'
        df_data_note[note_key] = df_data_note[note_key].apply(lambda x: ' '.join(x.split()[:threshold_[1]]))
        print('Start tokenizing.')
        df_data_note[note_key] = df_data_note[note_key].apply(
            lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
        print('End of tokenizing.')
    elif embedding_type == 'Bert':
        #    vocab = [line.strip() for line in open('./bert-base-uncased/vocab.txt', 'r', encoding='utf-8').readlines()]
        model_path = './bert-base-uncased/'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = AutoModel.from_pretrained(model_path)
        threshold_ = [0, 30000]
        #    thres_list = [len(t.split()) >= threshold_[0] and len(t.split()) <= threshold_[1] for t in df_data_note[note_key]]
        #    df_data_note = df_data_note[thres_list]
        #    df_data_note = df_data_note.reset_index()
        #    df_data_note = df_data_note.drop('index',axis=1)
        #    config_path = './bert-base-uncased/config.json'
        df_data_note[note_key] = df_data_note[note_key].apply(lambda x: ' '.join(x.split()[:threshold_[1]]))
        print('Start tokenizing.')
        df_data_note[note_key] = df_data_note[note_key].apply(
            lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
        print('End of tokenizing.')
    elif embedding_type == 'BlueBert':
        #    vocab = [line.strip() for line in open('./bert-base-uncased/vocab.txt', 'r', encoding='utf-8').readlines()]
        model_path = '../BlueBert/'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertModel.from_pretrained(model_path)
        posi_len = [len(df_data_note[note_key][t].split()) for t in range(len(df_data_note[note_key])) if
                    df_data_note[sub_key][t] in Disease_Patient_IDs]
        threshold_ = [0, 30000]
        #    threshold_ = [0,max(posi_len)]
        #    thres_list = [len(t.split()) >= threshold_[0] and len(t.split()) <= threshold_[1] for t in df_data_note[note_key]]
        #    df_data_note = df_data_note[thres_list]
        #    df_data_note = df_data_note.reset_index()
        #    df_data_note = df_data_note.drop('index',axis=1)
        df_data_note[note_key] = df_data_note[note_key].apply(lambda x: ' '.join(x.split()[:threshold_[1]]))
        #    config_path = './bert-base-uncased/config.json'
        print('Start tokenizing.')
        df_data_note[note_key] = df_data_note[note_key].apply(
            lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
        print('End of tokenizing.')


    print('Start embedding.')
    if embedding_type == 'Word2vec':
        for i in range(len(df_data_note[note_key])):
            li = []
            for word in df_data_note[note_key][i]:
                try:
                    li.append(model[word])
                except:
                    pass
            df_data_note[note_key][i] = li

        maxlen = 512
        average = int(np.mean(np.asarray([len(t) for t in df_data_note[note_key]])))
        # maxlen = int(max([len(t) for t in df_data_merge['Vector']]))
        dim = model.vector_size

        for i in range(len(df_data_note[note_key])):
            if len(df_data_note[note_key][i]) < maxlen:
                df_data_note[note_key][i] = np.asarray(
                    df_data_note[note_key][i] + [np.zeros(dim)] * (maxlen - len(df_data_note[note_key][i])))
            else:
                chunk_list = []
                for j in range(int(len(df_data_note[note_key][i]) / maxlen) + 1):
                    if len(df_data_note[note_key][i][j * maxlen:(j + 1) * maxlen]) == maxlen:
                        chunk_list.append(np.asarray(df_data_note[note_key][i][j * maxlen:(j + 1) * maxlen]))
                    else:
                        chunk_list.append(np.asarray(
                            df_data_note[note_key][i][j * maxlen:(j + 1) * maxlen] + [np.zeros(dim)] * (
                                        maxlen - len(df_data_note[note_key][i][j * maxlen:(j + 1) * maxlen]))))
                df_data_note[note_key][i] = np.asarray(sum(chunk_list) / len(chunk_list))

    elif embedding_type == 'Bert' or 'BlueBert':
        bert_model.cuda()
        maxlen = 512
        maxsen = max(
            [int(len(t) / maxlen) + 1 if len(t) % maxlen != 0 else int(len(t) / maxlen) for t in df_data_note[note_key]])


        def id2chunk(x):
            if len(x) < maxlen:
                return [np.asarray(x + [0] * (maxlen - len(x)))] + [np.zeros(maxlen).astype('int64')] * (maxsen - 1)
            else:
                chunk_list = []
                for j in range(int(len(x) / maxlen) + 1):
                    if len(x[j * maxlen:(j + 1) * maxlen]) == maxlen:
                        chunk_list.append(np.asarray(x[j * maxlen:(j + 1) * maxlen]))
                    else:
                        chunk_list.append(np.asarray(x[j * maxlen:(j + 1) * maxlen] + [0] * (
                                maxlen - len(x[j * maxlen:(j + 1) * maxlen]))))
                return chunk_list + [np.zeros(maxlen).astype('int64')] * (maxsen - len(chunk_list))


        df_data_note[note_key] = df_data_note[note_key].apply(id2chunk)


        def chunks2vec(x):
            return bert_model(torch.tensor(x, dtype=torch.int64).cuda())[1].detach().cpu().numpy().reshape(-1, maxsen)


        def chunk2vec(x):
            maxl = int(len(x) / 4)
            bert_list = []
            for i in range(int(len(x) / maxl) + 1):
                try:
                    bert_list += list(bert_model(torch.tensor(x[maxl * i:maxl * (i + 1)], dtype=torch.int64).cuda())[
                                          1].detach().cpu().numpy())
                except:
                    pass
            return np.asarray(bert_list).reshape(-1, maxsen)


        start = time.time()
        df_data_note[note_key] = df_data_note[note_key].apply(chunk2vec)
        end = time.time()
        print("Time consumed : %.2f s" % (end - start))

        dim = maxsen
        maxlen = 768
    print('End of embedding.')

    df_merge = df_input_merge.merge(df_data_note, on=sub_key, how='outer')

    df_label = pd.DataFrame(
        {"label": list(np.zeros(len(df_merge), dtype=int))}
    )
    df_merge = pd.concat([df_merge, df_label], axis=1)
    df_merge.loc[df_merge[sub_key].isin(Disease_Patient_IDs), 'label'] = 1
    df_merge.loc[df_merge[sub_key].isin(NonDisease_Patient_IDs), 'label'] = 0

    df_merge['combined'].fillna(value=-1, inplace=True)
    df_merge['combined'] = df_merge['combined'].apply(
        lambda x: [float(0)] * len(df_input_merge['combined'][0]) if type(x) == int else x)

    df_merge[note_key].fillna(value=-1, inplace=True)
    df_merge[note_key] = df_merge[note_key].apply(
        lambda x: np.zeros((df_data_note[note_key][0].shape[0], df_data_note[note_key][0].shape[1])) if type(
            x) == int else x)

    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import numpy as np

    if save_data:
        with open('../Data/data_merge_Word2vec_'+disease+'_random.pkl', 'wb') as file:
            pickle.dump(np.asarray(list(df_merge[sub_key])).astype('int64'), file, protocol = 4)
            pickle.dump(np.asarray(list(df_merge['combined'])).astype('float32'), file, protocol = 4)
            pickle.dump(np.asarray(list(df_merge[note_key])).astype('float32'), file, protocol = 4)
            pickle.dump(np.asarray(list(df_merge['label'])).astype('float32'), file, protocol = 4)




