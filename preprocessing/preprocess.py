# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:25:54 2025

@author: Administrator
"""

import pandas as pd
import gzip
import json
import tqdm
import random
import os
import os
import numpy as np
import random
import argparse
import json
import warnings



def get_history(data, uid_set):
    pos_seq_dict = {}
    for uid in tqdm.tqdm(uid_set):
        pos = data[(data.uid == uid)&(data.y>3)].iid.values.tolist()
        pos_seq_dict[uid] = pos
    return pos_seq_dict

def preprocess(tgt, src, uid_dict, iid_dict_src, iid_dict_tgt, co_users, output_root):
    src.uid = src.uid.map(uid_dict)
    src.iid = src.iid.map(iid_dict_src)
    tgt.uid = tgt.uid.map(uid_dict)
    tgt.iid = tgt.iid.map(iid_dict_tgt)
    tgt_users = set(tgt.uid.unique())
    test_users = set(random.sample(co_users, round(0.2 * len(co_users))))
    train_src = src
    train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
    test = tgt[tgt['uid'].isin(test_users)]
    pos_seq_dict={}
    pos_seq_dict = get_history(src, co_users)
    train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
    train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
    test['pos_seq'] = test['uid'].map(pos_seq_dict)
    train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
    train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
    train_meta.to_csv(output_root + '/train_meta.csv', sep=',', header=None, index=False)
    test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)

def read_mid(root, field):
    path = root + field
    re = pd.read_csv(path)
    return re



output_root='<path/to/your/directory>/multidomain/_8_2/targetcd'

df_book = read_mid(root, field1)
df_cd = read_mid(root, field2)
df_mv = read_mid(root, field3)
df_el = read_mid(root, field4)

co_uid = set(df_book.uid) & set(df_cd.uid) & set(df_mv.uid) & set(df_el.uid)
all_uid = set(df_book.uid) | set(df_cd.uid) | set(df_mv.uid) | set(df_el.uid)
uid_dict = dict(zip(all_uid, range(len(all_uid))))
co_users = pd.DataFrame(co_uid).iloc[:, 0].map(uid_dict)
co_users = set(co_users)

len(co_uid)

tgt = df_cd
src_iid = set(df_mv.iid) | set(df_book.iid) | set(df_el.iid)
iid_dict_src = dict(zip(src_iid, range(len(set(src_iid)))))
iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src_iid)), len(set(src_iid)) + len(set(tgt.iid)))))
src = df_book.append(df_mv, ignore_index=True)
src = src.append(df_el, ignore_index=True)
src.isnull().value_counts()
preprocess(tgt, src, uid_dict, iid_dict_src, iid_dict_tgt, co_users, output_root)

