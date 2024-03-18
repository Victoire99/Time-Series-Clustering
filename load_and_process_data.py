'''
Author: KEWEI ZHANG
Date: 2024-03-05 15:19:49
LastEditors: KEWEI ZHANG
LastEditTime: 2024-03-11 10:53:19
FilePath: \WorkNote\cluster\load_and_process_data.py
Description: 

'''
import pandas as pd
import numpy as np
import os
import operator
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "C:/WorkNote/data/"

def cleanxls2dct(filename):
    '''netting duration xls to dictionary'''
    df = pd.read_csv(filename, encoding='gb18030',header=0, index_col=0)

    ts = np.array(df.columns[1:])
    cus = df['Customer'].tolist()
    df.drop(['Customer'], axis=1,inplace=True)
    matrx = df.values
    dct = {}
    dct['time'] = ts
    for i in range(len(cus)):
        dct[cus[i]] = matrx[i]

    #np.savetxt('output.txt', dct, fmt='%f')
    return dct

def xls2dct(filename):
    
    df = pd.read_excel(INPUT_PATH + filename, header=1)

    # Filter rows where 'Company ID' contains 'PRD' but not 'DIS'
    df = df[df['Company ID'].str.contains('PRD') & ~df['Company ID'].str.contains('DIS')]
    df = df.fillna(0)
    
    if filename == 'Part Count.xlsx':
        # Group by 'Customer', 'Company ID' and 'Server' and sum the values for rows where 'Unnamed:3' is 'MPS', 'MRP' or 'MPS Config'
        df = df[df['Unnamed: 3'].isin(['MPS', 'MRP', 'MPS Config'])]
        df = df.groupby(['Customer', 'Company ID', 'Server']).sum().reset_index()

    # Create a new column by concatenating 'Customer', 'Company ID' and 'Server'
    df['name'] = df['Customer'] +" / " +  df['Company ID'] + " / " + df['Server']
    # Select columns from '20230701' to '20231231' and 'name'
    cols = ['name'] + df.loc[:, '20230701':'20231231'].columns.to_list()
    df = df[cols]
    

    ts = np.array(df.columns[1:])
    cus = df['name'].tolist()

    df = df.drop(['name'], axis=1)
    matrx = df.values
    dct = {}
    dct['time'] = ts
    for i in range(len(cus)):
        dct[cus[i]] = matrx[i]

    
    return dct

def dctGenerator(path = 'C:/WorkNote/data'):

    item = os.listdir(path)    
    if not os.path.exists('C:/WorkNote/output'):
        os.makedirs('C:/WorkNote/output')


    xlsx_list = []
    for i in range(len(item)):
        if operator.contains(item[i], "xlsx"):
            xlsx_list.append(item[i])

    temp_dct = {}
    for i in range(len(xlsx_list)):
        dct = xls2dct(xlsx_list[i])
        name = xlsx_list[i][:-5]
        temp_dct[name] = dct
    file_list = list(temp_dct.keys())

    new_dct = {}

    for xlsx in temp_dct:
        for customer in temp_dct[xlsx]:
            if customer not in new_dct:
                new_dct[customer] = {}
            new_dct[customer][xlsx] = temp_dct[xlsx][customer]

    customer_list = list(new_dct.keys())
    customer_list = customer_list[1:]

    # time series duplicate check
    new_dct['time'] = temp_dct[xlsx_list[0][:-5]]['time']

    return new_dct, temp_dct, file_list, customer_list


def calculate_scores(data):
    # 计算每个键的得分
    scores = {}
    for key, value in data.items():
        sil_score = value['sil']
        cal_score = value['cal']
        dav_score = 1 / value['dav']  # dav越小越好，所以我们取倒数
        total_score = sil_score + cal_score + dav_score
        scores[key] = total_score

    # 归一化得分
    min_score = min(scores.values())
    max_score = max(scores.values())
    for key in scores:
        scores[key] = (scores[key] - min_score) / (max_score - min_score)

    # 找到每个类别中得分最高的键
    best_keys = {}
    for key, score in scores.items():
        group_key = key[:-2]
        if group_key not in best_keys or score > scores[best_keys[group_key]]:
            best_keys[group_key] = key

    return scores,best_keys
